import torch, os
import numpy as np
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasePolicy(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.buffer = None
    
    # @abstractmethod
    def learn(self):
        raise NotImplementedError

    def action(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def process(self, **kwargs):
        self.buffer.append(**kwargs)

    def copy_net(self, model):
        model_ = deepcopy(model)
        model_.load_state_dict(model.state_dict())
        return model_.eval()

    # def choose_action(self, state, eval=False):
    #     state = torch.tensor(state, dtype=torch.float32, device=device)
    #     if eval:
    #         self.actor_eval.eval()
    #     # return self.actor_eval.action(state, eval)
    #     return self.action(state, eval)

    def warm_up(self, warm_size=0):
        # for off-policy algo, warm up is better
        if warm_size:
            return len(self.buffer) < warm_size
        return not self.buffer.is_full()

    def save_model(self, save_dir, save_file, save_step, save_actor=False, save_critic=False):
        assert isinstance(save_dir, str) and isinstance(save_file, str)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_file)
        actor_save = '_'.join((save_path, save_step, 'actor.pth'))
        critic_save = '_'.join((save_path, save_step, 'critic.pth'))

        if save_actor:
            torch.save(self.actor_eval.state_dict(), actor_save)
            print (f'Save actor model in {actor_save}')
        if save_critic:
            torch.save(self.critic_eval.state_dict(), critic_save)
            print (f'Save critic model in {critic_save}')

    def load_model(self, save_dir, load_actor=False, load_critic=False):
        load_file = None
        load_step = -1
        for f in os.listdir(save_dir):
            if 'actor.pth' in f: 
                load_file = save_dir + '/' + f
                step_num = int(f.split('_')[-2])
                if step_num > load_step:
                    load_step = step_num
                    load_file = save_dir + '/' + f

        assert load_file, f'No model file to load in {save_dir}'
        actor_save = load_file
        critic_save = '_'.join(load_file.split('_')[:-1]) + '_critic.pth'

        if load_actor:
            assert os.path.exists(actor_save), f'No {actor_save} file to load'
            self.actor_eval.load_state_dict(torch.load(actor_save))
            print (f'Loading actor model success in {actor_save}!')
        if load_critic:
            assert os.path.exists(critic_save), f'No {critic_save} file to load'
            self.critic_eval.load_state_dict(torch.load(critic_save))
            print (f'Loading critic model success in {critic_save}!')

    @staticmethod
    def soft_sync_weight(target, source, tau=0.01):
        with torch.no_grad():
            for target_param, eval_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def GAE(rewards, v_evals, next_v_eval=0, masks=None, gamma=0.99, lam=1): # [r1, r2, ..., rT], [V1, V2, ... ,VT, VT+1]
        r'''
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

        :param gamma: (float) Discount factor
        :param lam: (float) GAE factor
        : GAE(lam=0) = td_error; GAE(lam=1) = MC
        '''
        assert isinstance(rewards, np.ndarray), 'rewards must be np.ndarray'
        assert isinstance(v_evals, np.ndarray), 'v_evals must be np.ndarray'
        assert len(rewards) == len(v_evals), 'V_pred length must equal rewards length'

        rew_len = len(rewards)
        masks = np.ones(rew_len) if masks is None else masks # nonterminal
        v_evals = np.append(v_evals, next_v_eval)
        adv_gae = np.empty(rew_len, 'float32')
        last_gae = 0
        for i in reversed(range(rew_len)):
            delta = rewards[i] + gamma * v_evals[i+1] * masks[i] - v_evals[i]
            adv_gae[i] = last_gae = delta + gamma * lam * masks[i] * last_gae

        return adv_gae