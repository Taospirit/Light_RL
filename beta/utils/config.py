config = {
    'ddpg': # copy from test_ddpg
    {
        'env_name': 'Pendulum-v0',
        'buffer_size': 50000,
        'actor_learn_freq': 1,
        'update_iteration': 10,
        'target_update_freq': 10,
        'target_update_tau': 1e-1, # to large
        'lr': 3e-3,
        'batch_size': 128,
        'hidden_dim': 32,
        'episodes': 2000,
        'max_step': 300,
        'SAVE_DIR': '/save/ddpg_',
        'PKL_DIR': '/pkl/ddpg_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'DDPG_',
    },
    'td3':
    {
        'env_name': 'Pendulum-v0',
        'buffer_size': 50000,
        'actor_learn_freq': 2,
        'update_iteration': 10,
        'target_update_freq': 10,
        'target_update_tau': 1e-1, # to large
        'lr': 3e-3,
        'batch_size': 128,
        'hidden_dim': 32,
        'episodes': 2000,
        'max_step': 300,
        'SAVE_DIR': '/save/td3_',
        'PKL_DIR': '/pkl/td3_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'TD3_',
    },
    'sac':
    {
        'env_name': 'Pendulum-v0',
        'buffer_size': 50000,
        'actor_learn_freq': 2,
        'update_iteration': 10,
        'target_update_freq': 10,
        'target_update_tau': 1e-2,
        'lr': 3e-3,
        'batch_size': 128,
        'hidden_dim': 32,
        'episodes': 2000,
        'max_step': 300,
        'SAVE_DIR': '/save/sac_',
        'PKL_DIR': '/pkl/sac_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'SAC_',
    },
    'sacv':
    {
        'env_name': 'HalfCheetah-v2',
        'buffer_size': int(1e6),
        'actor_learn_freq': 2,
        'update_iteration': 10,
        'target_update_freq': 10,
        'lr': 1e-4, # notice
        'batch_size': 256,
        'hidden_dim': 400,
        'episodes': int(1e6),
        'max_step': 300,
        'tau': 0.005,
        'SAVE_DIR': '/save/sacv_',
        'PKL_DIR': '/pkl/sacv_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'SACV_',
    }

}