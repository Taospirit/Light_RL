from gym import envs
# import gym_rocketlander

envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)
