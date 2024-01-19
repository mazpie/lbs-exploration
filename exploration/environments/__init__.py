from gym.envs.registration import registry, register, make, spec

register(
    id='MountainCarSparse-v0',
    entry_point='exploration.environments.mountain_car_sparse:Continuous_MountainCarEnv',
    max_episode_steps=100,
)

register(
    id='MagellanAnt-v2',
    entry_point='exploration.environments.magellan_ant:MagellanAntEnv',
    max_episode_steps=300
)

register(
    id='HalfCheetahSparse-v3',
    entry_point='exploration.environments.half_cheetah_sparse:HalfCheetahEnv',
    max_episode_steps=500,
)