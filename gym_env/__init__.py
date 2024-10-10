"""__init__.py for LEO Environment."""
from gymnasium.envs.registration import register


########################################################################################
# Custom Environment
register(
    id='LEOSat-v0',
    entry_point='gym_env.leosat.leosat_env:LEOSatEnv',
    # kwargs={'row': 1, 'col': 1},
    max_episode_steps=50
)

register(
    id='RealWorld-v0',
    entry_point='gym_env.realworld.realworld_env:RealWorldEnv',
    max_episode_steps=50
)

register(
    id='DigitalWorld-v0',
    entry_point='gym_env.digitalworld.digitalworld_env:DigitalWorldEnv',
    max_episode_steps=50
)
