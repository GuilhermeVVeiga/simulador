from gymnasium.envs.registration import register

register(
    id='gym_env/InvertedPendulum-v22',
    entry_point='gym_env.envs:InvertedPendulum',
    kwargs={'render_mode': 'human','xRef': 0.5},
)

register(
    id='gym_env/Elevator-v29',
    entry_point='gym_env.envs:Elevator',
    kwargs={'render_mode': 'human','xRef': 5},
)


