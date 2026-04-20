class DistSettings:
    # Network.
    LISTEN_HOST = "0.0.0.0"
    LEARNER_HOST = "100.101.69.103"
    PORT = 6767

    # Env.
    ENV_NAME = "Pong-v5"
    N_FRAME_STACK = 4
    OBS_SHAPE = (N_FRAME_STACK, 84, 84)
