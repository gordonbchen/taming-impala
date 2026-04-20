class DistSettings:
    # Network.
    HOST = "127.0.0.1"
    PORT = 6767

    # Env.
    ENV_NAME = "Pong-v5"
    N_FRAME_STACK = 4
    OBS_SHAPE = (N_FRAME_STACK, 84, 84)
