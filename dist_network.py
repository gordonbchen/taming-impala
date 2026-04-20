import socket
import json
import base64
import numpy as np


class MessageType:
    GET_WEIGHTS = 0
    WEIGHTS = 1
    ROLLOUT = 2
    ACK = 3


PROTOCOL_MAGIC = b"DISTRL"
MSG_SIZE_LEN = 8


def send_msg(sock: socket.socket, msg: dict) -> int:
    msg_bytes = json.dumps(msg).encode("utf-8")
    msg_size = len(msg_bytes).to_bytes(MSG_SIZE_LEN, byteorder="big", signed=False)
    full_msg = PROTOCOL_MAGIC + msg_size + msg_bytes
    sock.sendall(full_msg)
    return len(full_msg)


def recv_msg(sock: socket.socket) -> tuple[dict, int]:
    magic = read_socket(sock, len(PROTOCOL_MAGIC))
    if magic != PROTOCOL_MAGIC:
        raise ConnectionError(f"Protocol magic does not match: {magic}")
    msg_size = read_socket(sock, MSG_SIZE_LEN)
    msg_size = int.from_bytes(msg_size, byteorder="big", signed=False)
    msg_bytes = read_socket(sock, msg_size)
    total_size = len(PROTOCOL_MAGIC) + MSG_SIZE_LEN + msg_size
    return json.loads(msg_bytes.decode("utf-8")), total_size


def read_socket(sock: socket.socket, n_bytes: int) -> bytes:
    chunks = []
    while n_bytes > 0:
        chunk = sock.recv(n_bytes)
        if len(chunk) == 0:
            raise ConnectionError("Socket CLOSED, failed to read.")
        chunks.append(chunk)
        n_bytes -= len(chunk)
    return b"".join(chunks)


# TODO: more efficient serialization.
def serialize_np(arr: np.ndarray) -> dict:
    return {"type": "np.ndarray", "dtype": np.dtype(arr.dtype).name, "shape": list(arr.shape),
            "data": base64.b64encode(arr.tobytes()).decode("utf-8")}


def deserialize_np(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["data"]), dtype=d["dtype"]).reshape(d["shape"])
