import socket
import json
import base64
import numpy as np


class MessageType:
    GET_WEIGHTS = 0
    WEIGHTS = 1
    ROLLOUT = 2


HEADER_LEN = 8


def send_msg(sock: socket.socket, msg: dict):
    msg_bytes = json.dumps(msg).encode("utf-8")
    header = len(msg_bytes).to_bytes(HEADER_LEN, byteorder="big", signed=False)
    sock.sendall(header + msg_bytes)


def recv_msg(sock: socket.socket):
    header = read_socket(sock, HEADER_LEN)
    msg_len = int.from_bytes(header, byteorder="big", signed=False)
    msg_bytes = read_socket(sock, msg_len)
    return json.loads(msg_bytes.decode("utf-8"))


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
