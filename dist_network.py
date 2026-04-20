import socket
import json
import base64
import numpy as np


HEADER_LEN = 8


def send_msg(sock: socket.socket, msg: dict):
    msg_bytes = serialize(msg)
    header = len(msg_bytes).to_bytes(HEADER_LEN, byteorder="big", signed=False)
    sock.sendall(header + msg_bytes)


def serialize(msg: dict) -> bytes:
    for k, v in msg:
        if isinstance(v, np.ndarray):
            msg[k] = {"type": "np.ndarray", "dtype": np.dtype(v.dtype).name, "shape": list(v.shape),
                          "data": base64.b64encode(v.tobytes())}
    return json.dumps(msg).encode("utf-8")


def recv_msg(sock: socket.socket):
    header = read_socket(sock, HEADER_LEN)
    msg_len = int.from_bytes(header, byteorder="big", signed=False)
    msg_bytes = read_socket(sock, msg_len)
    return deserialize(msg_bytes)


def read_socket(sock: socket.socket, n_bytes: int) -> bytes:
    chunks = []
    while n_bytes > 0:
        chunk = sock.recv(n_bytes)
        if len(chunk) == 0:
            raise ConnectionError("Socket CLOSED, failed to read.")
        chunks.append(chunk)
        n_bytes -= len(chunk)
    return b"".join(chunks)


def deserialize(msg_bytes: bytes) -> dict:
    payload = json.loads(msg_bytes)
    for k, v in payload:
        if isinstance(v, dict) and v.get("type", None) == "np.ndarray":
            payload[k] = np.frombuffer(base64.b64decode(v["data"]), dtype=v["dtype"]).reshape(v["shape"])
    return payload
