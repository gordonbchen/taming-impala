import socket
import struct
import lz4.frame
import numpy as np


class MessageType:
    GET_WEIGHTS = 0
    WEIGHTS = 1
    ROLLOUT = 2
    ACK = 3


PROTOCOL_MAGIC = b"DISTRL"
MSG_SIZE_LEN = 8
FLAG_COMPRESSED = 1
ROLLOUT_ARRAY_KEYS = ("obss", "dones", "actions", "rewards", "old_log_probs")


class ByteReader:
    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.offset = 0

    def take(self, n_bytes: int) -> memoryview:
        out = self.data[self.offset:self.offset + n_bytes]
        self.offset += n_bytes
        if len(out) != n_bytes:
            raise ConnectionError("Received truncated message.")
        return out

    def unpack(self, fmt: str):
        value, = struct.unpack(fmt, self.take(struct.calcsize(fmt)))
        return value

    def take_str(self) -> str:
        return self.take(self.unpack("!H")).tobytes().decode("utf-8")


def send_msg(sock: socket.socket, msg: dict) -> int:
    msg_bytes = encode_msg(msg)
    sock.sendall(msg_bytes)
    return len(msg_bytes)


def recv_msg(sock: socket.socket) -> tuple[dict, int]:
    magic = read_socket(sock, len(PROTOCOL_MAGIC))
    if magic != PROTOCOL_MAGIC:
        raise ConnectionError(f"Protocol magic does not match: {magic}")

    msg_type = read_socket(sock, 1)[0]
    total_size = len(PROTOCOL_MAGIC) + 1
    if msg_type == MessageType.GET_WEIGHTS:
        policy_version = struct.unpack("!q", read_socket(sock, 8))[0]
        return {"type": msg_type, "policy_version": policy_version}, total_size + 8
    if msg_type == MessageType.ACK:
        return {"type": msg_type}, total_size

    flags = read_socket(sock, 1)[0]
    payload_size = int.from_bytes(read_socket(sock, MSG_SIZE_LEN), byteorder="big", signed=False)
    payload = read_socket(sock, payload_size)
    total_size += 1 + MSG_SIZE_LEN + payload_size
    if flags & FLAG_COMPRESSED:
        payload = lz4.frame.decompress(payload)
    return decode_payload(msg_type, payload), total_size


def encode_msg(msg: dict) -> bytes:
    msg_type = msg["type"]
    header = bytearray(PROTOCOL_MAGIC)
    header.append(msg_type)
    if msg_type == MessageType.GET_WEIGHTS:
        header.extend(struct.pack("!q", msg["policy_version"]))
        return bytes(header)
    if msg_type == MessageType.ACK:
        return bytes(header)
    payload = encode_payload(msg_type, msg)

    compressed = lz4.frame.compress(payload)
    flags = 0
    if len(compressed) < len(payload):
        payload = compressed
        flags = FLAG_COMPRESSED
    header.append(flags)
    header.extend(len(payload).to_bytes(MSG_SIZE_LEN, byteorder="big", signed=False))
    return bytes(header) + payload


def encode_payload(msg_type: int, msg: dict) -> bytes:
    out = bytearray()
    if msg_type == MessageType.WEIGHTS:
        state_dict = {} if msg.get("state_dict") is None else msg["state_dict"]
        out.extend(struct.pack("!qI", msg["policy_version"], len(state_dict)))
        for name, arr in state_dict.items():
            append_str(out, name)
            append_array(out, arr)
        return bytes(out)
    if msg_type == MessageType.ROLLOUT:
        append_str(out, msg["actor_id"])
        out.extend(struct.pack("!qdI", msg["policy_version"], msg["total_reward"], msg["n_episodes"]))
        for key in ROLLOUT_ARRAY_KEYS:
            append_array(out, msg[key])
        return bytes(out)
    raise ValueError(f"Unknown message type: {msg_type}.")


def decode_payload(msg_type: int, payload: bytes) -> dict:
    reader = ByteReader(payload)
    if msg_type == MessageType.WEIGHTS:
        state_dict = {}
        msg = {"type": msg_type, "policy_version": reader.unpack("!q"), "state_dict": state_dict}
        for _ in range(reader.unpack("!I")):
            name = reader.take_str()
            state_dict[name] = take_array(reader)
        return msg
    if msg_type == MessageType.ROLLOUT:
        msg = {
            "type": msg_type,
            "actor_id": reader.take_str(),
            "policy_version": reader.unpack("!q"),
            "total_reward": reader.unpack("!d"),
            "n_episodes": reader.unpack("!I"),
        }
        for key in ROLLOUT_ARRAY_KEYS:
            msg[key] = take_array(reader)
        return msg
    raise ValueError(f"Unknown message type: {msg_type}.")


def append_array(out: bytearray, arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    dtype_bytes = arr.dtype.str.encode("ascii")
    out.extend(struct.pack("!H", len(dtype_bytes)))
    out.extend(dtype_bytes)
    out.extend(struct.pack("!H", arr.ndim))
    for dim in arr.shape:
        out.extend(struct.pack("!I", dim))
    out.extend(memoryview(arr).cast("B"))


def take_array(reader: ByteReader) -> np.ndarray:
    dtype = np.dtype(reader.take(reader.unpack("!H")).tobytes().decode("ascii"))
    ndim = reader.unpack("!H")
    shape = tuple(reader.unpack("!I") for _ in range(ndim))
    n_bytes = dtype.itemsize
    for dim in shape:
        n_bytes *= dim
    return np.frombuffer(reader.take(n_bytes), dtype=dtype).reshape(shape)


def append_str(out: bytearray, s: str):
    encoded = s.encode("utf-8")
    out.extend(struct.pack("!H", len(encoded)))
    out.extend(encoded)


def read_socket(sock: socket.socket, n_bytes: int) -> bytes:
    chunks = []
    while n_bytes > 0:
        chunk = sock.recv(n_bytes)
        if len(chunk) == 0:
            raise ConnectionError("Socket CLOSED, failed to read.")
        chunks.append(chunk)
        n_bytes -= len(chunk)
    return b"".join(chunks)
