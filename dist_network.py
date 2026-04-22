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
ROLLOUT_ARRAY_KEYS = ("obss", "reset_prefixes", "dones", "actions", "rewards", "old_log_probs")
COMPRESSED_FLAG = 1

PROTOCOL_MAGIC_SIZE = len(PROTOCOL_MAGIC)
MESSAGE_TYPE_SIZE = 1
FLAGS_SIZE = 1
MSG_LEN_SIZE = 8
POLICY_VERSION_SIZE = 8


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


def encode_msg(msg: dict) -> bytes:
    header = bytearray(PROTOCOL_MAGIC)
    header.extend(struct.pack("!b", msg["type"]))
    if msg["type"] == MessageType.GET_WEIGHTS:
        header.extend(struct.pack("!q", msg["policy_version"]))
        return bytes(header)
    if msg["type"] == MessageType.ACK:
        return bytes(header)
    
    payload = bytearray()
    if msg["type"] == MessageType.WEIGHTS:
        state_dict = {} if msg.get("state_dict") is None else msg["state_dict"]
        payload.extend(struct.pack("!qI", msg["policy_version"], len(state_dict)))
        for name, arr in state_dict.items():
            append_str(payload, name)
            append_array(payload, arr)
    elif msg["type"] == MessageType.ROLLOUT:
        append_str(payload, msg["actor_id"])
        payload.extend(struct.pack("!qdI", msg["policy_version"], msg["total_reward"], msg["n_episodes"]))
        for key in ROLLOUT_ARRAY_KEYS:
            append_array(payload, msg[key])
    else:
        raise ValueError(f"Unknown message type: {msg['type']}.")

    payload = bytes(payload)
    compressed = lz4.frame.compress(payload)
    flags = 0
    if len(compressed) < len(payload):
        payload = compressed
        flags = COMPRESSED_FLAG
    header.extend(struct.pack("!b", flags))
    header.extend(struct.pack("!q", len(payload)))
    return bytes(header) + payload


def recv_msg(sock: socket.socket) -> tuple[dict, int]:
    magic = read_socket(sock, PROTOCOL_MAGIC_SIZE)
    if magic != PROTOCOL_MAGIC:
        raise ConnectionError(f"Protocol magic does not match: {magic}")

    msg_type, = struct.unpack("!b", read_socket(sock, MESSAGE_TYPE_SIZE))
    msg_size = PROTOCOL_MAGIC_SIZE + MESSAGE_TYPE_SIZE

    if msg_type == MessageType.GET_WEIGHTS:
        policy_version, = struct.unpack("!q", read_socket(sock, POLICY_VERSION_SIZE))
        return {"type": msg_type, "policy_version": policy_version}, msg_size + POLICY_VERSION_SIZE
    if msg_type == MessageType.ACK:
        return {"type": msg_type}, msg_size

    flags, = struct.unpack("!b", read_socket(sock, FLAGS_SIZE))
    payload_size, = struct.unpack("!q", read_socket(sock, MSG_LEN_SIZE))
    payload = read_socket(sock, payload_size)
    if flags & COMPRESSED_FLAG:
        payload = lz4.frame.decompress(payload)

    msg_size += FLAGS_SIZE + MSG_LEN_SIZE + payload_size
    return decode_payload(msg_type, payload), msg_size


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
