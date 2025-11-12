import mmap
import sys
from struct import pack, unpack, calcsize

class IPCChannel:

    def __init__(self, payload_length: int, tag: str):
        self.ipc = mmap.mmap(-1, 1 + payload_length, tag)

    def close(self) -> None:
        self.ipc.flush()
        self.ipc.close()

    def is_put_locked(self) -> bool:
        self.ipc.seek(0)
        return 1 & self.ipc.read_byte()

    def is_consume_locked(self) -> bool:
        return not self.is_put_locked()

    def set_lock(self, value: bool) -> None:
        self.ipc.seek(0)
        self.ipc.write(value.to_bytes(byteorder=sys.byteorder))
        self.ipc.flush()

    def unlock_consume(self) -> None:
        self.set_lock(True)

    def unlock_put(self) -> None:
        self.set_lock(False)

    def lock_consume(self) -> None:
        self.unlock_put()

    def lock_put(self) -> None:
        self.unlock_consume()

    def put(self, payload: bytes) -> None:
        self.ipc.seek(1)
        self.ipc.write(payload)
        self.unlock_consume()

    def consume(self) -> bytes:
        self.ipc.seek(1)
        payload = self.ipc.read(-1)
        self.unlock_put()
        return payload

    def put_blocking(self, *payload) -> None:
        while self.is_put_locked():
            pass
        self.put(*payload)

    def consume_blocking(self):
        while self.is_consume_locked():
            pass
        return self.consume()

class MappedIPCChannel(IPCChannel):

    def __init__(self, map: str | list[str], tag: str):
        if isinstance(map, str):
            map = [map]
        self.map = map
        self.fmt_str = '@'+ ''.join(map)
        self.sizes = [calcsize('@' + fmt) for fmt in map]
        self.start = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.end = [self.start[i] + self.sizes[i] for i in range(len(self.sizes))] 
        super().__init__(sum(self.sizes), tag)

    def consume(self):
        payload = super().consume()
        if len(self.map) == 1:
            data = unpack(self.fmt_str, payload)
            if len(data) == 1:
                return data[0]
            else:
                return data
        else:
            data = [unpack(f'@{self.map[i]}', payload[self.start[i]:self.end[i]]) for i in range(len(self.sizes))]
            return [x[0] if len(x) == 1 else x for x in data]

    def put(self, *payload):
        if len(self.map) == 1:
            super().put(pack(self.fmt_str, *payload))
        payload = [[p] if not isinstance(p, (list, tuple)) else p for p in payload]
        payload = sum(payload, start=[])
        super().put(pack(self.fmt_str, *payload))


class ControllerIPC:

    INPUT = 'controller_input.ipc'
    OUTPUT = 'controller_output.ipc'

    def __init__(self, map='4e'):
        self.map = map
        self.controller_input_channel = MappedIPCChannel(self.map, ControllerIPC.INPUT)
        self.controller_output_channel = MappedIPCChannel(self.map, ControllerIPC.OUTPUT)

    def write_action(self, *action: float):
        self.controller_output_channel.put(*action)

    def get_action(self):
        return self.controller_output_channel.consume_blocking()

    def write_state(self, *state: float):
        self.controller_input_channel.put_blocking(*state)

    def get_state(self):
        return self.controller_input_channel.consume_blocking()

class GameIPC:

    GAME = 'game.ipc'

    def __init__(self, map=['3f', '3f', 'i']):
        self.map = map
        self.game_ipc = MappedIPCChannel(map, GameIPC.GAME)

    def get_state(self):
        return self.game_ipc.consume_blocking()

    def request_state(self):
        self.game_ipc.unlock_put()