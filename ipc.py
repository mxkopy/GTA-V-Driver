import mmap
from typing import Iterable
import sys
import time
from struct import pack, unpack, calcsize

N_FLAGS = 8
FLAGS_TAG = "flags.ipc"

class IPCFlags:

    def __init__(self, n: int = N_FLAGS, tagname: str = FLAGS_TAG):
        self.flags = mmap.mmap(-1, -(n // -8), tagname)

    def set_flag(self, idx: int, value: bool) -> None:
        pos, offset = idx // 8, idx % 8
        mask = ~(1 << offset)
        self.flags.seek(pos)
        state = self.flags.read_byte()
        self.flags.seek(pos)
        updated_state = (state & mask) | (value << offset)
        self.flags.write_byte(updated_state)
        self.flags.flush()

    def get_flag(self, idx: int) -> bool:
        pos, offset = idx // 8, idx % 8
        mask = 1 << offset
        self.flags.seek(pos)
        state = self.flags.read_byte()
        return (state & mask) != 0

    def wait_until(self, idx: int | Iterable[int], value: bool | Iterable[bool], fn = lambda: time.sleep(1e-3)) -> None:
        if isinstance(idx, int) and isinstance(value, bool):
            while self.get_flag(idx) != value:
                fn()
        else:
            while sum(self.get_flag(i) != v for i, v in zip(idx, value)) > 0:
                fn()

class IPCChannel:

    def __init__(self, size: int, tagname: str):
        self.ipc = mmap.mmap(-1, size, tagname)

    def close(self) -> None:
        self.ipc.flush()
        self.ipc.close()

    def put(self, payload: bytes) -> None:
        self.ipc.seek(0)
        self.ipc.write(payload)
        self.ipc.flush()

    def take(self) -> bytes:
        self.ipc.seek(0)
        payload = self.ipc.read(-1)
        return payload

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

    def take(self):
        payload = super().take()
        if len(self.map) == 1:
            data = unpack(self.fmt_str, payload)
            if len(data) == 1:
                return data[0]
            else:
                return data
        else:
            data = [unpack(f'@{self.map[i]}', payload[self.start[i]:self.end[i]]) for i in range(len(self.sizes))]
            # return [x[0] if len(x) == 1 else x for x in data]
            return data

    def put(self, *payload):
        if len(self.map) == 1:
            super().put(pack(self.fmt_str, *payload))
        payload = [[p] if not isinstance(p, (list, tuple)) else p for p in payload]
        payload = sum(payload, start=[])
        super().put(pack(self.fmt_str, *payload))


class FLAGS:

    REQUEST_GAME_STATE = 0
    REQUEST_INPUT = 1
    REQUEST_ACTION = 2

    GAME_STATE_WRITTEN = 3
    INPUT_WRITTEN = 4
    ACTION_WRITTEN = 5

    RESET = 6
    IS_TRAINING = 7

class ControllerIPC(IPCFlags):

    INPUT_TAG = 'controller_input.ipc'
    OUTPUT_TAG = 'controller_output.ipc'

    def __init__(self, map: str | list[str]='4f', **kwargs):
        super().__init__(**kwargs)
        self.map: str | list[str] = map
        self.controller_input_channel: MappedIPCChannel = MappedIPCChannel(self.map, ControllerIPC.INPUT_TAG)
        self.controller_output_channel: MappedIPCChannel = MappedIPCChannel(self.map, ControllerIPC.OUTPUT_TAG)

    def pop_input(self):
        self.wait_until(FLAGS.INPUT_WRITTEN, True)
        state = self.controller_input_channel.take()
        # self.set_flag(FLAGS.INPUT_WRITTEN, False)
        return state

    def pop_action(self):
        self.wait_until(FLAGS.ACTION_WRITTEN, True)
        action = self.controller_output_channel.take()
        self.set_flag(FLAGS.ACTION_WRITTEN, False)
        return action

    def write_input(self, *state: float):
        # self.wait_until(FLAGS.REQUEST_INPUT, True)
        self.controller_input_channel.put(*state)
        self.set_flag(FLAGS.INPUT_WRITTEN, True)

    def write_action(self, *action: float):
        self.wait_until(FLAGS.REQUEST_ACTION, True)
        self.controller_output_channel.put(*action)
        self.set_flag(FLAGS.ACTION_WRITTEN, True)

    def request_input(self):
        self.set_flag(FLAGS.REQUEST_INPUT, True)
    
    def request_action(self):
        self.set_flag(FLAGS.REQUEST_ACTION, True)

    def block_input(self):
        self.set_flag(FLAGS.REQUEST_INPUT, False)
    
    def block_action(self):
        self.set_flag(FLAGS.REQUEST_ACTION, False)

class GameStateIPC(IPCFlags):

    GAME_STATE = 'game_state.ipc'

    def __init__(self, map=['3f', '3f', '1i'], **kwargs):
        super().__init__(**kwargs)
        self.map = map
        self.game_state_channel = MappedIPCChannel(map, GameStateIPC.GAME_STATE)

    def pop_game_state(self):
        self.wait_until(FLAGS.GAME_STATE_WRITTEN, True)
        game_state = self.game_state_channel.take()
        self.set_flag(FLAGS.GAME_STATE_WRITTEN, False)
        return game_state

    def request_game_state(self):
        self.set_flag(FLAGS.REQUEST_GAME_STATE, True)
    
    def block_game_state(self):
        self.set_flag(FLAGS.REQUEST_GAME_STATE, False)

    def debug_write_game_state(self, state: tuple[list[float], list[float], list[int]]):
        self.wait_until(FLAGS.REQUEST_GAME_STATE, True)
        self.game_state_channel.put(*state)
        self.set_flag(FLAGS.GAME_STATE_WRITTEN, True)


def debug_flags():
    flags = IPCFlags()
    print(f'REQUEST_GAME_STATE: {flags.get_flag(FLAGS.REQUEST_GAME_STATE)}')
    print(f'REQUEST_INPUT: {flags.get_flag(FLAGS.REQUEST_INPUT)}')
    print(f'REQUEST_ACTION: {flags.get_flag(FLAGS.REQUEST_ACTION)}')
    print(f'GAME_STATE_WRITTEN: {flags.get_flag(FLAGS.GAME_STATE_WRITTEN)}')
    print(f'INPUT_WRITTEN: {flags.get_flag(FLAGS.INPUT_WRITTEN)}')
    print(f'ACTION_WRITTEN: {flags.get_flag(FLAGS.ACTION_WRITTEN)}')
    print(f'RESET: {flags.get_flag(FLAGS.RESET)}')
    print(f'IS_TRAINING: {flags.get_flag(FLAGS.IS_TRAINING)}')
    flags.flags.seek(0)
    print(f'{flags.flags.read_byte()}')

# debug_flags()
# exit()

if __name__ == '__main__':
    debug_flags()
    # ipc = GameStateIPC()
    # while True:
    #     if ipc.get_flag(FLAGS.IS_TRAINING):
    #         state = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0,]
    #         print('writing game state')
    #         ipc.debug_write_game_state(state)
    #         print('wrote game state')
    #         while ipc.get_flag(FLAGS.GAME_STATE_WRITTEN) and ipc.get_flag(FLAGS.IS_TRAINING):
    #             time.sleep(0)
    #     else:
    #         print('not training')


# flags.set_flag(FLAGS.IS_TRAINING, False)

# debug_flags()
# ipc = GameStateIPC()
# ipc.set_flag(FLAGS.GAME_STATE_WRITTEN, True)
# print(ipc.pop_game_state())
