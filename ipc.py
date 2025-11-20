import mmap
from typing import Iterable
import time
import msgs_pb2
from google.protobuf.message import Message
from google.protobuf.descriptor import FieldDescriptor
N_FLAGS = 8
FLAGS_TAG = "flags.ipc"
IPC_SLEEP_DURATION = 1e-3

class FLAGS:

    REQUEST_GAME_STATE = 0
    REQUEST_INPUT = 1
    REQUEST_ACTION = 2

    GAME_STATE_WRITTEN = 3
    INPUT_WRITTEN = 4
    ACTION_WRITTEN = 5

    RESET = 6
    IS_TRAINING = 7

class IPCFlags:

    flags = mmap.mmap(-1, -(N_FLAGS // -8), FLAGS_TAG)

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

    def wait_until(self, idx: int | Iterable[int], value: bool | Iterable[bool], fn = lambda: time.sleep(IPC_SLEEP_DURATION)) -> None:
        if isinstance(idx, int) and isinstance(value, bool):
            while self.get_flag(idx) != value:
                fn()
        else:
            while sum(self.get_flag(i) != v for i, v in zip(idx, value)) > 0:
                fn()

class Channel:

    def __init__(self, size: int, tagname: str):
        self.ipc = mmap.mmap(-1, size, tagname)

    def close(self) -> None:
        self.ipc.flush()
        self.ipc.close()

    def push_nbl(self, payload: bytes) -> None:
        self.ipc.seek(0)
        self.ipc.write(payload)
        self.ipc.flush()

    def pop_nbl(self) -> bytes:
        self.ipc.seek(0)
        payload = self.ipc.read(-1)
        return payload

class MappedChannel(Channel):

    N_BYTES: int
    MSG_TYPE: type[Message]

    def __init__(self, tagname: str):
        super().__init__(self.__class__.N_BYTES, tagname=tagname)

    def push_nbl(self, payload: Message):
        assert payload.__class__ == self.__class__.MSG_TYPE
        msg: bytes = payload.SerializeToString()
        super().push_nbl(msg)

    def pop_nbl(self) -> Message:
        msg: bytes = super().pop_nbl()
        return self.__class__.MSG_TYPE.FromString(msg)

class StateQueue(IPCFlags, MappedChannel):

    READY_TO_READ: int
    NEW_MESSAGE_WRITTEN: int
    TAGNAME: str

    N_BYTES: int
    MSG_TYPE: type[Message]

    def __init__(self):
        IPCFlags.__init__(self)
        MappedChannel.__init__(self, tagname=self.__class__.TAGNAME)

    def push(self, state: Message):
        self.wait_until(self.__class__.READY_TO_READ, True)
        self.push_nbl(state)
        self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, True)

    def pop(self) -> Message:
        self.wait_until(self.__class__.NEW_MESSAGE_WRITTEN, True)
        state: Message = self.pop_nbl()
        self.set_flag(self.__class__.NEW_MESSAGE_WRITTEN, False)
        return state
    
    def __xor__(self, update: Message):
        current_state: Message = self.pop_nbl()
        current_state.MergeFrom(update)
        return current_state
    
    def __ixor__(self, *update):
        updated_state = self ^ update
        self.push_nbl(updated_state)
        return self

class FixedSizeState(type):

    def default_init_dict(descriptor: FieldDescriptor):
        descriptor.fields: list[FieldDescriptor]
        return {
            field.name: 0 if field.message_type is None else FixedSizeState.default_init_dict(field.message_type)
            for field in descriptor.fields
        }

    def init(self: Message):
        self.CopyFrom(self.__class__(**FixedSizeState.default_init_dict(self.DESCRIPTOR)))
        return self

    def tolist(self: Message):
        return [value if not isinstance(value, Message) else FixedSizeState.tolist(value) for _, value in self.ListFields()]

    def __init__(cls, *args, **kwargs):
        cls.MSG_TYPE.init = FixedSizeState.init
        cls.MSG_TYPE.tolist = FixedSizeState.tolist
        cls.N_BYTES = cls.MSG_TYPE(**FixedSizeState.default_init_dict(cls.MSG_TYPE.DESCRIPTOR)).ByteSize()

class VirtualControllerState(StateQueue, metaclass=FixedSizeState):

    READY_TO_READ = FLAGS.REQUEST_ACTION
    NEW_MESSAGE_WRITTEN = FLAGS.ACTION_WRITTEN
    TAGNAME = 'virtual_controller.ipc'
    MSG_TYPE = msgs_pb2.ControllerState

    # def tolist(self) -> list:
        # return [msg.left_joystick_x, msg.left_joystick_y, msg.left_trigger, msg.right_trigger]
    
class PhysicalControllerState(StateQueue, metaclass=FixedSizeState):

    READY_TO_READ = FLAGS.REQUEST_INPUT
    NEW_MESSAGE_WRITTEN = FLAGS.INPUT_WRITTEN
    TAGNAME = 'physical_controller.ipc'
    MSG_TYPE = msgs_pb2.ControllerState

    # push = VirtualControllerState.push
    # pop = VirtualControllerState.pop


class GameState(StateQueue, metaclass=FixedSizeState):

    READY_TO_READ = FLAGS.REQUEST_GAME_STATE
    NEW_MESSAGE_WRITTEN = FLAGS.GAME_STATE_WRITTEN
    TAGNAME = 'game_state.ipc'
    MSG_TYPE = msgs_pb2.GameState

    # def push(self, x: float, y: float, lt: float, rt: float):
    #     msg = self.MSG_TYPE(left_joystick_x=x, left_joystick_y=y, left_trigger=lt, right_trigger=rt)
    #     super().push(msg)

gs = GameState().MSG_TYPE().init().tolist()
print(list(gs))

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
    # ipc = GameIPC()
    # while True:
    #     if ipc.get_flag(FLAGS.IS_TRAINING):
    #         state = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0,]
    #         print('writing game state')
    #         ipc.debug_write_game_state(state)
    #         print('wrote game state')
    #         while ipc.get_flag(FLAGS.GAME_STATE_WRITTEN) and ipc.get_flag(FLAGS.IS_TRAINING):
    #             time.sleep(1e-3)
    #     else:
    #         print('not training')