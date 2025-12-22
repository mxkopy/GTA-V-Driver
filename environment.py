# import cupy as cp

# a = cp.empty((10,))
# a[:] = 3
# print(a)
# exit()

import torch
import bettercam
import random
import config
import torchvision
import math
import mmap
import cupy
import numpy as np
# import msgs_pb2
from msgs_pb2 import ControllerState
from google.protobuf.message import Message
from collections import namedtuple
from typing import Iterable, TypeVarTuple, TypeAlias
from ipc import GameState, VirtualControllerState, PhysicalControllerState, FLAGS
from struct import unpack

class TensorTuple[T: tuple[torch.Tensor, ...]]:

    def cat(states: Iterable[T]) -> T:
        assert all(len(state) for state in states)
        return states[0].__class__(
            *(torch.cat(tuple(tuple.__getitem__(state, i) for state in states)) for i in range(len(states[0])))
        )

    def __add__(self, other: T) -> T:
        return self.__class__(
            *(torch.cat((s, o)) for s, o in zip(self, other))
        )
    
    def __radd__(self, other: int | None | T):
        if isinstance(other, T):
            return self.__class__(*other).__add__(self)
        else:
            return self
    
    def __getitem__(self, idx: int | slice | Iterable[slice]) -> T:
        if isinstance(idx, int):
            return self.__class__( *(x[idx:idx+1, ...] for x in self) )
        elif isinstance(idx, slice):
            return self.__class__( *(x[idx, ...] for x in self))
        else:
            return self.__class__( *(x[idx] for x in self))

    def to(self, **kwargs) -> T:
        return self.__class__(
            *(x.to(**kwargs) for x in self)
        )

    def dynamic_shapes(self, shapes=torch.export.ShapesCollection()):
        for state in self:
            shapes[state] = {0: torch.export.Dim.DYNAMIC}
        return shapes

StateTuple = namedtuple('State', tuple(config.state_sizes.keys()))
class State(StateTuple, TensorTuple):

    def rand(batch_size = 1):
        return State(
            *(torch.rand(batch_size, *config.state_sizes[key]) for key in State._fields)
        )
    
    def size(self):
        return tuple(x.size() for x in self)

type Action = torch.Tensor
type NextState = State
type Reward = torch.Tensor
type Final = torch.Tensor

TransitionTuple = namedtuple('Transition', ('state', 'action', 'nextstate', 'reward', 'final'))
class Transition(TransitionTuple):

    def cat(transitions: Iterable[TransitionTuple]) -> TransitionTuple:
        return Transition(
            State.cat(tuple(transition.state for transition in transitions)),
            torch.cat(tuple(transition.action for transition in transitions)),
            State.cat(tuple(transition.nextstate for transition in transitions)),
            torch.cat(tuple(transition.reward for transition in transitions)),
            torch.cat(tuple(transition.final for transition in transitions))
        )

    def to(self, **kwargs) -> TransitionTuple:
        return Transition(*(x.to(**kwargs) for x in self))

    def __add__(self, other: TransitionTuple):
        return self.__class__(
            self.state + other.state,
            torch.cat((self.action, other.action)),
            self.nextstate + other.nextstate,
            torch.cat((self.reward, other.reward)),
            torch.cat((self.final, other.final))
        )

    def __radd__(self, other: int | None | TransitionTuple):
        if isinstance(other, Transition):
            return self.__class__(*other).__add__(self)
        else:
            return self

    def __getitem__(self, idx: int | slice | Iterable[slice]):
        if isinstance(idx, int):
            return self.__class__( *(x[idx:idx+1, ...] for x in self) )
        elif isinstance(idx, slice):
            return self.__class__( *(x[idx, ...] for x in self))
        else:
            return self.__class__( *(x[idx] for x in self))

class VideoState:

    def __init__(self, queue_length=100, depth=True):
        self.depth = depth
        self.cuda_array = None

    def init_cuda_array(idx=1):
        from ipc import Channel
        array_handle = Channel(64, f"CudaArray{idx}")
        array_format = Channel(32, f"CudaArray{idx}Info")
        memory_handle = array_handle.pop_nbl()
        components, bpp, pitch, height = unpack("@4P", array_format.pop_nbl())
        if components != 0:
            arrayPtr = cupy.cuda.runtime.ipcOpenMemHandle(memory_handle)
            membuffer = cupy.cuda.UnownedMemory(arrayPtr, pitch * height, owner=VideoState, device_id=0)
            return cupy.ndarray(shape=(components, height, pitch // bpp), dtype=cupy.float32, memptr=cupy.cuda.MemoryPointer(membuffer, 0))
        return None

    def linearize_depth(array, near, far):
        y = (torch.pow(far/near,array)-1) * (near / far)
        return ((y * near) / far)
    
    def pop(self) -> torch.Tensor:
        if self.cuda_array is None:
            self.cuda_array = VideoState.init_cuda_array(idx=1)
            if self.cuda_array is None:
                print("Something went wrong initializing VideoState")
                exit()
        tensor = torch.from_dlpack(self.cuda_array)
        if self.depth:
            if not hasattr(self, 'nearclipfarclip'):
                from ipc import Channel
                self.nearclipfarclip = Channel(8, "NearClipFarClip")
            near, far = unpack('@2f', self.nearclipfarclip.pop_nbl())
            img = VideoState.linearize_depth(tensor, near, far).unsqueeze(0)
        else:
            img = img[:, :, :3].permute(2, 0, 1).unsqueeze(0)
            if self.grayscale:
                img = torchvision.transforms.functional.rgb_to_grayscale(img)
            img = img.to(dtype=torch.float16)
        img = torch.nn.functional.interpolate(img, config.state_sizes['image'][1:], mode='bilinear', antialias=True)
        return img
        
    def display(self):
        import numpy as np
        from PIL import Image
        img = self.pop()
        if img.shape[1] == 1:
            img = img.squeeze().cpu().numpy()
            Image.fromarray(((img / img.max())*255).astype(np.uint8)).show()
        else:
            img = img.squeeze().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(((img / img.max())*255).astype(np.uint8)).show()

class FrameQueue:

    def __init__(self, maxlen=100):
        self.maxlen = maxlen
        self.buffer = []
    
    def append(self, x):
        self.buffer.append(x)
        if len(self.buffer) == self.maxlen:
            self.buffer.pop(0)

    def collate(self, f=lambda tensor: tensor):
        return f(torch.cat(self.buffer, dim=0))
    
    def reset(self):
        self.buffer = []

    def mul_window(window, batch):
        window = window.reshape((-1,) + tuple(1 for _ in batch.size()[1:]))
        return window * batch

    def blueshift(batch):
        n = batch.shape[0]
        r = torch.linspace(0, 1, n).to(device=batch.device)
        g = torch.cat((torch.linspace(0, 1, n // 2), torch.linspace(1, 0, n-(n // 2) + 1)[1:])).to(device=batch.device)
        b = torch.linspace(1, 0, n).to(device=batch.device)
        R = FrameQueue.mul_window(r, batch).sum(dim=0, keepdim=True)
        G = FrameQueue.mul_window(g, batch).sum(dim=0, keepdim=True)
        B = FrameQueue.mul_window(b, batch).sum(dim=0, keepdim=True)
        Y = torch.cat((R, G, B), dim=1)
        return Y

    def polynomial_decay(batch, p=1):
        n = batch.shape[0]
        window = torch.linspace(0, 1, n).pow(p).to(device=batch.device)
        y = FrameQueue.mul_window(window, batch).sum(dim=0, keepdim=True)
        return y

class ReplayBuffer:

    def __init__(self, capacity=300):
        self.capacity = capacity
        self.reserve = [None for _ in range(self.capacity)]
        self.buffer = [None for _ in range(self.capacity)]
        self.idx = 0

    def __iadd__(self, transition: Transition):
        self.buffer[self.idx%self.capacity] = transition
        self.idx = (self.idx+1)%self.capacity
        return self

    def add(self, transition: Transition):
        self += transition

    def sample(self, batch_size):
        batch = [x for x in self.buffer if x is not None] + [x for x in self.reserve if x is not None]
        observations = random.sample(batch, min(len(batch), batch_size))
        return Transition.cat(observations)

    def reset(self):
        self.reserve = random.sample(self.reserve + self.buffer, self.capacity)
        self.buffer = [None for _ in range(self.capacity)]
        self.idx = 0

class Environment:

    def __init__(self, queue_length=100):
        from controller import VirtualController
        self.video_state = VideoState(queue_length=queue_length)
        self.game_state = GameState()
        self.virtual_controller = VirtualController
    
    def observe(self) -> State:
        game_state: tuple = self.game_state.pop()
        video_state: torch.Tensor = self.video_state.pop()
        game_state = (torch.tensor([x] if not isinstance(x, Iterable) else x).unsqueeze(0) for x in game_state)
        return State(video_state, *game_state)

    def perform_action(self, action: torch.Tensor) -> tuple[Reward, NextState, Final]:
        self.virtual_controller.update(tuple(*action.tolist()))
        nextstate: NextState = self.observe()
        reward: Reward = -nextstate.damage
        return (nextstate, reward, nextstate.damage > 0)
    
    def pause_training(self):
        self.game_state.set_flag(FLAGS.REQUEST_GAME_STATE, True)
        self.game_state.set_flag(FLAGS.IS_TRAINING, False)

    def resume_training(self):
        self.game_state.set_flag(FLAGS.IS_TRAINING, True)

    def reset(self):
        pass
        # self.video_state.display()
        # self.video_state.queue.reset()
