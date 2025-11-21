import torch
import bettercam
import random
import config
import msgs_pb2
from google.protobuf.message import Message
from collections import namedtuple
from typing import Iterable, TypeVarTuple, TypeAlias
from ipc import GameState, VirtualControllerState, PhysicalControllerState, FLAGS


# The bettercam repo is kind of broken
# TODO: Honestly I should look into how it works under the hood & just elide it as a dependency
from bettercam.processor.cupy_processor import CupyProcessor
CupyProcessor.process_cvtcolor = lambda self, image: image

# Amazing one-liner to visualize stuff
# Image.fromarray((img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).show()
def take_screenshot(sct: bettercam.BetterCam) -> torch.Tensor:
    img = sct.grab()
    while img is None:
        img = sct.grab()
    img = torch.as_tensor(img, device=config.device)
    img = img[:, :, :3].permute(2, 0, 1).unsqueeze(0)
    # img = torchvision.transforms.functional.rgb_to_grayscale(img)
    img = img.to(dtype=torch.float16) / 255
    img = torch.nn.functional.interpolate(img, config.state_sizes['image'][1:], mode='bilinear', antialias=True)
    img = (255 * img).to(dtype=torch.uint8)
    return img



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
        self.reserve = random.sample(self.reserve + self.buffer, self.n)
        self.buffer = [None for _ in range(self.capacity)]
        self.idx = 0

class Environment:

    def __init__(self):
        self.game_state = GameState()
        self.physical_controller_state = PhysicalControllerState()
        self.virtual_controller_state = VirtualControllerState()
        self.sct = bettercam.create(device_idx=0, nvidia_gpu=True)
        self.game_state.set_flag(FLAGS.REQUEST_GAME_STATE, True)
        self.physical_controller_state.set_flag(FLAGS.REQUEST_INPUT, True)
        self.virtual_controller_state.set_flag(FLAGS.REQUEST_ACTION, True)
    
    def debug_observe(self) -> State:
        return State.rand()

    def observe(self) -> State:
        screenshot: torch.Tensor = take_screenshot(self.sct)
        self.game_state.set_flag(FLAGS.REQUEST_GAME_STATE, True)
        game_state: msgs_pb2.GameState = self.game_state.pop()
        self.game_state.set_flag(FLAGS.REQUEST_GAME_STATE, False)
        physical_controller_state: msgs_pb2.ControllerState = self.physical_controller_state.pop()        
        physical_controller_state: torch.Tensor = torch.tensor(physical_controller_state.to_tuple()).unsqueeze(0)
        game_state = (torch.tensor([x] if not isinstance(x, list) else x).unsqueeze(0) for x in game_state.to_tuple())
        return State(screenshot, physical_controller_state, *game_state)

    def perform_action(self, action: torch.Tensor) -> tuple[Reward, NextState, Final]:
        self.virtual_controller_state.push(*action.reshape(-1).tolist())
        nextstate: NextState = self.observe()
        reward: Reward = torch.dot(nextstate.velocity.view(-1), nextstate.camera_direction.view(-1)).unsqueeze(0)
        return (nextstate, reward, nextstate.damage > 0)
    
    def pause_training(self):
        self.game_state.set_flag(FLAGS.REQUEST_GAME_STATE, False)
        self.game_state.set_flag(FLAGS.IS_TRAINING, False)

    def resume_training(self):
        self.game_state.set_flag(FLAGS.REQUEST_GAME_STATE, True)
        self.game_state.set_flag(FLAGS.IS_TRAINING, True)
# def random_transition():
#     return Transition(State.rand(), torch.rand(1, 4), State.rand(), torch.rand(1, 1), torch.rand(1, 1))

# yry = Transition.cat([random_transition()] * 10)
# print(yry[2:4].state.image.size())

# exit()