import torch
from ipc import GameStateIPC, ControllerIPC
from model import IMG_RESOLUTION, T, DEVICE
import bettercam
import random
import config
from collections import namedtuple


# The bettercam repo is kind of broken
# TODO: Honestly I should look into how it works under the hood & just elide it as a dependency
from bettercam.processor.cupy_processor import CupyProcessor
CupyProcessor.process_cvtcolor = lambda self, image: image

# Amazing one-liner to visualize stuff
# Image.fromarray((img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).show()
def take_screenshot(sct: bettercam.BetterCam):
    img = sct.grab()
    while img is None:
        img = sct.grab()
    img = torch.as_tensor(img, device=DEVICE)
    img = img[:, :, :3].permute(2, 0, 1)
    # img = torchvision.transforms.functional.rgb_to_grayscale(img)
    img = img.to(device=DEVICE, dtype=torch.float16) / 255
    img = torch.nn.functional.interpolate(img.unsqueeze(0), config.IMAGE_SIZE[1:], mode='bilinear', antialias=True).reshape(*config.IMAGE_SIZE)
    img = (255 * img).to(device=DEVICE, dtype=torch.uint8)
    return img

# STATE_SIZES = {
#     'image': config.IMAGE_SIZE,
#     'controller': config.CONTROLLER_SIZE,
#     'camera_direction': config.CAMERA_DIRECTION_SIZE,
#     'velocity': config.VELOCITY_SIZE,
#     'forward_direction': config.FORWARD_DIRECTION_SIZE
# }

State = namedtuple('State', tuple(config.state_sizes.keys()))

class State(State):

    def cat(states: list[State] | tuple[State]) -> State:
        return State(
            *(torch.cat(tuple(state.__getattribute__(x) for state in states)) for x in State._fields) 
        )

    def __add__(self, other: State) -> State:
        return self.__class__(
            *(torch.cat((s, o)) for s, o in zip(self, other))
        )
    
    def __radd__(self, other: int | None | State):
        if isinstance(other, State):
            return self.__class__(*other).__add__(self)
        else:
            return self
    
    def __getitem__(self, idx: int | slice | tuple[slice] | list[slice]):
        if isinstance(idx, int):
            return self.__class__( *(x[idx:idx+1, ...] for x in self) )
        elif isinstance(idx, slice):
            return self.__class__( *(x[idx, ...] for x in self))
        else:
            return self.__class__( *(x[idx] for x in self))

    def rand(batch_size = 1):
        return State(
            *(torch.rand(batch_size, *config.state_sizes[key]) for key in State._fields)
        )

    def size_of(self, key):
        return config.state_sizes[key]

type Action = torch.tensor
type NextState = State
type Reward = torch.tensor
type Final = torch.tensor

Transition = namedtuple('Transition', ('state', 'action', 'nextstate', 'reward', 'final'))
class Transition(Transition):

    def cat(transitions: list[Transition] | tuple[Transition]) -> Transition:
        return Transition(
            State.cat(tuple(transition.state for transition in transitions)),
            torch.cat(tuple(transition.action for transition in transitions)),
            State.cat(tuple(transition.nextstate for transition in transitions)),
            torch.cat(tuple(transition.reward for transition in transitions)),
            torch.cat(tuple(transition.final for transition in transitions))
        )

    def __add__(self, other: Transition):
        return self.__class__(
            self.state + other.state,
            torch.cat((self.action, other.action)),
            self.nextstate + other.nextstate,
            torch.cat((self.reward, other.reward)),
            torch.cat((self.final, other.final))
        )

    def __radd__(self, other: int | None | Transition):
        if isinstance(other, Transition):
            return self.__class__(*other).__add__(self)
        else:
            return self

    def __getitem__(self, idx: int | slice | tuple[slice] | list[slice]):
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
        self.game_ipc = GameStateIPC()
        self.controller_ipc = ControllerIPC()
        self.sct = bettercam.create(device_idx=0, nvidia_gpu=True)
        self.game_ipc.request_game_state()
        self.controller_ipc.request_input()
        self.controller_ipc.request_action()
    
    def debug_observe(self) -> State:
        return State.rand()

    def observe(self) -> State:
        screenshot: torch.tensor = take_screenshot(self.sct)
        self.game_ipc.request_game_state()
        game_state: list[torch.tensor] = [torch.tensor(x).unsqueeze(0) for x in self.game_ipc.pop_game_state()]
        self.game_ipc.block_game_state()
        controller_state: torch.tensor = torch.tensor(self.controller_ipc.pop_input()).unsqueeze(0)
        return State(screenshot, controller_state, *game_state)

    def act(self, action: torch.tensor) -> State:
        self.controller_ipc.write_action(*action.reshape(-1).tolist())
        return self.observe()
    
# def random_transition():
#     return Transition(State.rand(), torch.rand(1, 4), State.rand(), torch.rand(1, 1), torch.rand(1, 1))

# yry = Transition.cat([random_transition()] * 10)
# print(yry[2:4].state.image.size())

# exit()