import vgamepad as vg
import XInput as xinput
import mmap
import struct
import bettercam
from time import sleep
import torch
import torchvision.io
import random
import copy
from model import DriverActorModel, DriverCriticModel, T, DEVICE, IMG_RESOLUTION
from PIL import Image
from collections import namedtuple
from ipc import ControllerIPC, GameIPC

# The bettercam repo is kind of broken
# TODO: Honestly I should look into how it works under the hood & just elide it as a dependency
from bettercam.processor.cupy_processor import CupyProcessor

def process_cvtcolor(self, image):
    return image

CupyProcessor.process_cvtcolor = process_cvtcolor

class TimeDelayFrameBuffer:

    def __init__(self, img_resolution=IMG_RESOLUTION, n=300, k=3, dt=4):
        self.img_resolution = img_resolution
        self.n = n
        self.k = k
        self.dt = dt
        self.buffer = torch.zeros(n+(dt*k), 1, *IMG_RESOLUTION).to(device=DEVICE, dtype=T)
        self.idx = 0

    def __getitem__(self, idx):
        if idx == -1:
            return self[self.idx]
        idxs = [(idx-(self.dt*o))%self.n for o in range(self.k)] 
        return self.buffer[idxs, ...].reshape(-1, *self.img_resolution)

    def __setitem__(self, idx, frame):
        self.buffer[idx%self.n, ...] = frame.reshape(-1, *self.img_resolution)

    def __iadd__(self, frame):
        self[self.idx] = frame
        self.idx = (self.idx+1) % self.n
        return self

    def get_latest_frame(self):
        return self[self.idx]

class Environment:

    def __init__(self, img_resolution=IMG_RESOLUTION, time_delay_frame_dt=4, n_time_delays=3):
        self.game_ipc = GameIPC()
        self.controller_ipc = ControllerIPC()
        self.img_resolution = img_resolution
        self.sct = bettercam.create(device_idx=0, nvidia_gpu=True)
        self.fb = TimeDelayFrameBuffer(img_resolution=IMG_RESOLUTION, n=time_delay_frame_dt*n_time_delays+1, k=n_time_delays, dt=time_delay_frame_dt)
    
    def observe(self):
        self.controller_ipc.get_state()
        return [self.grab_screenshot()] + [torch.zeros(1, 4, dtype=T, device=DEVICE), torch.zeros(1, 3, dtype=T, device=DEVICE), torch.zeros(1, 3, dtype=T, device=DEVICE), torch.tensor([0], dtype=T, device=DEVICE)]

    def _observe(self):
        # TODO: make the next 3 methods async
        game_state = self.game_ipc.get_state()
        controller_state = self.controller_ipc.get_state()
        screenshot = self.grab_screenshot()
        return [torch.tensor(x, dtype=T, device=DEVICE).unsqueeze(0) for x in (screenshot, controller_state, *game_state)]

    def act(self, action):
        self.controller_ipc.write_action(*action.tolist())
        # self.game_ipc.request_state()
        return self.observe()

    def grab_screenshot(self):
        img = self.sct.grab()
        while img is None:
            img = self.sct.grab()
        img = torch.as_tensor(img, device=DEVICE)
        img = img[:, :, :3].permute(2, 0, 1)
        img = torchvision.transforms.functional.rgb_to_grayscale(img)
        img = img.to(device=DEVICE, dtype=torch.float16) / 255
        img = torch.nn.functional.interpolate(img.unsqueeze(0), self.img_resolution, mode='bilinear', antialias=True)
        img = (255 * img).to(device=DEVICE, dtype=torch.uint8)
        self.fb += img
        return self.fb.get_latest_frame().unsqueeze(0)

# Transition = namedtuple('Transition', ('S', 'A', 'NS', 'R'))
class Transition(namedtuple('_Transition', ('S', 'A', 'NS', 'R'))):

    def __iadd__(self, other):
        return Transition(
            torch.cat(self.S, other.S), 
            torch.cat(self.A, other.A),
            torch.cat(self.NS, other.NS),
            torch.cat(self.R, other.R)
        )

    def __add__(self, other):
        if other != 0:
            self += other
        return self

class ReplayBuffer:

    def __init__(self, n=300):
        self.n = n
        self.buffer = [None for _ in range(n)]
        self.idx = 0

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __iadd__(self, transition: Transition):
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.n
        return self

    def __getitem__(self, idx):
        return self.buffer[idx]

    def add(self, transition: Transition):
        self += transition

    def sample(self, batch_size):
        observations = random.sample([x for x in self.buffer if x is not None], batch_size)
        return sum(observations)

    def reset(self):
        self.buffer = [None for _ in range(n)]
        self.idx = 0


# Implements https://spinningup.openai.com/en/latest/algorithms/ddpg.html

GAMMA = 0.990
POLYAK = 0.995
RANDOM_EXPLORE_COUNT = 50

N_TRANSITIONS = 600
BATCH_SIZE = 100
N_UPDATES = 4

LR = 1e-2

torch.cuda.empty_cache()

environment = Environment(img_resolution=IMG_RESOLUTION)

actor = DriverActorModel().to(device=DEVICE, dtype=T)
critic = DriverCriticModel().to(device=DEVICE, dtype=T)

actor_target = DriverActorModel(img_resolution=IMG_RESOLUTION).to(device=DEVICE, dtype=T)
critic_target = DriverCriticModel(img_resolution=IMG_RESOLUTION).to(device=DEVICE, dtype=T)

actor_target.update_parameters(actor)
critic_target.update_parameters(critic)

actor_opt = torch.optim.AdamW(actor.parameters(), LR)
critic_opt = torch.optim.AdamW(critic.parameters(), LR)

replay_buffer = ReplayBuffer(n=N_TRANSITIONS)
actor_buffer = TimeDelayFrameBuffer(n=12)

# def optimize_model():

print('starting loop')
while True:
    # Interact with the environment using actor network
    with torch.no_grad():
        *S, _ = environment.observe()
        if RANDOM_EXPLORE_COUNT > 0:
            action = torch.rand(4)
            action[:2] *= 2
            action[:2] -= 1
        else:
            with torch.autocast(device_type='cuda'):
                action = actor(*S)
        *NS, DMG = environment.act(action)
        reward = torch.tensor([1]) if DMG == 0 else torch.tensor([0])
        S, NS, A, REW = tuple(s.to('cpu') for s in S), tuple(s.to('cpu') for s in NS), action.to('cpu'), reward.to('cpu')
        replay_buffer += Transition(S, A, NS, REW)
        print(torch.abs(A - S[1]).sum().item())
        
    # If there's a collision, end the episode & update the models
    if DMG > 0:

        for n in range(N_UPDATES):

            torch.cuda.empty_cache()

            # Sample experiences from replay buffer
            batch = replay_buffer.sample(BATCH_SIZE)

            # Get 'best estimate' from target networks
            with torch.autocast(device_type='cuda'):
                ACT_T  = actor_target(*batch.NS)
                CRT_T  = critic_target(*batch.NS, ACT_T)
                # Reward is 1 unless the car collided, which indicates a terminating state
                TARGET = batch.R + batch.R * (GAMMA * CRT_T)  

            # Update critic against best estimate via gradient descent
            critic_opt.zero_grad()
            CRIT_LOSS = (critic(*batch.S, batch.A) - TARGET).square().sum() / BATCH_SIZE
            CRIT_LOSS.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_opt.step()

            # Update actor against best estimate via gradient ascent
            actor_opt.zero_grad()
            ACT = actor(*batch.S)
            ACTOR_LOSS = -critic(*batch.S, ACT).sum() / BATCH_SIZE
            ACTOR_LOSS += 0.1 * (RANDOM_EXPLORE_COUNT == 0) * torch.abs(ACT - batch.S[1]).sum()
            ACTOR_LOSS.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()

            # Update target network parameters
            actor_target.polyak_update_parameters(actor, POLYAK)
            critic_target.polyak_update_parameters(critic, POLYAK)

        RANDOM_EXPLORE_COUNT -= RANDOM_EXPLORE_COUNT > 0
        replay_buffer.reset()