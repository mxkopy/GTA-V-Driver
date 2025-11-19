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
import math
from model import DriverActorModel, DriverCriticModel, T, DEVICE, IMG_RESOLUTION
from PIL import Image
from collections import namedtuple, deque
from controller import ControllerHandler
from ipc import FLAGS
from environment import Environment
from replaybuffer import ReplayBuffer, Transition


# literally just a fifo queue
class AveragingWindow:

    def __init__(self, maxlen=100):
        self.maxlen = maxlen
        self.buffer = []
    
    def append(self, x):
        self.buffer.append(x)
        if len(self.buffer) == self.maxlen:
            self.buffer.pop(0)

    def collate(self, f=lambda tensor: tensor.to(dtype=T).mean(dim=0, keepdim=True)):
        return f(torch.cat(self.buffer, dim=0))
    
    def reset(self):
        self.buffer = []

def linear_decay(batch):
    n = batch.shape[0]
    window = 1/torch.arange(n, 0, -1)
    window = window.reshape( (-1,)+tuple(1 for _ in batch.size()[1:]) )
    y = (window * batch).sum(dim=0, keepdim=True)
    return y

def polynomial_decay(batch, p=2):
    n = batch.shape[0]
    window = 1/torch.arange(n, 0, -1).pow(p)
    window = window.reshape( (-1,)+tuple(1 for _ in batch.size()[1:]) )
    y = (window * batch).sum(dim=0, keepdim=True)
    return y

def exponential_decay(batch):
    n = batch.shape[0]
    expavg = (math.exp(n) - 1)/(n*math.exp(n) * (math.exp(1) - 1))
    window = 1/torch.exp(torch.arange(n-1, -1, -1))
    window = window.reshape( (-1,)+tuple(1 for _ in batch.size()[1:]))
    y =  (window * batch).sum(dim=0, keepdim=True)
    return y

# Implements https://spinningup.openai.com/en/latest/algorithms/ddpg.html

GAMMA = 0.85
POLYAK = 0.90
RANDOM_EXPLORE_COUNT = 0

N_TRANSITIONS = 600
BATCH_SIZE = 100
N_UPDATES = 4
WINDOW_SIZE = 100

LR = 1e-5
import gc
gc.collect()
torch.cuda.empty_cache()

environment = Environment(img_resolution=IMG_RESOLUTION)

actor = DriverActorModel(img_resolution=IMG_RESOLUTION).to(device=DEVICE, dtype=T).jit()
critic = DriverCriticModel(img_resolution=IMG_RESOLUTION).to(device=DEVICE, dtype=T).jit()

actor_target = copy.deepcopy(actor).to(device=DEVICE, dtype=T)
critic_target = copy.deepcopy(critic).to(device=DEVICE, dtype=T)

actor_opt = torch.optim.AdamW(actor.parameters(), LR)
critic_opt = torch.optim.AdamW(critic.parameters(), LR)

replay_buffer = ReplayBuffer(n=N_TRANSITIONS)
action_decay_window = AveragingWindow(maxlen=8)
# visual_decay_window = AveragingWindow(maxlen=8)

random_direction = torch.distributions.Normal(torch.tensor([0, 0, 0.5, 0.5]), torch.tensor([1, 1, 0.5, 0.5]))
print('Starting loop')
environment.game_ipc.set_flag(FLAGS.IS_TRAINING, True)
n = 1
i = 0
while True:
    # Interact with the environment using actor network
    with torch.no_grad():
        *S, DMG = environment.observe()
        F = DMG == 0

        # visual_decay_window.append(S[0])
        # S[0] = visual_decay_window.collate(exponential_decay)

        with torch.autocast(device_type='cuda'):
            noise = random_direction.sample().to(device=DEVICE, dtype=T) * (1 / math.log(n + 2))
            action = actor(*S)
            action = action
            action = torch.cat((torch.clamp(action[:, :2], min=-1, max=1), torch.clamp(action[:, 2:], min=0, max=1)), dim=1).to(device=DEVICE, dtype=T)
            # print(action.view(-1).tolist())

        action_decay_window.append(action.to('cpu'))
        action = action_decay_window.collate()

        *NS, DMG = environment.act(action)
        reward = torch.clamp(torch.dot(torch.tensor(S[-1]).view(-1), torch.tensor(S[-2]).view(-1)).unsqueeze(0), min=0) if DMG == 0 else torch.tensor([0])
        print(reward.item())
        F = torch.tensor([F])
        S, NS, A, R, F = tuple(s.to('cpu') for s in S), tuple(s.to('cpu') for s in NS), action.to('cpu'), reward.to('cpu'), F.to('cpu')
        replay_buffer += Transition(S, A, NS, R, F)
        n += 1

    # If there's a collision, end the episode & update the models
    if DMG > 0 or i > 1000:
        i = 0
        environment.game_ipc.set_flag(FLAGS.REQUEST_GAME_STATE, True)
        environment.game_ipc.set_flag(FLAGS.IS_TRAINING, False)
        print(RANDOM_EXPLORE_COUNT)
        for n in range(N_UPDATES):
            gc.collect()
            torch.cuda.empty_cache()

            # Sample experiences from replay buffer
            batch = replay_buffer.sample(BATCH_SIZE)

            # Get 'best estimate' from target networks
            with torch.autocast(device_type='cuda'):
                ACT_T  = actor_target(*batch.NS)
                CRT_T  = critic_target(*batch.NS, ACT_T)
                # Reward is 1 unless the car collided, which indicates a terminating state
                TARGET = batch.R + (batch.F * GAMMA * CRT_T)

            # Update critic against best estimate via gradient descent
            critic_opt.zero_grad()
            CRIT_LOSS = (critic(*batch.S, batch.A) - TARGET).square().mean(dim=0).sum()
            CRIT_LOSS.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_opt.step()

            # Update actor against best estimate via gradient ascent
            actor_opt.zero_grad()
            ACT = actor(*batch.S)
            ACTOR_LOSS = -critic(*batch.S, ACT).mean(dim=0).sum()
            # ACTOR_LOSS += 0.05 * (RANDOM_EXPLORE_COUNT == 0) * torch.abs(ACT - batch.S[1]).sum()
            ACTOR_LOSS.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()

            # Update target network parameters
            DriverActorModel.soft_update(actor_target, actor, POLYAK)
            DriverCriticModel.soft_update(critic_target, critic, POLYAK)

            print('done')

        environment.game_ipc.set_flag(FLAGS.IS_TRAINING, True)
        RANDOM_EXPLORE_COUNT -= 1 * (RANDOM_EXPLORE_COUNT > 0)        
        replay_buffer.reset()
        # visual_decay_window.reset()
        action_decay_window.reset()
        # random_direction = torch.distributions.Normal(environment.random_action(), torch.ones(4))
