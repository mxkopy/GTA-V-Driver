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

# torch.from_dlpack()

# Performs virtual input actions & allows polling of controller state
class ControllerHandler(xinput.EventHandler):

    def __init__(self):
        connected = xinput.get_connected()
        self.controller = [i for i in range(4) if connected[i]][0]
        super().__init__(self.controller)
        self.set_filter(xinput.FILTER_NONE)
        self.output_pad = vg.VX360Gamepad()
        
    def process_button_event(self, event):
        if event.type == xinput.EVENT_BUTTON_PRESSED:
            self.output_pad.press_button(event.button_id)
        if event.type == xinput.EVENT_BUTTON_RELEASED:
            self.output_pad.release_button(event.button_id)
        self.output_pad.update()

    def process_stick_event(self, event):
        if event.stick == xinput.RIGHT:
            self.output_pad.right_joystick_float(event.x, event.y)
            self.output_pad.update()

    def perform_action(self, action):
        action = action.squeeze()
        self.output_pad.left_joystick_float(min(max(action[0].item(), -1), 1), min(max(action[1].item(), -1), 1))
        self.output_pad.left_trigger_float(min(max(action[2].item(), 0), 1))
        self.output_pad.right_trigger_float(min(max(action[3].item(), 0), 1))
        self.output_pad.update()

    def poll_state(self):
        state = xinput.get_state(self.controller)
        return torch.tensor(xinput.get_thumb_values(state)[0] + xinput.get_trigger_values(state)).to(device=DEVICE, dtype=T)

    def process_connection_event(self, event):
        pass

    def process_trigger_event(self, event):
        pass

ACK_END = 1
CAM_END = 13
VEL_END = 25
DMG_END = 29

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

    def __init__(self, IPC_BYTES=DMG_END, img_resolution=IMG_RESOLUTION, time_delay_frame_dt=4, n_time_delays=3):
        self.ipc = mmap.mmap(-1, IPC_BYTES, "ipc.mem")
        self.gamepad = ControllerHandler()
        self.img_resolution = img_resolution
        self.sct = bettercam.create(device_idx=0, nvidia_gpu=True)
        self.fb = TimeDelayFrameBuffer(img_resolution=IMG_RESOLUTION, n=time_delay_frame_dt*n_time_delays+1, k=n_time_delays, dt=time_delay_frame_dt)
        # self.sct.start()
        xinput.GamepadThread(self.gamepad)
    
    def _observe(self):
        return [self.grab_screenshot()] + [torch.zeros(1, 4, dtype=T, device=DEVICE), torch.zeros(1, 3, dtype=T, device=DEVICE), torch.zeros(1, 3, dtype=T, device=DEVICE), torch.tensor([0], dtype=T, device=DEVICE)]

    def observe(self):
        self.ipc.seek(0)
        while not (1 & self.ipc.read_byte()):
            self.ipc.seek(0)
        PCKT = b'1' + self.ipc.read(DMG_END)
        CAM = struct.unpack('<3f', PCKT[ACK_END:CAM_END])
        VEL = struct.unpack('<3f', PCKT[CAM_END:VEL_END])
        DMG = struct.unpack('i', PCKT[VEL_END:DMG_END])
        IMG, INP, CAM, VEL, DMG = [self.grab_screenshot()] + [torch.tensor(x, dtype=T, device=DEVICE).unsqueeze(0) for x in (self.gamepad.poll_state(), CAM, VEL, DMG)]
        return IMG, INP, CAM, VEL, DMG

    def act(self, action):
        self.gamepad.perform_action(action)
        self.ipc.seek(0)
        self.ipc.write_byte(0)
        # TODO: wait on game update
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
    



torch.cuda.empty_cache()

# Implements https://spinningup.openai.com/en/latest/algorithms/ddpg.html

GAMMA = 0.990
POLYAK = 0.995

N_TRANSITIONS = 16
BATCH_SIZE = N_TRANSITIONS
N_UPDATES = 4

LR = 1e-2

ENV = Environment(img_resolution=IMG_RESOLUTION)

actor = DriverActorModel().to(device=DEVICE, dtype=T)
critic = DriverCriticModel().to(device=DEVICE, dtype=T)
try:
    actor.load_state_dict(torch.load('actor.pt'))
except:
    actor = actor.train_for_input_passthrough()
    torch.save(actor.state_dict(), 'actor.pt')

# import numpy as np
# Image.fromarray((actor.visual(ENV.grab_screenshot()).squeeze().cpu().detach().numpy() * 255).astype(np.uint8)).show()

actor_target = DriverActorModel(img_resolution=IMG_RESOLUTION).to(device=DEVICE, dtype=T)
critic_target = DriverCriticModel(img_resolution=IMG_RESOLUTION).to(device=DEVICE, dtype=T)

torch.nn.utils.vector_to_parameters(torch.nn.utils.parameters_to_vector(actor.parameters()), actor_target.parameters())
torch.nn.utils.vector_to_parameters(torch.nn.utils.parameters_to_vector(critic.parameters()), critic_target.parameters())

actor_opt = torch.optim.AdamW(actor.parameters(), LR)
critic_opt = torch.optim.AdamW(critic.parameters(), LR)

replay_buffer = ReplayBuffer(n=N_TRANSITIONS)
actor_buffer = TimeDelayFrameBuffer(n=12)

print('starting loop')

import time
while True:
    # Interact with the environment using actor network
    with torch.no_grad():
        # a = time.time()
        *S, _ = ENV.observe()
        # b = time.time()
        with torch.autocast(device_type='cuda'):
            A = actor(*S)
        *NS, DMG = ENV.act(A)
        REW = torch.tensor([-100]) if DMG != 0 else torch.tensor([0])
        S, NS, A, REW = tuple(s.to('cpu') for s in S), tuple(s.to('cpu') for s in NS), A.to('cpu'), REW.to('cpu')
        replay_buffer += Transition(S, A, NS, REW)
        print(torch.nn.functional.mse_loss(A, S[1]).item())
        # print(b - a)
        
    # If the buffer is full, update the models
    if replay_buffer.idx == N_TRANSITIONS:
    # if buffer.tail == N_TRANSITIONS-1:

        for n in range(N_UPDATES):

            torch.cuda.empty_cache()

            # Sample indices from circular buffer
            batch = replay_buffer.sample(BATCH_SIZE)

            # Get 'best estimate' from target networks
            with torch.autocast(device_type='cuda'):
                ACT_T  = actor_target(*batch.NS)
                CRT_T  = critic_target(*batch.NS, ACT_T)
                TARGET = batch.R + (GAMMA * CRT_T)  

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
            # ACTOR_LOSS += torch.nn.functional.mse_loss(ACT * (batch.R != 0), batch.S[1] * (batch.R != 0))
            ACTOR_LOSS.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()

            # Update target network parameters
            ACTOR_PARAMS    = torch.nn.utils.parameters_to_vector(actor.parameters())
            CRITIC_PARAMS   = torch.nn.utils.parameters_to_vector(critic.parameters())
            ACTOR_T_PARAMS  = torch.nn.utils.parameters_to_vector(actor_target.parameters())
            CRITIC_T_PARAMS = torch.nn.utils.parameters_to_vector(critic_target.parameters())
            torch.nn.utils.vector_to_parameters((ACTOR_T_PARAMS * POLYAK) + (1 - POLYAK) * ACTOR_PARAMS, actor_target.parameters())
            torch.nn.utils.vector_to_parameters((CRITIC_T_PARAMS * POLYAK) + (1 - POLYAK) * CRITIC_PARAMS, critic_target.parameters())

        replay_buffer.reset()