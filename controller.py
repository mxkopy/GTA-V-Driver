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
from model import DriverActorModel, DriverCriticModel, T, DEVICE
from PIL import Image
from collections import namedtuple

# torch.from_dlpack()

# Performs virtual input actions & allows polling of controller state
class ControllerHandler(xinput.EventHandler):

    def __init__(self):
        connected = xinput.get_connected()
        controller = [i for i in range(4) if connected[i]][0]
        super().__init__(controller)
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
        self.output_pad.left_joystick_float(max(min(action[0].item(), -1), 1), max(min(action[1].item(), -1), 1))
        self.output_pad.left_trigger_float(max(min(action[2].item(), 0), 1))
        self.output_pad.right_trigger_float(max(min(action[3].item(), 0), 1))
        self.output_pad.update()

    def poll_state(self):
        state = xinput.get_state()
        return torch.tensor(xinput.get_thumb_values(state)[0] + xinput.get_trigger_values(state)).to(device=DEVICE, dtype=T)

    def process_connection_event(self, event):
        pass


ACK_END = 1
CAM_END = 13
VEL_END = 25
DMG_END = 29

from bettercam.processor.cupy_processor import CupyProcessor

def process_cvtcolor(self, image):
    return image

CupyProcessor.process_cvtcolor = process_cvtcolor

class Environment:

    def __init__(self, IPC_BYTES=DMG_END, img_resolution=(720, 1280)):
        self.ipc = mmap.mmap(-1, IPC_BYTES, "ipc.mem")
        self.gamepad = ControllerHandler()
        self.img_resolution = img_resolution
        self.sct = bettercam.create(device_idx=0, nvidia_gpu=True)
        # self.sct.start()
        xinput.GamepadThread(self.gamepad)
    
    def observe(self):
        return [self.grab_screenshot()] + [torch.zeros(1, 4, dtype=T, device=DEVICE), torch.zeros(1, 3, dtype=T, device=DEVICE), torch.zeros(1, 3, dtype=T, device=DEVICE), torch.tensor([0], dtype=T, device=DEVICE)]

    def _observe(self):
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
        img = img.unsqueeze(0).to(device=DEVICE, dtype=T)
        img = torch.nn.functional.adaptive_avg_pool2d(img, self.img_resolution)
        # Amazing one-liner
        # Image.fromarray(img.squeeze().cpu().numpy(), mode='F').show()
        return img


Transition = namedtuple('Transition', ('S', 'A', 'NS', 'R'))

class CircularReplayBuffer:

    def __init__(self, N=300):
        self.N = N
        self.buffer = [None for _ in range(N)]
        self.head = 0
        self.tail = 0

    def __iadd__(self, transition: Transition):
        self.buffer[self.tail] = transition
        self.tail = (self.tail+1) % self.N
        self.head = (self.tail+1) % self.N if self.head == self.tail else self.head
        return self

    def __getitem__(self, key):
        return self.buffer[key]
    
    def __setitem__(self, key, value):
        self.buffer[key] = value

    def add(self, transition: Transition):
        self += transition

    # TODO: For this application, sampling randomly is kind of an interesting strategy. It might make more sense to sample
    # contiguous chunks of time instead.
    def sample(self, n):
        transitions = random.sample(list(filter(None, self.buffer)), n)
        batch = transitions[0]._asdict()
        batch['S'] = list(batch['S'])
        batch['NS'] = list(batch['NS'])
        for transition in transitions[1:]:
            for i in range(len(batch['S'])):
                batch['S'][i] = torch.cat((batch['S'][i], transition.S[i]))
                batch['NS'][i] = torch.cat((batch['NS'][i], transition.NS[i]))
            batch['A'] = torch.cat((batch['A'], transition.A))
            batch['R'] = torch.cat((batch['R'], transition.R))
        batch['S'] = tuple(s.to(device=DEVICE, dtype=T) for s in batch['S'])
        batch['A'] = batch['A'].to(device=DEVICE, dtype=T)
        batch['NS'] = tuple(s.to(device=DEVICE, dtype=T) for s in batch['NS'])
        batch['R'] = batch['R'].to(device=DEVICE, dtype=T)
        return Transition(batch['S'], batch['A'], batch['NS'], batch['R'])

    def reset(self):
        self.buffer = [None for _ in range(self.N)]
        self.head = 0
        self.tail = 0


torch.cuda.empty_cache()

IMG_RESOLUTION = (360, 640)

# Implements https://spinningup.openai.com/en/latest/algorithms/ddpg.html

GAMMA = 0.990
POLYAK = 0.995

N_TRANSITIONS = 16
BATCH_SIZE = 4
N_UPDATES = 1

LR = 1e-5

ENV = Environment(img_resolution=IMG_RESOLUTION)

actor = DriverActorModel().to(device=DEVICE, dtype=T)
critic = DriverCriticModel().to(device=DEVICE, dtype=T)
# try:
#     actor.load_state_dict(torch.load('actor.pt'))
# except:
#     actor = actor.train_for_input_passthrough()
#     torch.save(actor.state_dict(), 'actor.pt')

actor_target = DriverActorModel(img_resolution=IMG_RESOLUTION).to(device=DEVICE, dtype=T)
critic_target = DriverCriticModel(img_resolution=IMG_RESOLUTION).to(device=DEVICE, dtype=T)

torch.nn.utils.vector_to_parameters(torch.nn.utils.parameters_to_vector(actor.parameters()), actor_target.parameters())
torch.nn.utils.vector_to_parameters(torch.nn.utils.parameters_to_vector(critic.parameters()), critic_target.parameters())

actor_opt = torch.optim.AdamW(actor.parameters(), LR)
critic_opt = torch.optim.AdamW(critic.parameters(), LR)


buffer = CircularReplayBuffer(N_TRANSITIONS)

import time
while True:
    # Interact with the environment using actor network
    with torch.no_grad():
        a = time.time()
        *S, _ = ENV.observe()
        b = time.time()
        with torch.autocast(device_type='cuda'):
            A = actor(*S)
        *NS, DMG = ENV.act(A)
        REW = -DMG*0
        S, NS, A, REW = tuple(s.to('cpu') for s in S), tuple(s.to('cpu') for s in NS), A.to('cpu'), REW.to('cpu')
        buffer += Transition(S, A, NS, REW)
        # print(torch.nn.functional.mse_loss(A, S[1]).item())
        print(b - a)
        
    # If the buffer is full, update the models
    if False:
    # if buffer.tail == N_TRANSITIONS-1:

        for n in range(N_UPDATES):

            torch.cuda.empty_cache()

            # Sample indices from circular buffer
            batch = buffer.sample(BATCH_SIZE)

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
            ACTOR_LOSS += torch.nn.functional.mse_loss(ACT, batch.S[1])
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

        buffer.reset()