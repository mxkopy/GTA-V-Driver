import vgamepad as vg
import XInput as xinput
import mmap
import struct
import mss
from time import sleep
import torch
import torchvision.io
import random
import copy
from model import DriverActorModel, DriverCriticModel, T, DEVICE
from PIL import Image
from collections import namedtuple


X2VIG = {
    xinput.BUTTON_DPAD_DOWN: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
    xinput.BUTTON_DPAD_UP: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
    xinput.BUTTON_DPAD_LEFT: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
    xinput.BUTTON_DPAD_RIGHT: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
    xinput.BUTTON_A: vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    xinput.BUTTON_B: vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    xinput.BUTTON_X: vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    xinput.BUTTON_Y: vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
}


# TODO: synchronize this stuff with the environment. 
# Ideally the game waits for an action, and the AI polls game & input state as needed.
# The game should also keep a state buffer so the user's input during training or inference isn't recorded
class ControllerHandler(xinput.EventHandler):

    def __init__(self, lp_model=None):
        connected = xinput.get_connected()
        controller = [i for i in range(4) if connected[i]][0]
        super().__init__(controller)
        self.set_filter(xinput.FILTER_NONE)
        self.output_pad = vg.VX360Gamepad()
        self.lp_model = lp_model
        self.move_inputs = [0.0, 0.0, 0.0, 0.0]
        self.action = [None, None, None, None]
        
    def process_button_event(self, event):
        if event.type == xinput.EVENT_BUTTON_PRESSED:
            self.output_pad.press_button(X2VIG[event.button_id])
        if event.type == xinput.EVENT_BUTTON_RELEASED:
            self.output_pad.release_button(X2VIG[event.button_id])
        self.output_pad.update()

    def process_trigger_event(self, event):
        if event.trigger == xinput.LEFT:
            self.move_inputs[2] = event.value
        if event.trigger == xinput.RIGHT:
            self.move_inputs[3] = event.value
        if self.action[2] is not None:
            self.output_pad.left_trigger_float(self.action[2])
            self.action[2] = None
            self.output_pad.update()
        if self.action[3] is not None:
            self.output_pad.right_trigger_float(self.action[3])
            self.action[3] = None
            self.output_pad.update()

    def process_stick_event(self, event):
        if event.stick == xinput.LEFT:
            self.move_inputs[0] = event.x
            self.move_inputs[1] = event.y
            if self.action[0] is not None and self.action[1] is not None:
                self.output_pad.left_joystick_float(self.action[0], self.action[1])
                self.action[0] = None
                self.action[1] = None
                self.output_pad.update()
        if event.stick == xinput.RIGHT:
            self.output_pad.right_joystick_float(event.x, event.y)
            self.output_pad.update()

    def process_connection_event(self, event):
        pass


ACK_END = 1
CAM_END = 13
VEL_END = 25
DMG_END = 29

class Environment:

    def __init__(self, IPC_BYTES=VEL_END, resolution=(720, 1280)):
        self.ipc = mmap.mmap(-1, IPC_BYTES, "ipc.mem")
        self.gamepad = ControllerHandler()
        self.resolution = resolution
        xinput.GamepadThread(self.gamepad)
    
    def observe_base(self):
        return [self.grab_screenshot()] + [torch.zeros(1, 4, dtype=T, device=DEVICE), torch.zeros(1, 3, dtype=T, device=DEVICE), torch.zeros(1, 3, dtype=T, device=DEVICE), torch.tensor([0], dtype=T, device=DEVICE)]

    def _observe_base(self):
        while not 1 & self.ipc.read_byte():
            self.ipc.seek(0)
        self.ipc.seek(0)
        PCKT = self.ipc.read_byte() + self.ipc.read(DMG_END - ACK_END)
        CAM = struct.unpack('<3f', PCKT[ACK_END:CAM_END])
        VEL = struct.unpack('<3f', PCKT[CAM_END:VEL_END])
        DMG = struct.unpack('I', PCKT[VEL_END:DMG_END])
        REW = 1 if DMG == 0 else -100
        IMG, INP, CAM, VEL, REW = [self.grab_screenshot()] + [torch.tensor(x, dtype=T, device=DEVICE) for x in (self.gamepad.move_inputs, CAM, VEL, REW)]
        return IMG, INP, CAM, VEL, REW
    
    def observe(self):
        *S, FIN = self.observe_base()
        # This a rollie not a stop watch
        # Shit don't ever stop
        return *S, 0

    def act(self, action):
        self.gamepad.action = action.tolist()
        self.ipc.seek(0)
        self.ipc.write_byte(0)
        # TODO: spin on game update
        return self.observe_base()

    def grab_screenshot(self):
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", img.size, img.rgb)
            img = img.resize(self.resolution)
            return torchvision.transforms.functional.pil_to_tensor(img).to(device=DEVICE, dtype=T).unsqueeze(0)


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
        batch['S'] = tuple(batch['S'])
        batch['NS'] = tuple(batch['NS'])
        return Transition(batch['S'], batch['A'], batch['NS'], batch['R'])

    def reset(self):
        self.buffer = [None for _ in range(self.N)]
        self.head = 0
        self.tail = 0


torch.cuda.empty_cache()

# Implements https://spinningup.openai.com/en/latest/algorithms/ddpg.html

actor = DriverActorModel().to(device=DEVICE, dtype=T)
critic = DriverCriticModel().to(device=DEVICE, dtype=T)

actor_target = copy.deepcopy(actor).to(device=DEVICE, dtype=T)
critic_target = copy.deepcopy(critic).to(device=DEVICE, dtype=T)

actor_opt = torch.optim.Adam(actor.parameters(), 1e-2)
critic_opt = torch.optim.Adam(critic.parameters(), 1e-2)

ENV = Environment()

GAMMA = 0.990
POLYAK = 0.995
N_TRANSITIONS = 10
BATCH_SIZE = 4
N_UPDATES = 10

buffer = CircularReplayBuffer()

import time
while True:
    # Interact with the environment using actor network
    with torch.no_grad():
        a = time.time()

        *S, FIN = ENV.observe()
        A = actor(*S)
        *NS, REW = ENV.act(A)

        buffer += Transition(S, A, NS, REW)

        b = time.time()
        print(b - a)
        
    # If the buffer is full, update the models
    if buffer.tail == N_TRANSITIONS-1:

        for n in range(N_UPDATES):

            # Sample indices from circular buffer
            batch = buffer.sample(BATCH_SIZE)
            print('sampled')

            # Get 'best estimate' from target networks
            ACT_T  = actor_target(*batch.NS)
            CRT_T  = critic_target(*batch.NS, ACT_T)
            TARGET = REW + GAMMA * (1 - FIN) * CRT_T

            print('target')

            # Update critic against best estimate via gradient descent
            critic_opt.zero_grad()
            CRIT_LOSS = (critic(*batch.S, batch.A) - TARGET).square().sum() / BATCH_SIZE
            CRIT_LOSS.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_opt.step()

            print('critic opt')

            # Update actor against best estimate via gradient ascent
            actor_opt.zero_grad()
            ACTOR_LOSS = -critic(*batch.S, actor(*batch.S)).sum() / BATCH_SIZE
            ACTOR_LOSS.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()

            print('actor opt')

            # Update target network parameters
            ACTOR_PARAMS    = torch.nn.utils.parameters_to_vector(actor.parameters())
            CRITIC_PARAMS   = torch.nn.utils.parameters_to_vector(critic.parameters())
            ACTOR_T_PARAMS  = torch.nn.utils.parameters_to_vector(actor_target.parameters())
            CRITIC_T_PARAMS = torch.nn.utils.parameters_to_vector(critic_target.parameters())
            torch.nn.utils.vector_to_parameters((ACTOR_T_PARAMS * POLYAK) + (1 - POLYAK) * ACTOR_PARAMS, actor_target.parameters())
            torch.nn.utils.vector_to_parameters((CRITIC_T_PARAMS * POLYAK) + (1 - POLYAK) * CRITIC_PARAMS, critic_target.parameters())

            print('update target')

        buffer.reset()