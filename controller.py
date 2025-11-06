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
# Ideally the game waits for an action, and the actor polls game & input state as needed.
# Should provide the actor with the last state & input recorded before it starts processing. 
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

    def __init__(self, IPC_BYTES=VEL_END):
        self.ipc = mmap.mmap(-1, IPC_BYTES, "ipc.mem")
        self.gamepad = ControllerHandler()
        xinput.GamepadThread(self.gamepad)
    
    def observe(self):
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

    def act(self, action):
        self.gamepad.action = action.detach().clone().tolist()
        self.ipc.seek(0)
        self.ipc.write_byte(0)

    def grab_screenshot(self):
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", img.size, img.rgb)
            return torchvision.transforms.functional.pil_to_tensor(img).to(device=DEVICE, dtype=T)

torch.cuda.empty_cache()

# Implements https://spinningup.openai.com/en/latest/algorithms/ddpg.html

actor = DriverActorModel().to(DEVICE, T)
critic = DriverCriticModel().to(DEVICE, T)

actor_target = copy.deepcopy(actor).to(DEVICE, T)
critic_target = copy.deepcopy(critic).to(DEVICE, T)

actor_opt = torch.optim.Adam(actor.parameters(), 1e-2)
critic_opt = torch.optim.Adam(critic.parameters(), 1e-2)

ENV = Environment()

GAMMA = 0.990
POLYAK = 0.995
N_TRANSITIONS = 600
BATCH_SIZE = 4
N_UPDATES = 100

HEAD = 0
TAIL = 0

# State: (IMG, CAM, VEL, INP)
# S  -> BUFFER[0, ...]
# S' -> BUFFER[1, ...]
IMG_BUFFER = torch.zeros(2, N_TRANSITIONS, *ENV.grab_screenshot().shape, dtype=T, device=DEVICE)
CAM_BUFFER = torch.zeros(2, N_TRANSITIONS, 3, dtype=T, device=DEVICE)
VEL_BUFFER = torch.zeros(2, N_TRANSITIONS, 3, dtype=T, device=DEVICE)
INP_BUFFER = torch.zeros(2, N_TRANSITIONS, 4, dtype=T, device=DEVICE)

# Action: ~INP
ACT_BUFFER = torch.zeros(N_TRANSITIONS, 4, dtype=T, device=DEVICE)

# Reward: f(DMG)
REW_BUFFER = torch.zeros(N_TRANSITIONS, dtype=T, device=DEVICE)

# Final flag: {0, 1}
FIN_BUFFER = torch.zeros(N_TRANSITIONS, dtype=T, device=DEVICE)

while True:
    # Interact with the environment using actor network
    with torch.no_grad():
        IMG, INP, CAM, VEL, REW = ENV.observe()
        FIN = REW != 1
        IMG_BUFFER[0, TAIL, ...], INP_BUFFER[0, TAIL, ...], CAM_BUFFER[0, TAIL, ...], VEL_BUFFER[0, TAIL, ...], FIN_BUFFER[TAIL] = IMG, INP, CAM, VEL, FIN
        ACT = actor(IMG, INP, CAM, VEL)
        ENV.act(ACT)
        ACT_BUFFER[TAIL, ...] = ACT
        IMG, INP, CAM, VEL, REW = ENV.observe()
        IMG_BUFFER[1, TAIL, ...], INP_BUFFER[1, TAIL, ...], CAM_BUFFER[1, TAIL, ...], VEL_BUFFER[1, TAIL, ...], REW_BUFFER[TAIL] = IMG, INP, CAM, VEL, REW
        TAIL = (TAIL+1) % N_TRANSITIONS
        HEAD = (TAIL+1) % N_TRANSITIONS if HEAD == TAIL else HEAD

    # If the interaction episode is over, update the models
    if FIN:

        # Calculate indices population for circular buffer
        if TAIL == HEAD:
            IDX = list(range(N_TRANSITIONS))
        elif TAIL < HEAD:
            IDX = list(range(HEAD, N_TRANSITIONS)) + list(range(0, TAIL+1))
        elif HEAD < TAIL:
            IDX = list(range(HEAD, TAIL+1))
        HEAD = 0
        TAIL = 0

        # TODO: For this application, sampling randomly is kind of an interesting strategy. It might make more sense to sample
        # contiguous chunks of time instead. 
        for n in N_UPDATES:

            # Sample indices from circular buffer
            SAMPLE = random.sample(IDX, BATCH_SIZE)

            IMG, INP, CAM, VEL = [BUFFER[:, SAMPLE, ...] for BUFFER in (IMG_BUFFER, INP_BUFFER, CAM_BUFFER, VEL_BUFFER)]
            ACT, FIN, REW      = ACT_BUFFER[SAMPLE, ...], FIN_BUFFER[SAMPLE], REW_BUFFER[SAMPLE]

            STATE      = IMG[0, :, ...].squeeze(), INP[0, :, ...].squeeze(), CAM[0, :, ...].squeeze(), VEL[0, :, ...].squeeze()
            NEXT_STATE = IMG[1, :, ...].squeeze(), INP[1, :, ...].squeeze(), CAM[1, :, ...].squeeze(), VEL[1, :, ...].squeeze()

            # Get 'best estimate' from target networks
            ACT_T  = actor_target(*NEXT_STATE)
            CRT_T  = critic_target(*NEXT_STATE, ACT_T)
            TARGET = REW + GAMMA * (1 - FIN) * CRT_T

            # Update critic against best estimate via gradient descent
            critic_opt.zero_grad()
            CRIT_LOSS = (critic(*STATE, ACT[:, ...].squeeze()) - TARGET).square().sum() / BATCH_SIZE
            CRIT_LOSS.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_opt.step()

            # Update actor against best estimate via gradient ascent
            actor_opt.zero_grad()
            ACTOR_LOSS = -critic(*STATE, actor(*STATE)).sum() / BATCH_SIZE
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