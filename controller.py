import vgamepad as vg
import XInput as xinput
import mmap
import struct
import mss
from time import sleep
import torch
import torchvision.io
from model import DriverModel, T, DEVICE
from PIL import Image


ACK_DAM_END = 1
CAM_END = 13
VEL_END = 25

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

class ControllerHandler(xinput.EventHandler):

    def __init__(self, lp_model=None):
        connected = xinput.get_connected()
        controller = [i for i in range(4) if connected[i]][0]
        super().__init__(controller)
        self.set_filter(xinput.FILTER_NONE)
        self.output_pad = vg.VX360Gamepad()
        self.ipc = mmap.mmap(-1, VEL_END, 'ipc.mem')
        self.lp_model = lp_model
        self.move_inputs   = [0.0, 0.0, 0.0, 0.0]
        self.move_computed = [None, None, None, None]
        
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
        if self.move_computed[2] is not None:
            self.output_pad.left_trigger_float(self.move_computed[2])
            self.move_computed[2] = None
            self.output_pad.update()
        if self.move_computed[3] is not None:
            self.output_pad.right_trigger_float(self.move_computed[3])
            self.move_computed[3] = None
            self.output_pad.update()

    def process_stick_event(self, event):
        if event.stick == xinput.LEFT:
            self.move_inputs[0] = event.x
            self.move_inputs[1] = event.y
            if self.move_computed[0] is not None and self.move_computed[1] is not None:
                self.output_pad.left_joystick_float(self.move_computed[0], self.move_computed[1])
                self.move_computed[0] = None
                self.move_computed[1] = None
                self.output_pad.update()
        if event.stick == xinput.RIGHT:
            self.output_pad.right_joystick_float(event.x, event.y)
            self.output_pad.update()

    def process_connection_event(self, event):
        pass

torch.cuda.empty_cache()
model = DriverModel().to(DEVICE, T)
opt = torch.optim.Adam(model.parameters(), 1e-4)
ipc = mmap.mmap(-1, VEL_END, "ipc.mem")
gamepad = ControllerHandler()
xinput.GamepadThread(gamepad)
relu = lambda x: x * (x > 0)

while True:
    ipc.seek(0)
    ACK_DAM = ipc.read_byte()
    if 1 & ACK_DAM:
        with mss.mss() as sct:
            PCKT = ACK_DAM.to_bytes() + ipc.read(VEL_END - ACK_DAM_END)
            DMG = ACK_DAM >> 1
            CAM = struct.unpack('<3f', PCKT[ACK_DAM_END:CAM_END])
            VEL = struct.unpack('<3f', PCKT[CAM_END:VEL_END])
            img = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", img.size, img.rgb)
            IMG = torchvision.transforms.functional.pil_to_tensor(img).to(device=DEVICE, dtype=T).unsqueeze(0)
            MVE = gamepad.move_inputs
            CAM, VEL, MVE = [torch.tensor(x, dtype=T, device=DEVICE) for x in (CAM, VEL, MVE)]
            output = model(IMG, CAM, VEL, MVE)
            loss = torch.nn.functional.mse_loss(output, torch.tensor(gamepad.move_inputs, device=DEVICE, dtype=T))
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss.item())
            gamepad.move_computed = output.detach().clone().tolist()
            ipc.seek(0)
            ipc.write_byte(0)

# while True:
    # sleep(0.01)

# my_handler = MyHandler(1)
# my_gamepad_thread = xinput.GamepadThread(my_handler)
# my_gamepad_thread.start()
# while True:
    # pass