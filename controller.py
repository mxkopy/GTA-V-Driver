import vgamepad as vg
import XInput as xinput
import math
import time
from msgs_pb2 import ControllerState
from time import sleep
from ipc import VirtualControllerState, PhysicalControllerState, FLAGS


# Performs virtual input actions & allows polling of controller state
# TODO: Run this is another process so you don't have to restart the game 
# every time you want to change the code (the game won't recognize the reconnected controller)
class ControllerHandler(xinput.EventHandler):

    def __init__(self):
        self.controller = self.get_connected_controller()
        self.output_pad = self.initialize_output_pad()
        super().__init__(self.controller)
        self.set_filter(xinput.FILTER_NONE)    
        self.virtual_controller_state = VirtualControllerState()
        self.physical_controller_state = PhysicalControllerState()
        self.disconnected = False

    def get_connected_controller(self):
        print("\nWaiting for controller")
        connected = xinput.get_connected()
        while len(list(filter(None, connected))) == 0:
            sleep(1)
            connected = xinput.get_connected()
        self.controller = [i for i, c in enumerate(connected) if c][0]
        return self.controller

    def initialize_output_pad(self):
        print("\nInitializing output pad")  
        self.output_pad = None
        while self.output_pad is None:
            try:
                self.output_pad = vg.VX360Gamepad()
            except Exception as e:
                print(e)
                self.output_pad = None
                sleep(1)
        return self.output_pad

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

    def process_connection_event(self, event):
        if event.type == xinput.EVENT_DISCONNECTED:
            self.disconnected = True
            self.physical_controller_state ^= ControllerState.init()
        if event.type == xinput.EVENT_CONNECTED:
            self.disconnected = False

    def process_trigger_event(self, event: xinput.Event):
        pass

    def clamp(x: float, min_val: float=-1, max_val: float=1):
        return min(max(0 if math.isnan(x) else x, min_val), max_val)

    def clamp_controller(x: float, y: float, lt: float, rt: float):
        return ControllerHandler.clamp(x), ControllerHandler.clamp(y), ControllerHandler.clamp(lt, min_val=0), ControllerHandler.clamp(rt, min_val=0)

    def virtual_controller_update(self, action: tuple[float, float, float, float]):
        x, y, lt, rt = action
        self.output_pad.left_joystick_float(ControllerHandler.clamp(x), ControllerHandler.clamp(y))
        self.output_pad.left_trigger_float(ControllerHandler.clamp(lt, min_val=0))
        self.output_pad.right_trigger_float(ControllerHandler.clamp(rt, min_val=0))
        self.output_pad.update()

    def virtual_controller_update_thread(self):
        while True:
            if not self.disconnected:
                xinput_state = xinput.get_state(self.controller)
                (left_joystick_x, left_joystick_y), _ = xinput.get_thumb_values(xinput_state)
                left_trigger, right_trigger = xinput.get_trigger_values(xinput_state)
                self.physical_controller_state ^= ControllerState(
                    left_joystick_x=left_joystick_x, 
                    left_joystick_y=left_joystick_y,
                    left_trigger=left_trigger, 
                    right_trigger=right_trigger
                )
            action = self.virtual_controller_state.pop()
            self.virtual_controller_update(action)

    def physical_controller_update_thread(self):
        return xinput.GamepadThread(self)

    def start(self):
        thread = self.physical_controller_update_thread()
        self.virtual_controller_update_thread()
        return thread


def initialize_output_pad():
    print("\nInitializing virtual controller")  
    output_pad = None
    while output_pad is None:
        try:
            output_pad = vg.VX360Gamepad()
        except Exception as e:
            print(e)
            sleep(1)
    print("Virtual controller initialized")
    return output_pad

class VirtualController:

    output_pad = initialize_output_pad()

    def update(action: tuple[float, float, float, float]):
        x, y, lt, rt = action
        VirtualController.output_pad.left_joystick_float(ControllerHandler.clamp(x), ControllerHandler.clamp(y))
        VirtualController.output_pad.left_trigger_float(ControllerHandler.clamp(lt, min_val=0))
        VirtualController.output_pad.right_trigger_float(ControllerHandler.clamp(rt, min_val=0))
        VirtualController.output_pad.update()

if __name__ == '__main__':
    ControllerHandler().start()