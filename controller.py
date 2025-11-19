import vgamepad as vg
import XInput as xinput
import math
from time import sleep
from ipc import ControllerIPC, FLAGS

# Performs virtual input actions & allows polling of controller state
# TODO: Run this is another process so you don't have to restart the game 
# every time you want to change the code (the game won't recognize the reconnected controller)
class ControllerHandler(xinput.EventHandler):

    def __init__(self):
        self.controller = self.get_connected_controller()
        super().__init__(self.controller)
        self.set_filter(xinput.FILTER_NONE)    
        self.output_pad = self.initialize_output_pad()
        self.controller_ipc_loop = ControllerIPC()
        self.controller_ipc_main = ControllerIPC()
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
        if event.stick == xinput.LEFT:
            self.update()

    def process_connection_event(self, event):
        if event.type == xinput.EVENT_DISCONNECTED:
            self.disconnected = True
        if event.type == xinput.EVENT_CONNECTED:
            self.disconnected = False

    def process_trigger_event(self, event):
        self.update()

    def clamp(x: float, min_val: float=-1, max_val: float=1):
        return min(max(0 if math.isnan(x) else x, min_val), max_val)

    def clamp_controller(x: float, y: float, lt: float, rt: float):
        return ControllerHandler.clamp(x), ControllerHandler.clamp(y), ControllerHandler.clamp(lt, min_val=0), ControllerHandler.clamp(rt, min_val=0)

    def update(self):
        if self.disconnected:
            self.controller_ipc_main.write_input(0.0, 0.0, 0.0, 0.0)
        else:
            xinput_state = xinput.get_state(self.controller)
            state = xinput.get_thumb_values(xinput_state)[0] + xinput.get_trigger_values(xinput_state)
            self.controller_ipc_main.write_input(*state)

    def act(self, x: float, y: float, lt: float, rt: float):
        self.output_pad.left_joystick_float(ControllerHandler.clamp(x), ControllerHandler.clamp(y))
        self.output_pad.left_trigger_float(ControllerHandler.clamp(lt, min_val=0))
        self.output_pad.right_trigger_float(ControllerHandler.clamp(rt, min_val=0))
        self.output_pad.update()

    def perform_action(self):
        self.act(*self.controller_ipc_loop.pop_action())

    def start_thread():
        gamepad_thread = ControllerHandler()
        xinput.GamepadThread(gamepad_thread)
        while True:
            # gamepad_thread.update()
            # gamepad_thread.perform_action()
            # if gamepad_thread.controller_ipc.get_flag(FLAGS.ACTION_WRITTEN):
            gamepad_thread.perform_action()

if __name__ == '__main__':
    ControllerHandler.start_thread()