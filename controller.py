import vgamepad as vg
import XInput as xinput
import struct
from time import sleep
from ipc import ControllerIPC

# Performs virtual input actions & allows polling of controller state
# TODO: Run this is another process so you don't have to restart the game 
# every time you want to change the code (the game won't recognize the reconnected controller)
class ControllerHandler(xinput.EventHandler):

    def get_connected_controller(self):
        connected = xinput.get_connected()
        while len(list(filter(None, connected))) == 0:
            sleep(1)
            connected = xinput.get_connected()
        self.controller = [i for i, c in enumerate(connected) if c][0]
        return self.controller

    def initialize_output_pad(self):
        self.output_pad = None
        while self.output_pad is None:
            try:
                self.output_pad = vg.VX360Gamepad()
            except Exception as e:
                print(e)
                self.output_pad = None
                sleep(1)
        return self.output_pad

    def __init__(self):
        print("\nWaiting for controller")
        self.controller = self.get_connected_controller()
        super().__init__(self.controller)
        self.set_filter(xinput.FILTER_NONE)
        print("\nInitializing output pad")      
        self.output_pad = self.initialize_output_pad()
        self.controller_ipc = ControllerIPC()

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
        pass

    def process_trigger_event(self, event):
        pass

    def update_state(self):
        xinput_state = xinput.get_state(self.controller)
        state = xinput.get_thumb_values(xinput_state)[0] + xinput.get_trigger_values(xinput_state)
        self.controller_ipc.write_state(*state)

    def perform_action(self):
        self.update_state()
        action = self.controller_ipc.get_action()
        self.update_state()
        self.output_pad.left_joystick_float(min(max(action[0], -1), 1), min(max(action[1], -1), 1))
        self.output_pad.left_trigger_float(min(max(action[2], 0), 1))
        self.output_pad.right_trigger_float(min(max(action[3], 0), 1))
        self.output_pad.update()

    def start_thread(self):
        gamepad_thread = ControllerHandler()
        xinput.GamepadThread(gamepad_thread)
        while True:
            gamepad_thread.perform_action()

ControllerHandler().start_thread()