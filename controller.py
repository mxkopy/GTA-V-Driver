import vgamepad as vg
import XInput as xinput
import math
import msgs_pb2
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
            self.physical_controller_state ^= msgs_pb2.ControllerState(left_joystick_x=event.x, left_joystick_y=event.y)
            # TODO: let's assume we're working with lists or tuples outside of ipc.py so we can use this prettier syntax again
            # self.physical_controller_state ^= (event.x, event.y, None, None)
        self.physical_controller_state.set_flag(FLAGS.INPUT_WRITTEN, True)

    def process_connection_event(self, event):
        pass
        # if event.type == xinput.EVENT_DISCONNECTED:
            # self.physical_controller_state ^= (0.0, 0.0, 0.0, 0.0)

    def process_trigger_event(self, event: xinput.Event):
        if event.trigger == xinput.LEFT:
            self.physical_controller_state ^= msgs_pb2.ControllerState(left_trigger=event.value)
            # self.physical_controller_state ^= (None, None, event.value, None)
        if event.trigger == xinput.RIGHT:
            self.physical_controller_state ^= msgs_pb2.ControllerState(right_trigger=event.value)
            # self.physical_controller_state ^= (None, None, None, event.value)
        self.physical_controller_state.set_flag(FLAGS.INPUT_WRITTEN, True)

    def clamp(x: float, min_val: float=-1, max_val: float=1):
        return min(max(0 if math.isnan(x) else x, min_val), max_val)

    def clamp_controller(x: float, y: float, lt: float, rt: float):
        return ControllerHandler.clamp(x), ControllerHandler.clamp(y), ControllerHandler.clamp(lt, min_val=0), ControllerHandler.clamp(rt, min_val=0)

    def virtual_controller_update(self, x: float, y: float, lt: float, rt: float):
        self.output_pad.left_joystick_float(ControllerHandler.clamp(x), ControllerHandler.clamp(y))
        self.output_pad.left_trigger_float(ControllerHandler.clamp(lt, min_val=0))
        self.output_pad.right_trigger_float(ControllerHandler.clamp(rt, min_val=0))
        self.output_pad.update()

    def virtual_controller_update_thread(self):
        while True:
            # self.virtual_controller_state.push(self.physical_controller_state.pop())
            action = self.virtual_controller_state.pop().to_tuple()
            self.virtual_controller_update(*action)

    def physical_controller_update_thread(self, exit=True):
        controller_handler = ControllerHandler()
        xinput.GamepadThread(controller_handler)
        while not exit:
            pass

if __name__ == '__main__':
    import sys
    if sys._is_gil_enabled():
        print("Warning: running controller loop with GIL")
    controller_handler = ControllerHandler()
    controller_handler.physical_controller_update_thread()
    controller_handler.virtual_controller_update_thread()