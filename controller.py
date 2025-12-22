import vgamepad as vg
import math
import time
from msgs_pb2 import ControllerState
from time import sleep
from ipc import VirtualControllerState, PhysicalControllerState, FLAGS

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

    def clamp(x: float, min_val: float=-1, max_val: float=1):
        return min(max(0 if math.isnan(x) else x, min_val), max_val)

    def update(action: tuple[float, float, float, float]):
        x, y, lt, rt = action
        VirtualController.output_pad.left_joystick_float(VirtualController.clamp(x), VirtualController.clamp(y))
        VirtualController.output_pad.left_trigger_float(VirtualController.clamp(lt, min_val=0))
        VirtualController.output_pad.right_trigger_float(VirtualController.clamp(rt, min_val=0))
        VirtualController.output_pad.update()
