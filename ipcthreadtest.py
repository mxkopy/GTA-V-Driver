import threading
import io
import multiprocessing
import config
import time
import random
from collections.abc import Callable, Buffer
from model import DeterministicPolicyGradient, DriverActorModel, DriverCriticModel
from ipc import GameState, PhysicalControllerState, VirtualControllerState, FLAGS
from controller import ControllerHandler

LOG_FILE = 'thread_execution_order_test.log'

def logging_hook(method: Callable, tag: str | bytes, logfile: str = LOG_FILE ) -> Callable:
    counter = [0]
    def log(*args, **kwargs):
        with io.FileIO(logfile, 'a') as file:
            file.write(f'{counter[0]} {tag} BEGIN\n'.encode())
            file.flush()
            output = method(*args, **kwargs)
            file.write(f'{counter[0]} {tag} END\n'.encode())
            file.flush()
            counter[0] += 1
            return output
    return log

GameState.pop = logging_hook(GameState.pop, 'GR')
DeterministicPolicyGradient.compute_action = logging_hook(DeterministicPolicyGradient.compute_action, 'CA')
VirtualControllerState.push = logging_hook(VirtualControllerState.push, 'VW')
VirtualControllerState.pop = logging_hook(VirtualControllerState.pop, 'VR')
DeterministicPolicyGradient.update_actor = logging_hook(DeterministicPolicyGradient.update_actor, 'Updating Actor')
DeterministicPolicyGradient.update_critic = logging_hook(DeterministicPolicyGradient.update_critic, 'Updating Critic')

def faux_game_thread():
    GameState.push = logging_hook(GameState.push, "GW")
    game_state = GameState()
    def game_idle():
        while game_state.get_flag(FLAGS.GAME_STATE_WRITTEN) and game_state.get_flag(FLAGS.IS_TRAINING):
            time.sleep(1e-3)
    game_idle = logging_hook(game_idle, "GI")
    pause_msg = False
    while True:
        if game_state.get_flag(FLAGS.IS_TRAINING):
            if pause_msg:
                file.write('Resumed training\n'.encode())
                pause_msg = False
            time.sleep(random.random() / 100)
            state = [random.random()]*3, [random.random()]*3, [0]
            state = game_state.MSG_TYPE().from_iterable(state)
            game_state.push(state)
            game_idle()
        else:
            if not pause_msg:
                with io.FileIO(LOG_FILE, 'a') as file:
                    file.write('Paused training\n'.encode())
                    file.flush()
                pause_msg = True

def physical_controller_thread():
    controller = ControllerHandler()
    controller.physical_controller_update_thread(exit=False)

def virtual_controller_thread():
    controller = ControllerHandler()
    controller.virtual_controller_update_thread()
    
def train_thread():
    actor = DriverActorModel().jit().to(device=config.device)
    critic = DriverCriticModel().jit().to(device=config.device)
    ddpg = DeterministicPolicyGradient(actor, critic)
    ddpg.train(episodes=1)


if __name__ == '__main__':

    with io.FileIO(LOG_FILE, 'w') as file:
        file.write(b'')
        file.flush()    

    train = threading.Thread(target=train_thread, daemon=True)
    train.start()

    threads = [multiprocessing.Process(target=f, daemon=True) for f in [physical_controller_thread, virtual_controller_thread, faux_game_thread]]
    for thread in threads:
        thread.start()

    train.join() 
    print('we made it')
    exit()



# from ipc import VirtualControllerState, PhysicalControllerState, FixedSizeState, debug_flags
# from controller import ControllerHandler
# import msgs_pb2

# bits = b'\r\x00\x00\x80?'

# msgs_pb2.GameState.init = FixedSizeState.create_init(msgs_pb2.GameState)
# msgs_pb2.GameState.to_tuple = FixedSizeState.to_tuple
# msgs_pb2.GameState.from_iterable = FixedSizeState.from_iterable

# lst = msgs_pb2.GameState.init().to_tuple()
# state = msgs_pb2.GameState.init().from_iterable(lst)
# print(state)
# print(lst)
# exit()
# bits = msgs_pb2.ControllerState().init().SerializeToString()
# print(bits)
# print( msgs_pb2.ControllerState.FromString(bits) )
# exit()

# from ipc import FLAGS
# with io.FileIO('test.log', 'a') as logfile:
#     VirtualControllerState.push = logging_hook(VirtualControllerState.push, logfile, 'Virtual Write')
#     PhysicalControllerState.push = logging_hook(PhysicalControllerState.push, logfile, 'Physical Write')
#     VirtualControllerState.pop = logging_hook(VirtualControllerState.pop, logfile, 'Virtual Read')
#     PhysicalControllerState.pop = logging_hook(PhysicalControllerState.pop, logfile, 'Physical Read')
#     # controller = ControllerHandler()
#     debug_flags()
#     PhysicalControllerState.set_flag(FLAGS.REQUEST_ACTION, True)
#     PhysicalControllerState.set_flag(FLAGS.REQUEST_INPUT, True)

#     debug_flags()

#     # exit()
#     ControllerHandler().physical_controller_update_thread()
#     ControllerHandler().virtual_controller_update_thread()
