import io
import multiprocessing
import config
import time
import io
from collections.abc import Callable, Buffer
from model import DeterministicPolicyGradient, DriverActorModel, DriverCriticModel
from ipc import GameState, PhysicalControllerState, VirtualControllerState, FLAGS
from controller import ControllerHandler

def logging_hook(method: Callable, tag: str | bytes, logfile: str = 'thread_execution_order_test.log') -> Callable:
    counter = [0]
    def log(*args, **kwargs):
        with io.FileIO(logfile, 'a') as file:
            file.write(f'{tag} {counter[0]} BEGIN\n'.encode())
            file.flush()
            output = method(*args, **kwargs)
            file.write(f'{tag} {counter[0]} END\n'.encode())
            file.flush()
            counter[0] += 1
            return output
    return log

def faux_game_thread(logfile):
    GameState.push = logging_hook(GameState.push, logfile, "Game Write")
    game_state = GameState()
    def game_idle():
        while game_state.get_flag(FLAGS.GAME_STATE_WRITTEN) and game_state.get_flag(FLAGS.IS_TRAINING):
            time.sleep(1e-3)
    game_idle = logging_hook(game_idle, logfile, "Game Idle")
    done = False
    while True:
        if game_state.get_flag(FLAGS.IS_TRAINING):
            done = False
            state = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0,]
            state = game_state.MSG_TYPE().from_iterable(state)
            game_state.push(state)
            game_idle()
        else:
            if not done:
                logfile.write('Not training\n'.encode())
                logfile.flush()
                done = True

GameState.pop = logging_hook(GameState.pop, 'Game Read')
DeterministicPolicyGradient.compute_action = logging_hook(DeterministicPolicyGradient.compute_action, 'Computing Action')
VirtualControllerState.push = logging_hook(VirtualControllerState.push, 'Virtual Write')
VirtualControllerState.pop = logging_hook(VirtualControllerState.pop, 'Virtual Read')
DeterministicPolicyGradient.update_actor = logging_hook(DeterministicPolicyGradient.update_actor, 'Updating Actor')
DeterministicPolicyGradient.update_critic = logging_hook(DeterministicPolicyGradient.update_critic, 'Updating Critic')


def physical_controller_thread():
    controller = ControllerHandler()
    controller.physical_controller_update_thread(exit=False)

def virtual_controller_thread():
    controller = ControllerHandler()
    controller.virtual_controller_update_thread()
    
def ddpg_thread():
    actor = DriverActorModel().jit().to(device=config.device)
    critic = DriverCriticModel().jit().to(device=config.device)
    ddpg = DeterministicPolicyGradient(actor, critic)
    ddpg.train()

def game_thread():
    with io.FileIO('thread_execution_order_test.log', 'a') as file:
        faux_game_thread(file)


if __name__ == '__main__':

    import sys
    if sys._is_gil_enabled():
        print("Warning: running threading test loop with GIL")

    with io.FileIO('thread_execution_order_test.log', 'w') as file:
        file.write(b'')
        file.flush()    

    threads = [multiprocessing.Process(target=f, daemon=True) for f in [physical_controller_thread, virtual_controller_thread, game_thread]]
    for thread in threads:
        thread.start()

    import threading
    threading.Thread(target=ddpg_thread, daemon=True).start()

    time.sleep(10)
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
