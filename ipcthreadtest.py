import threading
import io
import multiprocessing
import config
import time
import random
import sys
from ddpg import DeterministicPolicyGradient
from collections.abc import Callable, Buffer
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

GameState.pop = logging_hook(GameState.pop, 'Game Read')

VirtualControllerState.push = logging_hook(VirtualControllerState.push, 'Virtual Write')
VirtualControllerState.pop = logging_hook(VirtualControllerState.pop, 'Virtual Read')


def faux_game_thread():
    GameState.push = logging_hook(GameState.push, "Game Write")
    game_state = GameState()
    def game_idle():
        while game_state.get_flag(FLAGS.GAME_STATE_WRITTEN) and game_state.get_flag(FLAGS.IS_TRAINING):
            time.sleep(1e-3)
    game_idle = logging_hook(game_idle, "Game Idle")
    pause_msg = False
    state = GameState.MSG_TYPE.init()
    while True:
        if game_state.get_flag(FLAGS.IS_TRAINING):
            if pause_msg:
                file.write('Resumed training\n'.encode())
                pause_msg = False
            time.sleep(random.random() / 100)
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
    


if __name__ == '__main__':

    # with io.FileIO(LOG_FILE, 'w') as file:
    #     file.write(b'')
    #     file.flush()    

    if len(sys.argv) == 1:
        from ddpg import DeterministicPolicyGradient
        from model import DriverActorModel, DriverCriticModel
        DeterministicPolicyGradient.compute_action = logging_hook(DeterministicPolicyGradient.compute_action, 'Compute Action')
        DeterministicPolicyGradient.update_actor = logging_hook(DeterministicPolicyGradient.update_actor, 'Updating Actor')
        DeterministicPolicyGradient.update_critic = logging_hook(DeterministicPolicyGradient.update_critic, 'Updating Critic')
        actor = DriverActorModel().to(device=config.device).jit()
        critic = DriverCriticModel().to(device=config.device).jit()
        ddpg = DeterministicPolicyGradient(actor, critic)
        ddpg.environment.game_state.set_flag(FLAGS.REQUEST_GAME_STATE, True)
        print("Starting training")
        ddpg.train()

    if sys.argv[1] == 'game':
        faux_game_thread()

    if sys.argv[1] == 'debug':
        from ipc import debug_flags, Flags
        if len(sys.argv) == 2:
            debug_flags()
        elif len(sys.argv) == 3 or sys.argv[3] == '1':
            Flags().set_flag(int(sys.argv[2]), True)
        elif sys.argv[3] == '0':
            Flags().set_flag(int(sys.argv[2]), False)
        exit()
