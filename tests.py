import io
from collections.abc import Callable, Buffer

def logging_hook_function(method: Callable, logfile: io.BufferedIOBase, tag: str | bytes) -> Callable:
    counter = [0]
    def log(*args, **kwargs):
        logfile.write(f'{tag} {counter[0]} BEGIN')
        logfile.flush()
        method(*args, **kwargs)
        logfile.write(f'{tag} {counter[0]} END')
        logfile.flush()
        counter[0] += 1
    return log

def faux_game_thread(logfile):
    import time
    from ipc import FLAGS, GameState
    GameState.push = logging_hook_function(GameState.push, logfile, "Game Write")
    game_state = GameState()
    def game_idle():
        while game_state.get_flag(FLAGS.GAME_STATE_WRITTEN) and game_state.get_flag(FLAGS.IS_TRAINING):
            time.sleep(1e-3)
    game_idle = logging_hook_function(game_idle, logfile, "Game Idle")
    done = False
    while True:
        if game_state.get_flag(FLAGS.IS_TRAINING):
            done = False
            state = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0,]
            game_state.push(state)
            game_idle()
        else:
            if not done:
                logfile.write('Not training')
                logfile.flush()
                done = True


class IPCTests:

    def thread_execution_order_test():
        import threading
        import config
        import time
        import io
        from environment import Environment
        from ipc import GameState, PhysicalControllerState, VirtualControllerState
        from controller import ControllerHandler
        from model import DeterministicPolicyGradient, DriverActorModel, DriverCriticModel

        import sys
        if sys._is_gil_enabled():
            print("Warning: running controller loop with GIL")

        with io.FileIO('thread_execution_order_test.log', 'a') as logfile:

            GameState.pop = logging_hook_function(GameState.pop, logfile, 'Game Read')
            DeterministicPolicyGradient.compute_action = logging_hook_function(DeterministicPolicyGradient.compute_action, logfile, 'Computing Action')
            VirtualControllerState.push = logging_hook_function(VirtualControllerState.push, logfile, 'Virtual Write')
            VirtualControllerState.pop = logging_hook_function(VirtualControllerState.pop, logfile, 'Virtual Read')
            DeterministicPolicyGradient.update_actor = logging_hook_function(DeterministicPolicyGradient.update_actor, logfile, 'Updating Actor')
            DeterministicPolicyGradient.update_critic = logging_hook_function(DeterministicPolicyGradient.update_critic, logfile, 'Updating Critic')

            # ddpg.environment.game_state.pop = logging_hook_function(ddpg.environment.game_state.pop, logfile, 'Game Read')
            # ddpg.get_action = logging_hook_function(ddpg.get_action, logfile, 'Computing Action')
            # ddpg.environment.virtual_controller_state.push = logging_hook_function(controller.virtual_controller_state.push, logfile, 'Virtual Write')
            # controller.virtual_controller_state.pop = logging_hook_function(controller.virtual_controller_state.pop, logfile, 'Virtual Read')
            # ddpg.update_actor = logging_hook_function(ddpg.update_actor, logfile, 'Updating Actor')
            # ddpg.update_critic = logging_hook_function(ddpg.update_critic, logfile, 'Updating Critic')
            # controller.physical_controller_state.push = logging_hook_function(controller.physical_controller_state.push, logfile, 'Physical Write')
            # controller.physical_controller_state.pop = logging_hook_function(controller.physical_controller_state.pop, logfile, 'Physical Read')

            actor = DriverActorModel().to(device=config.device).jit()
            critic = DriverCriticModel().to(device=config.device).jit()

            controller = ControllerHandler()
            ddpg = DeterministicPolicyGradient(actor, critic)

            physical_controller_update_thread = threading.Thread(target=ControllerHandler.physical_controller_update_thread, args=(controller,), kwargs={'exit': False})
            virtual_controller_update_thread = threading.Thread(target=ControllerHandler.virtual_controller_update_thread, args=(controller,))
            ddpg_thread = threading.Thread(target=DeterministicPolicyGradient.run_episode, args=(ddpg,))
            game_thread = threading.Thread(target=faux_game_thread)
            
            for thread in [physical_controller_update_thread, virtual_controller_update_thread, ddpg_thread, game_thread]:
                thread.start()

            time.sleep(5)
            exit()

