import io
from collections.abc import Callable, Buffer

def logging_hook(method: Callable, logfile: io.BufferedIOBase, tag: str | bytes) -> Callable:
    counter = [0]
    def log(*args, **kwargs):
        logfile.write(f'{tag} {counter[0]} BEGIN\n'.encode())
        logfile.flush()
        output = method(*args, **kwargs)
        logfile.write(f'{tag} {counter[0]} END\n'.encode())
        logfile.flush()
        counter[0] += 1
        return output
    return log

# def exitflag_hook(method: Callable, exitflag_reference: list[bool]):

#     def with_exitflag(*args, **kwargs):


def faux_game_thread(logfile):
    import time
    from ipc import FLAGS, GameState
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
            game_state.push(state)
            game_idle()
        else:
            if not done:
                logfile.write('Not training\n'.encode())
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
            print("Warning: running threading test loop with GIL")

        with io.FileIO('thread_execution_order_test.log', 'w') as logfile:

            GameState.pop = logging_hook(GameState.pop, logfile, 'Game Read')
            DeterministicPolicyGradient.compute_action = logging_hook(DeterministicPolicyGradient.compute_action, logfile, 'Computing Action')
            VirtualControllerState.push = logging_hook(VirtualControllerState.push, logfile, 'Virtual Write')
            VirtualControllerState.pop = logging_hook(VirtualControllerState.pop, logfile, 'Virtual Read')
            DeterministicPolicyGradient.update_actor = logging_hook(DeterministicPolicyGradient.update_actor, logfile, 'Updating Actor')
            DeterministicPolicyGradient.update_critic = logging_hook(DeterministicPolicyGradient.update_critic, logfile, 'Updating Critic')

            # ddpg.environment.game_state.pop = logging_hook(ddpg.environment.game_state.pop, logfile, 'Game Read')
            # ddpg.get_action = logging_hook(ddpg.get_action, logfile, 'Computing Action')
            # ddpg.environment.virtual_controller_state.push = logging_hook(controller.virtual_controller_state.push, logfile, 'Virtual Write')
            # controller.virtual_controller_state.pop = logging_hook(controller.virtual_controller_state.pop, logfile, 'Virtual Read')
            # ddpg.update_actor = logging_hook(ddpg.update_actor, logfile, 'Updating Actor')
            # ddpg.update_critic = logging_hook(ddpg.update_critic, logfile, 'Updating Critic')
            # controller.physical_controller_state.push = logging_hook(controller.physical_controller_state.push, logfile, 'Physical Write')
            # controller.physical_controller_state.pop = logging_hook(controller.physical_controller_state.pop, logfile, 'Physical Read')

            actor = DriverActorModel().jit().to(device=config.device)
            critic = DriverCriticModel().jit().to(device=config.device)

            controller = ControllerHandler()
            ddpg = DeterministicPolicyGradient(actor, critic)

            physical_controller_update_thread = threading.Thread(target=ControllerHandler.physical_controller_update_thread, args=(controller,), kwargs={'exit': False})
            virtual_controller_update_thread = threading.Thread(target=ControllerHandler.virtual_controller_update_thread, args=(controller,))
            ddpg_thread = threading.Thread(target=DeterministicPolicyGradient.run_episode, args=(ddpg,))
            game_thread = threading.Thread(target=faux_game_thread, args=(logfile,))
            
            for thread in [physical_controller_update_thread, virtual_controller_update_thread, ddpg_thread, game_thread]:
                thread.start()
            print('we made it')

            time.sleep(5)
            exit()

IPCTests.thread_execution_order_test()


from ipc import VirtualControllerState, PhysicalControllerState, FixedSizeState, debug_flags
from controller import ControllerHandler
import msgs_pb2

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

from ipc import FLAGS
with io.FileIO('test.log', 'a') as logfile:
    VirtualControllerState.push = logging_hook(VirtualControllerState.push, logfile, 'Virtual Write')
    PhysicalControllerState.push = logging_hook(PhysicalControllerState.push, logfile, 'Physical Write')
    VirtualControllerState.pop = logging_hook(VirtualControllerState.pop, logfile, 'Virtual Read')
    PhysicalControllerState.pop = logging_hook(PhysicalControllerState.pop, logfile, 'Physical Read')
    # controller = ControllerHandler()
    debug_flags()
    PhysicalControllerState.set_flag(FLAGS.REQUEST_ACTION, True)
    PhysicalControllerState.set_flag(FLAGS.REQUEST_INPUT, True)

    debug_flags()

    # exit()
    ControllerHandler().physical_controller_update_thread()
    ControllerHandler().virtual_controller_update_thread()
