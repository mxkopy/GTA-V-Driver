import threading
import io
import multiprocessing
import config
import time
import random
import sys
from collections.abc import Callable, Buffer
from model import DriverActorModel, DriverCriticModel

if __name__ == '__main__':

    assert len(sys.argv) >= 2

    if sys.argv[1] == 'controller':
        from controller import ControllerHandler
        ControllerHandler().start()

    if sys.argv[1] == 'train':
        from ddpg import DeterministicPolicyGradient
        from model import DriverActorModel, DriverCriticModel
        from ipc import Flags, FLAGS
        from controller import VirtualController
        actor = DriverActorModel().to(device=config.device)
        critic = DriverCriticModel().to(device=config.device)
        print("Waiting for script to load")
        Flags().wait_until(FLAGS.REQUEST_ACTION, True)
        print("Script loaded")
        ddpg = DeterministicPolicyGradient(actor, critic)
        ddpg.train()

    if sys.argv[1] == 'debug':
        from ipc import debug_flags, Flags
        if len(sys.argv) == 2:
            debug_flags()
        else: 
            flags = Flags()
            if sys.argv[3] == '0':
                flags.set_flag(int(sys.argv[2]), False)
            elif sys.argv[3] == '1':
                flags.set_flag(int(sys.argv[2]), True)
            
