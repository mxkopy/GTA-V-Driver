import torch
from collections import namedtuple
from model import DEVICE, T
import random

class Transition(namedtuple('_Transition', ('S', 'A', 'NS', 'R', 'F'))):

    def __iadd__(self, other):
        return Transition(
            tuple(torch.cat(self.S[i], other.S[i]) for i in range(len(self.S))),
            torch.cat(self.A, other.A),
            tuple(torch.cat(self.NS[i], other.NS[i]) for i in range(len(self.NS))),
            torch.cat(self.R, other.R), 
            torch.cat(self.F, other.F)
        )

    def __add__(self, other):
        if other != 0 and other != None:
            self += other
        return self

class ReplayBuffer:

    def __init__(self, n=300):
        self.n = n
        self.reserve = [None for _ in range(n)]
        self.buffer = [None for _ in range(n)]
        self.idx = 0

    def __iadd__(self, transition: Transition):
        self.buffer[self.idx%self.n] = transition
        self.idx = (self.idx+1)%self.n
        return self

    def __getitem__(self, idx):
        return self.buffer[idx%self.n]           

    def add(self, transition: Transition):
        self += transition

    def sample(self, batch_size):
        batch = [x for x in self.buffer if x is not None] + [x for x in self.reserve if x is not None]
        observations = random.sample(batch, min(len(batch), batch_size))
        return Transition(
            tuple(torch.cat([observation.S[i] for observation in observations]).to(device=DEVICE, dtype=T) for i in range(len(observations[0].S))),
            torch.cat([observation.A for observation in observations]).to(device=DEVICE, dtype=T),
            tuple(torch.cat([observation.NS[i] for observation in observations]).to(device=DEVICE, dtype=T) for i in range(len(observations[0].NS))),
            torch.cat([observation.R for observation in observations]).to(device=DEVICE, dtype=T),
            torch.cat([observation.F for observation in observations]).to(device=DEVICE, dtype=T)
        )

    def reset(self):
        self.reserve = random.sample(self.reserve + self.buffer, self.n)
        self.buffer = [None for _ in range(self.n)]
        self.idx = 0

