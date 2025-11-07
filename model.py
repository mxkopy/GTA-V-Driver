import time
import torch
import torch.nn as nn
import math
from qpth.qp import QPFunction, QPSolvers
from torch.autograd import Variable
import torchvision.models as models

T = torch.float32
DEVICE = 'cuda'

class FourierLossWindow:
    def primes(num):
        n = 13
        primes = [2, 3, 5, 7, 11]
        while len(primes) < num:
            i = 0
            while primes[i] <= math.sqrt(n) and n % primes[i] != 0:
                i += 1
            if n % primes[i] != 0:
                primes.append(n)
            n += 1
        return primes[0:num]
    
    def __init__(self, model, n=128):
        self.step = 0
        self.model = model
        self.model_grads_vec = torch.nn.utils.parameters_to_vector(self.model.parameters()).to(device=DEVICE, dtype=T)
        self.distances = torch.tensor([2**i for i in range(n)], requires_grad=False).to(device=DEVICE, dtype=torch.int)
        # self.distances = torch.tensor(FourierLossWindow.primes(n), requires_grad=False).to(device=DEVICE, dtype=torch.int)
        self.grads = torch.zeros(n, torch.nn.utils.parameters_to_vector(model.parameters()).numel()).to(device=DEVICE, dtype=T)

    def backward(self):
        # Assign parameter grads for vectorized use
        idx = 0
        for parameter in self.model.parameters():
            numel = parameter.numel()
            self.model_grads_vec[idx:idx+numel] = parameter.grad.view(-1)
            idx += numel
        # Update grad windows, reassign grads & such
        self.step = (self.step+1) % self.distances[-1]
        eq_zero, neq_zero = (self.step % self.distances) == 0, (self.step % self.distances) != 0
        self.grads += self.model_grads_vec.unsqueeze(0) / self.distances.unsqueeze(1)
        model_grads = [parameter.grad for parameter in self.model.parameters()]
        torch.nn.utils.vector_to_parameters(self.grads.sum(dim=0), model_grads)
        self.grads *= neq_zero.unsqueeze(1)

class QPLayer(nn.Module):

    # TODO: add negative? Gz <= h && -Gz <= h
    def __init__(self, feature_size=4):
        super().__init__()
        self.Q = torch.diag(torch.ones(feature_size*2, dtype=T)).to(DEVICE)
        self.qpf = QPFunction(verbose=False, check_Q_spd=False, solver=QPSolvers.CVXPY, maxIter=20)
        self.e = Variable(torch.Tensor())
        # self.E = torch.diag(torch.cat((torch.ones(feature_size), -torch.ones(feature_size))))

    # A should be of size (hidden_size, feature_size)
    # b should be of size (hidden_size,)
    # layer solves Ax <= B (Gz <= h in the paper)
    def forward(self, A, x, b):
        A = torch.cat((A, A), dim=1)
        x = torch.cat((x, -x))
        y = self.qpf(self.Q, -x, A, b, self.e, self.e).squeeze()
        return (y[0:4] + y[4:]) / 2


class DriverModelBase(nn.Module):

    def __init__(self, controller_input_size=4):
        super().__init__()
        self.controller_input_size = controller_input_size
        self.rescale = nn.Conv2d(3, 3, (1, 2)).to(device=DEVICE, dtype=T)
        self.visual = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).to(device=DEVICE, dtype=T)
        self.visual.train()
        self.rotation_matrix = nn.Linear(3, controller_input_size * controller_input_size).to(device=DEVICE, dtype=T)
        self.pre_collate = nn.Sequential(
            nn.Linear(1000 + 3 + self.controller_input_size, 1000 + 3 + self.controller_input_size),
            nn.ELU(),
        ).to(device=DEVICE, dtype=T)
        self.collate = nn.Linear(1000 + 3 + self.controller_input_size, self.controller_input_size).to(device=DEVICE, dtype=T)
 
    def forward(self, IMG, INP, CAM, VEL):
        IMG_FEATURES = self.visual(self.rescale(IMG))
        CONTROLLER_INPUT_ROTATION_MATRIX = self.rotation_matrix(CAM).reshape(-1, self.controller_input_size, self.controller_input_size)
        ROTATED_CONTROLLER_INPUTS = torch.bmm(CONTROLLER_INPUT_ROTATION_MATRIX, INP.unsqueeze(-1)).squeeze(-1)
        X = torch.cat((IMG_FEATURES, VEL, ROTATED_CONTROLLER_INPUTS), dim=-1)
        Y = self.pre_collate(X)
        Y = self.collate(Y) @ ROTATED_CONTROLLER_INPUTS.t()
        return IMG_FEATURES, ROTATED_CONTROLLER_INPUTS, X, Y

# State -> Action
class DriverActorModel(DriverModelBase):

    def __init__(self, **kwargs):
        import copy
        super().__init__(**kwargs)
        self.pre_collate_P = copy.deepcopy(self.pre_collate).to(device=DEVICE, dtype=T)
        self.collate_P = self.collate.__class__(self.collate.in_features, 1).to(device=DEVICE, dtype=T)

    def forward(self, IMG, INP, CAM, VEL):
        IMG_FEATURES, ROTATED_CONTROLLER_INPUTS, X, COLLISION_AVOIDANCE = super().forward(IMG, INP, CAM, VEL)
        COLLISION_PROBABILITY = self.pre_collate_P(X)
        COLLISION_PROBABILITY = self.collate_P(COLLISION_PROBABILITY)
        COLLISION_PROBABILITY = nn.functional.sigmoid(COLLISION_PROBABILITY)
        COLLISION_AVOIDANCE_JOYSTICK = nn.functional.tanh(COLLISION_AVOIDANCE[:, :2])
        COLLISION_AVOIDANCE_TRIGGERS = nn.functional.sigmoid(COLLISION_AVOIDANCE[:, 2:])
        COLLISION_AVOIDANCE = torch.cat((COLLISION_AVOIDANCE_JOYSTICK, COLLISION_AVOIDANCE_TRIGGERS), dim=-1)
        return (INP * (1 - COLLISION_PROBABILITY)) + (COLLISION_PROBABILITY * COLLISION_AVOIDANCE)

# State, Action -> QValue
# 
# This model should predict how useful the collision avoidance guidance is. 
# Realistically, it can just predict how (un)likely a crash is going to happen given state & user input, 
# where 'user input' is the collision guidance from the actor model.
# 
# TODO: there might be a way to finagle this functionality just from P in the actor model. 
class DriverCriticModel(DriverModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.post_collate = nn.Linear(self.collate.out_features, 1)

    def forward(self, IMG, _, CAM, VEL, ACT):
        _, _, _, Y = super().forward(IMG, ACT, CAM, VEL)
        Y = nn.functional.elu(Y)
        Q = self.post_collate(Y)
        return Q
