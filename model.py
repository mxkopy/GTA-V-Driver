import time
import torch
import torch.nn as nn
import math
from qpth.qp import QPFunction, QPSolvers
from torch.autograd import Variable
import torchvision.models as models

T = torch.float32
DEVICE = 'cuda'


class GradList:

    def __init__(self, grads):
        self.grads = grads
    
    def __iadd__(self, other):
        for i in range(len(self.grads)):
            self.grads[i] += other[i]
        return self

    def __add__(self, other):
        return self.__iadd__(other)

    def from_model_params(model):
        return GradList([torch.zeros_like(parameter, requires_grad=False) for parameter in model.parameters()])

    def to_model_grads(self, model):
        for i, parameter in enumerate(model.parameters()):
            parameter.grad = self.grads[i] if parameter.grad is None else parameter.grad + self.grads[i]

    def zero(self):
        for grad in self.grads:
            grad.zero_()

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
    
    def __init__(self, model, n=1):
        self.model = model
        self.model_parameters = list(self.model.parameters())
        self.distances = torch.tensor([2**i for i in range(n)], requires_grad=False).to(DEVICE, dtype=torch.int)
        self.grads = [torch.expand_copy(parameter.unsqueeze(0), (n, *parameter.shape)).detach() for parameter in self.model.parameters()]

    def add_loss(self, loss, step):
        for p in range(len(self.grads)):
            self.grads[p] *= (step % self.distances != 0).view(-1, *(1 for _ in self.grads[p].shape[1:])).expand(*self.grads[p].shape)
            self.grads[p] += torch.stack(torch.autograd.grad([loss / d.float() for d in self.distances], self.model_parameters[p], retain_graph=True, materialize_grads=True))
            if self.model_parameters[p].grad is None:
                self.model_parameters[p].grad = torch.sum(self.grads[p] * (step % self.distances == 0).view(-1, *(1 for _ in self.grads[p].shape[1:])).expand(*self.grads[p].shape), dim=0)
            else:
                self.model_parameters[p].grad += torch.sum(self.grads[p] * (step % self.distances == 0).view(-1, *(1 for _ in self.grads[p].shape[1:])).expand(*self.grads[p].shape), dim=0)

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


class DriverModel(nn.Module):

    def __init__(self, controller_input_size=4, hidden_size=8):
        super().__init__()
        self.controller_input_size = controller_input_size
        self.hidden_size = hidden_size
        self.rescale = nn.Conv2d(3, 3, (1, 2), dtype=T).to(DEVICE)
        self.visual = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).to(DEVICE, dtype=T)
        self.visual.train()
        self.visual_adapter = nn.Linear(1000, hidden_size, dtype=T).to(DEVICE)
        self.rotation_matrix = nn.Linear(3+3, hidden_size * hidden_size, dtype=T).to(DEVICE)
        
        self.learned_A = nn.Linear(hidden_size, hidden_size*controller_input_size, dtype=T).to(DEVICE)
        self.learned_b = nn.Linear(hidden_size, hidden_size, dtype=T).to(DEVICE)

        self.qp = QPLayer(controller_input_size).to(DEVICE)
 
    # Should have a sort of producer & consumer paradigm
    # cvx needs to continuously produce controller input, while vgg can be updated once in awhile
    def forward(self, image, camera_direction, relative_velocity, controller_input):

        # Can take its time
        features = self.visual_adapter(self.visual(self.rescale(image)))
        rotation_matrix = self.rotation_matrix(torch.cat((camera_direction, relative_velocity)))
        rotated = rotation_matrix.reshape(self.hidden_size, self.hidden_size) @ features.t()
        rotated = nn.functional.relu(rotated)

        # Produce like crazy
        A = self.learned_A(rotated.t()).reshape(self.hidden_size, self.controller_input_size)
        A = torch.abs(A)
        b = self.learned_b(rotated.t())

        x = self.qp(A, controller_input, b)

        xy, triggers = torch.clamp(x[0:2], min=-1, max=1), torch.clamp(x[2:4], min=0, max=1)

        return torch.cat((xy, triggers))

# model = DriverModel()
# opt = torch.optim.Adam(model.parameters(), 1e-4)
# losswindow = FourierLossWindow(model)
# # step = 1
# while True:

#     image, camera_direction, relative_velocity, controller_input = torch.ones(1, 3, 512, 512, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(4, dtype=T).to(DEVICE)

#     a = time.time()
#     out = model(image, camera_direction, relative_velocity, controller_input)
#     mse_loss = torch.nn.functional.mse_loss(out, controller_input)
#     # losswindow.add_loss(mse_loss, step)
#     mse_loss.backward()
#     opt.step()
#     opt.zero_grad()
#     b = time.time()
#     print(f'{b - a}')
#     # print(mse_loss.item())
#     # step += 1



