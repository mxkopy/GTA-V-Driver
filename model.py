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
    
    def __init__(self, n=192):
        self.buckets = {
            p: torch.tensor(0, requires_grad=True, dtype=T).to(DEVICE) for p in FourierLossWindow.primes(n)
        }
    
    # I think this works
    # The only gradients zeroed out after a backwards pass are those accumulated here (?)
    def add_loss(self, loss, step):
        sparse_loss = torch.tensor(0, requires_grad=True, dtype=T).to(DEVICE)
        for p in self.buckets:
            if step % p == 0:
                sparse_loss += self.buckets[p]
                self.buckets[p] *= 0
            else:
                self.buckets[p] += loss / p
        return sparse_loss


class QPLayer(nn.Module):

    # TODO: add negative? Gz <= h && -Gz <= h
    def __init__(self, feature_size=4):
        super().__init__()
        self.Q = torch.diag(torch.ones(feature_size, dtype=T)).to(DEVICE)
        self.qpf = QPFunction(verbose=False, check_Q_spd=False, solver=QPSolvers.CVXPY, maxIter=200)

    # A should be of size (hidden_size, feature_size)
    # b should be of size (hidden_size,)
    # layer solves Ax <= B (Gz <= h in the paper)
    def forward(self, A, x, b):
        return self.qpf(self.Q, -x, A, b, Variable(torch.Tensor()), Variable(torch.Tensor()))

    
class DriverModel(nn.Module):

    def __init__(self, controller_input_size=4, hidden_size=8):
        super().__init__()
        self.controller_input_size = controller_input_size
        self.hidden_size = hidden_size
        self.rescale = nn.Conv2d(3, 3, (1, 2), dtype=T).to(DEVICE)
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).to(DEVICE)
        self.vgg.train()
        self.vgg.classifier[-1] = nn.Linear(4096, hidden_size, dtype=T).to(DEVICE)
        self.rotation_matrix = nn.Linear(3+3, hidden_size * hidden_size, dtype=T).to(DEVICE)
        
        self.learned_A = nn.Linear(hidden_size, hidden_size*controller_input_size, dtype=T).to(DEVICE)
        self.learned_b = nn.Linear(hidden_size, hidden_size, dtype=T).to(DEVICE)

        self.qp = QPLayer(controller_input_size).to(DEVICE)
 
    # Should have a sort of producer & consumer paradigm
    # cvx needs to continuously produce controller input, while vgg can be updated once in awhile
    def forward(self, image, camera_direction, relative_velocity, controller_input):

        # Can take its time
        features = self.vgg(self.rescale(image))
        rotation_matrix = self.rotation_matrix(torch.cat((camera_direction, relative_velocity)))
        rotated = rotation_matrix.reshape(self.hidden_size, self.hidden_size) @ features.t()
        rotated = nn.functional.relu(rotated)

        # Produce like crazy
        A = self.learned_A(rotated.t()).reshape(self.hidden_size, self.controller_input_size)
        A = torch.abs(A)
        b = self.learned_b(rotated.t())
        b = torch.abs(b)

        # a = time.time()
        (x,) = self.qp(A, controller_input, b)
        # b = time.time()
        # print(f'inner {b - a}')

        return x

model = DriverModel()

opt = torch.optim.Adam(model.parameters(), 1e-4)

while True:

    image, camera_direction, relative_velocity, controller_input = torch.rand(1, 3, 512, 512).to(DEVICE), torch.rand(3).to(DEVICE), torch.rand(3).to(DEVICE), torch.rand(4).to(DEVICE)

    a = time.time()
    out = model(image, camera_direction, relative_velocity, controller_input)
    loss = torch.sum(torch.square(out-controller_input))
    opt.zero_grad()
    loss.backward()
    opt.step()
    b = time.time()
    print(b - a)

