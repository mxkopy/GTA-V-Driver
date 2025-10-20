# pip install numpy scipy --extra-index-url https://urob.github.io/numpy-mkl
import mkl
import torch
import torch.nn as nn
import math
import cvxpy as cvx
from cvxpylayers.torch import CvxpyLayer


# import numpy
# numpy.show_config()
# exit()

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

    
class DriverModel(nn.Module):

    def __init__(self):
        import torchvision.models as models
        super().__init__()
        self.rescale = nn.Conv2d(3, 3, (1, 2), dtype=T).to(DEVICE)
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).to(DEVICE)
        self.vgg.train()
        self.vgg.classifier[-1] = nn.Linear(4096, 8, dtype=T).to(DEVICE)
        self.rotation_matrix = nn.Linear(3+3, 8*8, dtype=T).to(DEVICE)
        
        self.learned_A = nn.Linear(8, 8*4, dtype=T).to(DEVICE)
        self.learned_b = nn.Linear(8, 8, dtype=T).to(DEVICE)

        self.input = cvx.Parameter(4)

        self.A = cvx.Parameter((8, 4))
        self.b = cvx.Parameter(8)
        self.x = cvx.Variable(4)

        self.objective = cvx.Minimize(cvx.sum(cvx.abs(self.x - self.input)))
        self.constraint = [self.A @ self.x <= self.b]
        self.problem = cvx.Problem(self.objective, self.constraint)
        self.cvx = CvxpyLayer(self.problem, parameters=[self.input, self.A, self.b], variables=[self.x]).to(DEVICE)
        
    # Needs a sort of producer & consumer paradigm
    # cvx needs to continuously produce controller input, while vgg can be updated once in awhile
    def forward(self, image, camera_direction, relative_velocity, controller_input):

        # Can take its time
        features = self.vgg(self.rescale(image))
        rotation_matrix = self.rotation_matrix(torch.cat((camera_direction, relative_velocity)))
        rotated = rotation_matrix.reshape(8, 8) @ features.t()
        rotated = nn.functional.relu(rotated)
        # Produce like crazy
        A = self.learned_A(rotated.t()).reshape(8, 4)
        b = self.learned_b(rotated.t())

        a = time.time()
        (y,) = self.cvx(controller_input, A, b, solver_args={'solve_method': 'Clarabel'})
        b = time.time()
        print(f'inner {b - a}')

        return y

import time
model = DriverModel()
opt = torch.optim.Adam(model.parameters(), 1e-1)

x, y, z, w = torch.rand(1, 3, 512, 512, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(4, dtype=T).to(DEVICE)

a = time.time()
output = model(x, y, z, w)
opt.zero_grad()
output.sum().backward()
opt.step()
b = time.time()
print(b - a)

x, y, z, w = torch.rand(1, 3, 512, 512, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(4, dtype=T).to(DEVICE)

a = time.time()
output = model(x, y, z, w)
opt.zero_grad()
output.sum().backward()
opt.step()
b = time.time()
print(b - a)

x, y, z, w = torch.rand(1, 3, 512, 512, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(4, dtype=T).to(DEVICE)

a = time.time()
output = model(x, y, z, w)
opt.zero_grad()
output.sum().backward()
opt.step()
b = time.time()
print(b - a)

x, y, z, w = torch.rand(1, 3, 512, 512, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(3, dtype=T).to(DEVICE), torch.rand(4, dtype=T).to(DEVICE)

a = time.time()
output = model(x, y, z, w)
opt.zero_grad()
output.sum().backward()
opt.step()
b = time.time()
print(b - a)