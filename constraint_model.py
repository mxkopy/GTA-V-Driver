import torch
import torch.nn as nn
from torch.autograd import Variable
from model import T, DEVICE

# An interesting idea is using differentiable constraint programming to learn constraints on the user's inputs, depending on image features
from qpth.qp import QPFunction, QPSolvers
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
