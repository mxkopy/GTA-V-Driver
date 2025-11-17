import time
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from collections import OrderedDict
import torchvision.models as models
import torchvision
from environment import ReplayBuffer, State, Action, Reward, NextState


class DeterministicPolicyGradient:

    def __init__(
            self, 
            actor: nn.Module, 
            critic: nn.Module, 
            replay_buffer: ReplayBuffer = ReplayBuffer(), 
            action_min: torch.tensor | float | None = None,
            action_max: torch.tensor | float | None = None,
            noise_distribution = torch.distributions.Normal(0, 1)
        ):
        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.action_min = action_min
        self.action_max = action_max
        self.noise_distribution = noise_distribution

    def action(self, state: State, noise_scale: float = 1.0) -> Action:
        action: Action = self.actor(state)
        noise: torch.tensor = self.noise_distribution.sample(action.size()) * noise_scale
        noise: torch.tensor = action.new(noise)
        action = torch.clamp(action + noise, min=self.action_min, max=self.action_max)
        return action    




T = torch.float32
DEVICE = 'cuda'
IMG_RESOLUTION = (360, 640)

# TODO: This component of the actor/critic model should be trained separately & be otherwise static
# Otherwise, it might take too long to train the rest of the network 
# What it really should do is provide per-pixel kinematic information like distance & velocity relative to the viewer
class VisualModel(nn.Module):

    def __init__(self, img_resolution=(360, 640), channels=[3, 6, 9, 9, 9, 9, 9]):
        super().__init__()
        self.img_resolution = img_resolution
        self.channels = channels
        self.channels.insert(0, 3)
        self.sz = lambda i: max(img_resolution) // (2**i)
        self.model = nn.Sequential(
            torchvision.transforms.CenterCrop(self.sz(0)),
            *(nn.Sequential(
                nn.Conv2d(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=(3, 3)),
                nn.AdaptiveMaxPool2d(self.sz(i+1)),
                nn.LayerNorm((self.channels[i+1], self.sz(i+1), self.sz(i+1)), elementwise_affine=False)
            ) for i in range(len(self.channels)-1))
        )

    def forward(self, img):
        return self.model(img.to(device=DEVICE, dtype=T) / 255)

class DriverModelBase(nn.Module):

    def __init__(self, controller_input_size=4, img_resolution=(360, 640), visual_channels=[3, 6, 9, 9, 9, 9, 9]):
        super().__init__()
        self.input_shapes = OrderedDict({
            'IMG': (visual_channels[0], *img_resolution), 
            'INP': (controller_input_size,), 
            'CAM': (3,), 
            'VEL': (3,)
        })
        self.visual = VisualModel(img_resolution=img_resolution, channels=visual_channels).to(device=DEVICE, dtype=T)
        self.visual.train()
        self.visual_output_size = visual_channels[-1] * self.visual.sz(len(visual_channels)-1)**2
        self.controller_input_size = controller_input_size
        self.hidden_size = self.visual_output_size + 3 + self.controller_input_size
        # If rotation_matrix is any deeper than 1 layer, the model becomes incapable of input passthrough :0
        self.rotation_matrix = nn.Sequential(
            nn.Linear(3, self.visual_output_size),
            nn.LayerNorm(self.visual_output_size),
            nn.ELU(),
            nn.Linear(self.visual_output_size, self.visual_output_size**2)
        ).to(device=DEVICE, dtype=T)
        # This should probably be more complicated on the other hand
        self.collate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.controller_input_size)
        )
 
    def example_inputs(self, batch_size=1):
        return tuple(torch.rand(batch_size, *shape) for shape in self.input_shapes.values())

    def forward(self, IMG, INP, CAM, VEL):
        # Get relevant features from the screen frame
        IMG_FEATURES = self.visual(IMG).reshape(IMG.shape[0], self.visual_output_size, 1)
        # TODO: The camera angle determines some sort of rotation of the visual features relative to the front of the vehicle
        # which might be approximated here
        CAMERA_ROTATION_MATRIX = self.rotation_matrix(CAM).reshape(-1, self.visual_output_size, self.visual_output_size)
        ROTATED_IMG_FEATURES = torch.bmm(CAMERA_ROTATION_MATRIX, IMG_FEATURES).squeeze(-1)
        # Final action prediction depends on image features, car's velocity, and the controller inputs
        X = torch.cat((ROTATED_IMG_FEATURES.flatten(start_dim=1), VEL, INP), dim=-1)
        Y = self.collate(X)
        return IMG_FEATURES, ROTATED_IMG_FEATURES, CAMERA_ROTATION_MATRIX, X, Y

    def update_parameters(self, other):
        torch.nn.utils.vector_to_parameters(torch.nn.utils.parameters_to_vector(other.parameters()), self.parameters())
        return self
    
    def soft_update(self, policy, polyak):
        target_state_dict = self.state_dict()
        policy_state_dict = policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = (target_state_dict[key] * polyak) + (policy_state_dict[key] * (1 - polyak))
        self.load_state_dict(target_state_dict)

    def jit(self):
        return torch.export.export(
            self,
            args=tuple(x.to(device=DEVICE, dtype=T) for x in self.example_inputs(batch_size=4)),
            dynamic_shapes={arg: (torch.export.Dim.DYNAMIC,)+shape for arg, shape in self.input_shapes.items()}
        ).module()
        

# State -> Action
class DriverActorModel(DriverModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, IMG, INP, CAM, VEL):
        *_, AVOIDANCE = super().forward(IMG, INP, CAM, VEL)
        # AVOIDANCE = torch.cat((torch.clamp(AVOIDANCE[:, :2], min=-1, max=1), torch.clamp(torch.abs(AVOIDANCE[:, 2:]), min=0, max=1)), dim=1)
        return AVOIDANCE
        # return (AVOIDANCE - AVOIDANCE.mean()) / (AVOIDANCE.max() - AVOIDANCE.min())

    # Maybe to get input-modifying rather than overriding
    # we might want to train the initial network to pass the input through. 
    def train_for_input_passthrough(self, batchsize=16, lr=1e-4, cutoff=1e-2):
        opt = torch.optim.Adam(self.parameters(), lr)
        loss = cutoff
        for param in self.visual.parameters():
            param.requires_grad = False
        while loss >= cutoff:
            img = torch.rand(batchsize, 3, *IMG_RESOLUTION) * 255
            inp = torch.rand(batchsize, 4)
            inp[:, :2] *= 2
            inp[:, :2] -= 1
            cam = torch.rand(batchsize, 3)*2 -1
            vel = torch.rand(batchsize, 3)*2 -1
            cam = cam / (cam * cam).sum(dim=1, keepdim=True).sqrt()
            img, inp, cam, vel = [x.to(device=DEVICE, dtype=T) for x in (img, inp, cam, vel)]
            loss = torch.nn.functional.mse_loss(self(img, inp, cam, vel), inp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss.item())
        for param in self.visual.parameters():
            param.requires_grad = True
        return self

# State, Action -> QValue
# 
# This model should predict how useful the collision avoidance guidance is. 
# Realistically, it can just predict how (un)likely a crash is going to happen given state & controller input, 
# where 'controller input' here is the guidance from the actor model.
class DriverCriticModel(DriverModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_shapes = OrderedDict({
            'IMG': self.input_shapes['IMG'], 
            'INP': self.input_shapes['INP'], 
            'CAM': self.input_shapes['CAM'], 
            'VEL': self.input_shapes['VEL'],
            'ACT': self.input_shapes['INP']
        })
        self.post_collate = nn.Sequential(
            nn.Linear(self.controller_input_size, self.controller_input_size),
            nn.ELU(),
            nn.Linear(self.controller_input_size, 1)
        )

    def forward(self, IMG, INP, CAM, VEL, ACT):
        *_, Y = super().forward(IMG, ACT, CAM, VEL)
        Y = nn.functional.elu(Y)
        Q = self.post_collate(Y)
        return Q
