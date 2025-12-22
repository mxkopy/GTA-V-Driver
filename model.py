from threading import stack_size
import time
import torch
import torch.nn as nn
import copy
import config
import gc
from ipc import FLAGS
from environment import Environment, ReplayBuffer, State, Action, Reward, NextState, Final, Transition
# from abc import ABC, abstractmethod

# TODO: This component of the actor/critic model should be trained separately & be otherwise static
# Otherwise, it might take too long to train the rest of the network 
# What it really should do is provide per-pixel kinematic information like distance & velocity relative to the viewer
class VisualModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        for in_channels, out_channels in zip(config.visual_channels, config.visual_channels[1:]):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3, 3), padding='same'),
                nn.ELU()
            )
            self.model.append(conv_layer)
        self.model.append(nn.AdaptiveAvgPool2d(config.visual_features_size))

    def forward(self, img):
        return self.model(img)

class DriverModelBase(nn.Module):

    def __init__(self, distribution: None | type[torch.distributions.Distribution] = None):
        super().__init__()
        self.visual = VisualModel()
        self.hidden_size = config.visual_features_size[0] * config.visual_features_size[1] * config.visual_channels[-1] + 1
        self.collate_mean = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, config.action_sizes['controller'][0])
        )
        self.collate_std = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, config.action_sizes['controller'][0])
        )
        self.distribution = distribution

    def forward(self, state: State):
        features = self.visual(state.image)
        features = features.reshape(features.size(0), -1)
        features = torch.cat((features, torch.square(state.velocity).sum(dim=1, keepdim=True).sqrt()), dim=1)
        if self.distribution is None:
            return self.collate_mean(features)
        else:
            return self.distribution(self.collate_mean(features), self.collate_std(features)).sample()

class DriverActorModel(DriverModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, state: State):
        action = super().forward(state)
        action[:, 0:2] = torch.tanh(action[:, 0:2])
        return action

    def jit(self):
        device = list(self.visual.parameters())[0].device
        state = State.rand(batch_size=2).to(device=device)
        return torch.export.export(
            self,
            args=(state,),
            dynamic_shapes=state.dynamic_shapes()
        ).module()

class DriverCriticModel(DriverModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_function = nn.Sequential(
            nn.Linear(self.hidden_size + config.action_sizes['controller'][0], self.hidden_size + config.action_sizes['controller'][0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size + config.action_sizes['controller'][0], self.hidden_size + config.action_sizes['controller'][0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size + config.action_sizes['controller'][0], self.hidden_size + config.action_sizes['controller'][0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size + config.action_sizes['controller'][0], self.hidden_size + config.action_sizes['controller'][0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size + config.action_sizes['controller'][0], 1)
        )

    def forward(self, state: State, action: Action):
        features = self.visual(state.image)
        features = features.reshape(features.size(0), -1)
        features = torch.cat((features, torch.square(state.velocity).sum(dim=1, keepdim=True).sqrt()), dim=1)
        value = self.value_function(torch.cat((features, action), dim=1))
        return value

    def jit(self):
        dynamic_shapes = torch.export.ShapesCollection()
        device = list(self.visual.parameters())[0].device
        state = State.rand(batch_size=2).to(device=device)
        action = torch.rand(2, *config.action_sizes['controller']).to(device=device)
        args = (state, action)
        args[0].dynamic_shapes(dynamic_shapes)
        dynamic_shapes[action] = {0: torch.export.Dim.DYNAMIC}
        return torch.export.export(
            self,
            args=args,
            dynamic_shapes=dynamic_shapes
        ).module()