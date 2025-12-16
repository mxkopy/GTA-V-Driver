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
        return self.model(img / 255)

class DriverModelBase(nn.Module):

    def __init__(self, distribution: None | type[torch.distributions.Distribution] = None):
        super().__init__()
        self.visual = VisualModel()
        self.visual_output_size = config.visual_channels[-1] * config.visual_features_size[0] * config.visual_features_size[1]
        self.hidden_size = self.visual_output_size + config.state_sizes['controller'][0] + config.state_sizes['camera_direction'][0]
        # If rotation_matrix is any deeper than 1 layer, the model becomes incapable of input passthrough :0
        self.rotation_matrix = nn.Sequential(
            nn.Linear(3, self.visual_output_size),
            nn.LayerNorm(self.visual_output_size),
            nn.ELU(),
            nn.Linear(self.visual_output_size, self.visual_output_size**2)
        )
        self.collate_mean = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, config.state_sizes['controller'][0])
        )
        self.collate_std = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, config.state_sizes['controller'][0])
        )
        self.distribution = distribution
 
    def example_inputs(self, batch_size=1):
        return tuple(torch.rand(batch_size, *shape) for shape in self.input_shapes.values())

    def forward(self, state: State):
        visual_features = self.visual(state.image).reshape(state.image.shape[0], self.visual_output_size, 1)
        camera_rotation_matrix = self.rotation_matrix(state.camera_direction).reshape(-1, self.visual_output_size, self.visual_output_size)
        rotated_visual_features = torch.bmm(camera_rotation_matrix, visual_features).squeeze(-1)
        hidden_state = torch.cat((rotated_visual_features.flatten(start_dim=1), state.velocity, state.controller), dim=-1)
        if self.distribution is None:
            return self.collate_mean(hidden_state)
        else:
            return self.distribution(self.collate_mean(hidden_state), self.collate_std(hidden_state)).sample()

# State -> Action
class DriverActorModel(DriverModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, state: State):
        return super().forward(state)

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
        self.post_collate = nn.Sequential(
            nn.Linear(config.state_sizes['controller'][0], config.state_sizes['controller'][0]),
            nn.ELU(),
            nn.Linear(config.state_sizes['controller'][0], 1)
        )

    def forward(self, state: State, action: Action):
        state = state._replace(controller=action)
        Y = super().forward(state)
        Y = nn.functional.elu(Y)
        Q = self.post_collate(Y)
        return Q

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