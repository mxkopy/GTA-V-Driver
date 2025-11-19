from threading import stack_size
import time
import torch
import torch.nn as nn
import copy
import config
from ipc import FLAGS
from environment import Environment, ReplayBuffer, State, Action, Reward, NextState, Final, Transition


class DeterministicPolicyGradient:

    def __init__(
            self,
            actor: nn.Module, 
            critic: nn.Module,
            environment: Environment = Environment(),
            replay_buffer: ReplayBuffer = ReplayBuffer(),
            lr: float = 1e-3,
            gamma: float = 0.80,
            tau: float = 0.995,
            action_min: torch.Tensor | float = -float('inf'),
            action_max: torch.Tensor | float = float('inf'),
            noise_distribution = torch.distributions.Normal(0, 1)
        ):
        self.actor = actor
        self.critic = critic
        self.environment = environment
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = 100
        self.num_updates = 4
        self.action_min = action_min
        self.action_max = action_max
        self.noise_distribution = noise_distribution
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters())
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters())

    def get_action(self, state: State, noise_scale: float = 1.0) -> Action:
        action: Action = self.actor(state)
        noise: torch.Tensor = self.noise_distribution.sample(action.size()) * noise_scale
        noise: torch.Tensor = noise.to(device=action.device)
        action = torch.clamp(action + noise, min=self.action_min, max=self.action_max)
        return action

    def act(self, state: State, action: Action) -> Transition:
        reward, nextstate, final = self.environment.perform_action(action)
        return Transition(state, action, reward, nextstate, final).to(device=action.device)

    def run_episode(self, steps: int = 1000, offset: int=0):
        n = offset
        print('getting first observation')
        state: State = self.environment.observe()
        final: bool = False
        print('running episode')
        while n < steps + offset and not final:
            n += 1
            state: State = state.to(device=config.device)
            action: Action = self.get_action(state)
            transition: Transition = self.act(state, action)
            self.replay_buffer += transition.to(device='cpu')
            state: State = transition.nextstate
            final: bool = transition.final.item()
        

    def update_critic(self, batch):
        print('updating critic')
        with torch.no_grad():
            target: torch.Tensor = batch.reward + (~batch.final) * self.gamma * self.critic_target(batch.nextstate, self.actor_target(batch.nextstate))
        self.critic_optimizer.zero_grad()
        critic_loss: torch.Tensor = (self.critic(batch.state, batch.action) - target).square().mean(dim=0).sum()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()


    def update_actor(self, batch):
        print('updating actor')
        self.actor_optimizer.zero_grad()
        actor_loss: torch.Tensor = -self.critic(batch.state, self.actor(batch.state)).mean(dim=0).sum()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

    def soft_update(self, target, policy):
        target_state_dict = target.state_dict()
        policy_state_dict = policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = (target_state_dict[key] * self.tau) + (policy_state_dict[key] * (1 - self.tau))
        target.load_state_dict(target_state_dict)
        
    def optimize_step(self):
        self.environment.pause_training()
        batch: Transition = self.replay_buffer.sample(self.batch_size).to(device=config.device)
        self.update_critic(batch)
        self.update_actor(batch)
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        self.environment.resume_training()

# TODO: This component of the actor/critic model should be trained separately & be otherwise static
# Otherwise, it might take too long to train the rest of the network 
# What it really should do is provide per-pixel kinematic information like distance & velocity relative to the viewer
class VisualModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            *(nn.Sequential(
                nn.Conv2d(config.visual_channels[i], config.visual_channels[i+1], (3, 3), padding='same'),
                nn.ELU()
            ) for i in range(len(config.visual_channels)-1)),
            nn.AdaptiveAvgPool2d(config.visual_features_size)
        )

    def forward(self, img):
        return self.model(img / 255)

class DriverModelBase(nn.Module):

    def __init__(self):
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
        self.collate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, config.state_sizes['controller'][0])
        )
 
    def example_inputs(self, batch_size=1):
        return tuple(torch.rand(batch_size, *shape) for shape in self.input_shapes.values())

    def forward(self, state: State):
        visual_features = self.visual(state.image).reshape(state.image.shape[0], self.visual_output_size, 1)
        camera_rotation_matrix = self.rotation_matrix(state.camera_direction).reshape(-1, self.visual_output_size, self.visual_output_size)
        rotated_visual_features = torch.bmm(camera_rotation_matrix, visual_features).squeeze(-1)
        X = torch.cat((rotated_visual_features.flatten(start_dim=1), state.velocity, state.controller), dim=-1)
        Y = self.collate(X)
        return visual_features, rotated_visual_features, camera_rotation_matrix, X, Y

# State -> Action
class DriverActorModel(DriverModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, state: State):
        *_, action = super().forward(state)
        return action

    def jit(self):
        state = State.rand(batch_size=2)
        return torch.export.export(
            self,
            args=(state,),
            dynamic_shapes=state.dynamic_shapes()
        ).module()


    # Maybe to get input-modifying rather than overriding
    # we might want to train the initial network to pass the input through. 
    def train_for_input_passthrough(self, batchsize=16, lr=1e-4, cutoff=1e-2):
        opt = torch.optim.Adam(self.parameters(), lr)
        loss = cutoff
        for param in self.visual.parameters():
            param.requires_grad = False
        while loss >= cutoff:
            img = torch.rand(batchsize, *config.state_sizes['image']) * 255
            inp = torch.rand(batchsize, 4)
            inp[:, :2] *= 2
            inp[:, :2] -= 1
            cam = torch.rand(batchsize, 3)*2 -1
            vel = torch.rand(batchsize, 3)*2 -1
            cam = cam / (cam * cam).sum(dim=1, keepdim=True).sqrt()
            img, inp, cam, vel = [x.to(device=config.device) for x in (img, inp, cam, vel)]
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
        self.post_collate = nn.Sequential(
            nn.Linear(config.state_sizes['controller'][0], config.state_sizes['controller'][0]),
            nn.ELU(),
            nn.Linear(config.state_sizes['controller'][0], 1)
        )

    def forward(self, state: State, action: Action):
        state = state._replace(controller=action)
        *_, Y = super().forward(state)
        Y = nn.functional.elu(Y)
        Q = self.post_collate(Y)
        return Q

    def jit(self):
        dynamic_shapes = torch.export.ShapesCollection()
        state = State.rand(batch_size=2)
        action = torch.rand(2, *config.action_sizes['controller'])
        args = (state, action)
        args[0].dynamic_shapes(dynamic_shapes)
        dynamic_shapes[action] = {0: torch.export.Dim.DYNAMIC}
        return torch.export.export(
            self,
            args=args,
            dynamic_shapes=dynamic_shapes
        ).module()

import gc
actor, critic = DriverActorModel().to(device=config.device, dtype=torch.float32).jit(), DriverCriticModel().to(device=config.device, dtype=torch.float32).jit()

ddpg = DeterministicPolicyGradient(actor, critic)
ddpg.environment.game_ipc.set_flag(FLAGS.IS_TRAINING, True)
while True:
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        ddpg.run_episode()
    ddpg.optimize_step()