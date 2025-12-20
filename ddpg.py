import torch
import torch.nn as nn
import copy
import gc
import config
from ipc import FLAGS
from environment import Environment, ReplayBuffer, Action, State, Transition

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
        self.actor_target = copy.deepcopy(self.actor).to(device='cuda')
        self.critic_target = copy.deepcopy(self.critic).to(device='cuda')
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters())
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters())

    def compute_action(self, state: State, noise_scale: float = 0) -> Action:
        action: Action = self.actor(state)
        noise: torch.Tensor = self.noise_distribution.sample(action.size()) * noise_scale
        noise: torch.Tensor = noise.to(device=action.device)
        action = torch.clamp(action + noise, min=self.action_min, max=self.action_max)
        return action

    def act(self, state: State, action: Action) -> Transition:
        reward, nextstate, final = self.environment.perform_action(action)
        return Transition(state, action, reward, nextstate, final).to(device=action.device)

    def run_episode(self, steps: int = 1000, offset: int=0):
        gc.collect()
        torch.cuda.empty_cache()
        self.environment.reset()
        self.replay_buffer.reset()
        n = offset
        state: State = self.environment.observe()
        final: bool = False
        r = torch.rand(1).item()

        with torch.no_grad(), torch.autocast(device_type=config.device):
            while n < steps + offset and not final:
                n += 1
                state: State = state.to(device=config.device)
                action: Action = self.compute_action(state)
                action[:, 2] = 0
                action[:, 3] = r
                print(action)
                transition: Transition = self.act(state, action)
                self.replay_buffer += transition.to(device='cpu')
                state: State = transition.nextstate
                final: bool = transition.final.item()        

    def update_critic(self, batch):
        with torch.no_grad(), torch.autocast(device_type='cuda'):
            target: torch.Tensor = batch.reward + (~batch.final) * self.gamma * self.critic_target(batch.nextstate, self.actor_target(batch.nextstate))
        self.critic_optimizer.zero_grad()
        critic_loss: torch.Tensor = (self.critic(batch.state, batch.action) - target).square().mean(dim=0).sum()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

    def update_actor(self, batch):
        self.actor_optimizer.zero_grad()
        actions = self.actor(batch.state)
        actor_loss: torch.Tensor = -self.critic(batch.state, actions).mean(dim=0).sum()
        actor_loss += actions[:, 0:2].square().mean(dim=0).sum() * 0.001
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
        gc.collect()
        torch.cuda.empty_cache()
        self.environment.pause_training()
        batch: Transition = self.replay_buffer.sample(self.batch_size).to(device=config.device)
        self.update_critic(batch)
        self.update_actor(batch)
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        self.environment.resume_training()

    def train(self, episodes=float('inf')):
        n = 0
        while n < episodes:
            n += 1
            self.environment.game_state.set_flag(FLAGS.IS_TRAINING, True)
            print('Training')
            self.run_episode()
            self.environment.game_state.set_flag(FLAGS.IS_TRAINING, False)
            print('Optimizing')
            self.optimize_step()