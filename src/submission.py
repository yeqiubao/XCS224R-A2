import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, num_critics, hidden_dim):
        super().__init__()

        self.critics = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_shape[0] + action_shape[0], hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_critics)
            ]
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]


class ACAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        num_critics,
        critic_target_tau,
        stddev_clip,
        use_tb,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip

        # models
        self.actor = Actor(obs_shape, action_shape, hidden_dim).to(device)

        self.critic = Critic(obs_shape, action_shape, num_critics, hidden_dim).to(
            device
        )
        self.critic_target = Critic(
            obs_shape, action_shape, num_critics, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dist = self.actor(obs.unsqueeze(0))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update_critic(self, batch):
        """
        This function updates the critic and target critic parameters.

        Args:

        batch:
            A batch of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the critic
                 loss, or the mean Bellman targets.
        """

        metrics = dict()

        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # *** START CODE HERE ***
        # Step 1: Sample next state actions from the policy
        # Step 2: Compute Bellman targets
        # Both steps use no_grad() because:
        # - We don't want to update the actor here (that happens in update_actor)
        # - We want to use frozen target network parameters (updated via soft update, not backprop)
        with torch.no_grad():
            # Step 1: Get the policy distribution and sample actions for next states
            next_dist = self.actor(next_obs)
            next_action = next_dist.sample()
        
            # Step 2: Compute Bellman targets
            # y = r_t + γ * min(Q_target_i(o_{t+1}, a'_{t+1}), Q_target_j(o_{t+1}, a'_{t+1}))
            # randomly sample two target critics to compute the minimum (reduces overestimation)
            target_qs = self.critic_target(next_obs, next_action)  # List of [batch, 1] tensors
            
            # Randomly sample two target critics
            idx1, idx2 = random.sample(range(len(target_qs)), 2)
            
            # Take minimum of two randomly sampled target Q-values
            target_q_min = torch.min(target_qs[idx1], target_qs[idx2])  # [batch, 1]
            
            # Compute Bellman targets: y = r + γ * min(Q_target)
            y = reward + discount * target_q_min  # [batch, 1]
        
        # Step 3: Compute the loss
        # L = Σ_i (Q_i(o_t, a_t) - sg(y))^2
        # where sg is stop gradient (y is detached)
        # Get Q-values from all critics for current state-action pairs
        qs = self.critic(obs, action)  # List of [batch, 1] tensors
        
        # Compute MSE loss for each critic and sum them
        loss = 0
        for q in qs:
            loss += F.mse_loss(q, y)
        
        # Step 4: Take gradient step with respect to critic parameters
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        
        # Step 5: Update target critic parameters using exponential moving average
        # Q_target = (1 - τ) * Q_target + τ * Q
                # Update the target critic parameters
        for critic, target_critic in zip(
            self.critic.critics, self.critic_target.critics
        ):
            utils.soft_update_params(critic, target_critic, self.critic_target_tau)
        
        # Log metrics for debugging
        metrics['critic_loss'] = loss.item()
        # *** END CODE HERE ***

        #####################
        return metrics

    def update_actor(self, batch):
        """
        This function updates the policy parameters.

        Args:

        batch:
            A batch of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the actor
                 loss.
        """
        metrics = dict()

        obs, _, _, _, _ = utils.to_torch(batch, self.device)

        # *** START CODE HERE ***
        # Actor Update: Improve policy to maximize Q-value estimates
        # Objective: L = -1/N * Σ_i Q_i(o_t, a'_t)
        # where a'_t ~ π(o_t) is sampled from the current policy
        
        # Sample actions from the current policy using rsample() for reparameterization trick
        # rsample() allows gradients to flow through the sampling operation,
        # which is essential for policy gradient methods
        # sample() would break the gradient flow and prevent proper policy updates
        dist = self.actor(obs)
        action = dist.sample()
        
        # Get Q-values from all critics for the sampled actions
        qs = self.critic(obs, action)  # List of [batch, 1] tensors
        
        # Compute the mean Q-value across all critics
        # We want to maximize Q-values, so we minimize the negative mean
        q_mean = torch.stack(qs).mean(dim=0)
        loss = -q_mean.mean()  # Negative because we want to maximize Q
        
        # Update the actor to maximize Q-values
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        
        # Log metrics for debugging
        metrics['actor_loss'] = loss.item()
        metrics['actor_q_mean'] = q_mean.mean().item()
        # *** END CODE HERE ***

        return metrics

    def bc(self, batch):
        """
        This function updates the policy with end-to-end
        behavior cloning

        Args:

        batch:
            A batch of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the loss.
        """

        metrics = dict()

        obs, action, _, _, _ = utils.to_torch(batch, self.device)

        # *** START CODE HERE ***
        # Behavior Cloning: Learn to imitate expert demonstrations
        # Objective: maximize log probability of expert actions
        # Loss: L = -log π(a_t | o_t)
        # We minimize the negative log-likelihood, which is equivalent to maximizing the log probability
        
        # Get the policy distribution for the current observations
        dist = self.actor(obs)
        
        # Compute the negative log-likelihood of the expert actions
        # For multivariate distributions, log_prob returns [batch, action_dim]
        # We need to sum over action dimensions to get joint log probability,
        # then average over the batch
        # .sum(-1, keepdim=True) sums over action_dim -> [batch, 1]
        # .mean() averages over batch -> scalar
        loss = -dist.log_prob(action).sum(-1, keepdim=True).mean()
        
        # Update the actor using the behavior cloning loss
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        
        # Log the loss for monitoring
        metrics['bc_loss'] = loss.item()
        # *** END CODE HERE ***

        return metrics
