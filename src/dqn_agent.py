# src/dqn_agent.py

from collections import deque
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack([s for s in next_states if s is not None]) if not all(ns is None for ns in next_states) else None,
            np.array([ns is not None for ns in next_states], dtype=bool),
            np.array(dones, dtype=bool),
            next_states,  # original list, to align mask indices
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        min_buffer_size: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        target_update_freq: int = 1000,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # epsilon-greedy schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0

    def epsilon(self):
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        state: np.ndarray of shape (state_dim,)
        """
        if (not eval_mode) and (random.random() < self.epsilon()):
            action = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.q_net(s)
                action = int(torch.argmax(q_values, dim=1).item())
        if not eval_mode:
            self.steps_done += 1
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Tuple[float, int]:
        """
        Perform one optimization step on a batch from replay buffer.
        Returns (loss_value, batch_size_used) or (0.0, 0) if not enough data yet.
        """
        if len(self.replay_buffer) < max(self.batch_size, self.min_buffer_size):
            return 0.0, 0

        # Sample batch
        (
            states_np,
            actions_np,
            rewards_np,
            non_final_next_states_np,
            non_final_mask,
            dones_np,
            next_states_list,
        ) = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states_np).float().to(self.device)
        actions = torch.from_numpy(actions_np).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards_np).float().unsqueeze(1).to(self.device)
        dones = torch.from_numpy(dones_np.astype(np.uint8)).float().unsqueeze(1).to(self.device)

        # Current Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # Next-state values
        next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
        if non_final_next_states_np is not None:
            non_final_next_states = torch.from_numpy(non_final_next_states_np).float().to(self.device)
            non_final_q_vals = self.target_net(non_final_next_states).max(1)[0].detach().unsqueeze(1)
            # Mask indices
            idx = 0
            for i, ns in enumerate(next_states_list):
                if ns is not None:
                    next_state_values[i] = non_final_q_vals[idx]
                    idx += 1

        # Compute target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        targets = rewards + self.gamma * next_state_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item()), states.size(0)
