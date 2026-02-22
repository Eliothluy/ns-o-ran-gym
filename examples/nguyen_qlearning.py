#!/usr/bin/env python3
"""
Q-Learning for PRB Allocation in O-RAN

Implementation of Algorithm 1 from:
"Resource Allocation for Open Radio Access Networks Using Reinforcement Learning"
Nguyen et al., ATC 2025

This script implements:
- Q-learning with shared Q-table across RUs (Section V.A)
- Epsilon-greedy policy (Section V.A.2)
- State discretization for manageable Q-table size
- Training loop with multiple episodes
"""

import argparse
import json
import numpy as np
import pickle
import os
from collections import defaultdict
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.nguyen_env import NguyenEnv


class QLearningAgent:
    """
    Q-Learning agent with shared Q-table for all RUs.
    Implements Algorithm 1 from Nguyen et al.
    """

    def __init__(self, num_ues: int, num_rus: int = 2,
                 alpha: float = 0.15, gamma: float = 0.8,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.001,
                 prb_bins: int = 10):
        """
        Initialize Q-learning agent.

        Args:
            num_ues: Total number of UEs (K in article)
            num_rus: Number of RUs (I in article)
            alpha: Learning rate (default 0.15 from Section VI.A)
            gamma: Discount factor (default 0.8 from Section VI.A)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon (Eq. 9)
            prb_bins: Number of bins for PRB discretization
        """
        self.num_ues = num_ues
        self.num_rus = num_rus
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.prb_bins = prb_bins

        # Shared Q-table: Q[state][action] = value
        self.q_table = defaultdict(lambda: defaultdict(float))

        # UE to RU mapping
        ues_per_ru = num_ues // num_rus
        self.ue_to_ru = {ue: ue // ues_per_ru if ues_per_ru > 0 else 0
                         for ue in range(num_ues)}

        # Action space per RU
        self.actions_per_ru = self._compute_actions_per_ru()

        # Training statistics
        self.episode_rewards = []
        self.episode_acceptance_rates = []

    def _compute_actions_per_ru(self):
        """Compute valid actions for each RU."""
        actions = {}
        for ru in range(self.num_rus):
            ues_in_ru = [ue for ue in range(self.num_ues)
                         if self.ue_to_ru[ue] == ru]
            ru_actions = []
            for k_from in ues_in_ru:
                for k_to in ues_in_ru:
                    if k_from != k_to:
                        action = k_from * self.num_ues + k_to
                        ru_actions.append(action)
            # Add no-op action
            ru_actions.append(-1)
            actions[ru] = ru_actions
        return actions

    def discretize_state(self, state: tuple, max_prb: int = 79) -> tuple:
        """
        Discretize continuous state to reduce Q-table size.

        Args:
            state: Raw state tuple (prb_allocations + satisfaction_status)
            max_prb: Maximum PRBs per UE

        Returns:
            Discretized state tuple
        """
        discretized = []
        half = len(state) // 2

        # Discretize PRB allocations (first half of state)
        for i in range(half):
            prb = state[i]
            # Bin PRBs into discrete levels
            bin_size = max_prb / self.prb_bins
            bin_idx = min(int(prb / bin_size), self.prb_bins - 1) if bin_size > 0 else 0
            discretized.append(bin_idx)

        # Satisfaction status is already binary (second half)
        for i in range(half, len(state)):
            discretized.append(state[i])

        return tuple(discretized)

    def get_state_for_ru(self, full_state: tuple, ru_id: int) -> tuple:
        """
        Extract state relevant to a specific RU.

        Args:
            full_state: Full environment state
            ru_id: RU identifier

        Returns:
            State tuple for the specified RU
        """
        ues_in_ru = [ue for ue in range(self.num_ues)
                     if self.ue_to_ru[ue] == ru_id]

        half = len(full_state) // 2
        ru_state = []

        # PRB allocations for UEs in this RU
        for ue in ues_in_ru:
            if ue < half:
                ru_state.append(full_state[ue])

        # Satisfaction status for UEs in this RU
        for ue in ues_in_ru:
            if ue + half < len(full_state):
                ru_state.append(full_state[ue + half])

        return tuple(ru_state)

    def select_action(self, state: tuple, ru_id: int) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            ru_id: RU identifier

        Returns:
            Selected action
        """
        valid_actions = self.actions_per_ru[ru_id]

        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(valid_actions)
        else:
            # Exploitation: best action from Q-table
            state_key = (ru_id, state)
            q_values = {a: self.q_table[state_key][a] for a in valid_actions}

            if not q_values or all(v == 0 for v in q_values.values()):
                return np.random.choice(valid_actions)

            return max(q_values, key=q_values.get)

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, ru_id: int):
        """
        Update Q-table using Q-learning update rule (Eq. 12).

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            ru_id: RU identifier
        """
        state_key = (ru_id, state)
        next_state_key = (ru_id, next_state)
        valid_actions = self.actions_per_ru[ru_id]

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # Max Q-value for next state
        next_q_values = [self.q_table[next_state_key][a] for a in valid_actions]
        max_next_q = max(next_q_values) if next_q_values else 0

        # Q-learning update (Eq. 8/12)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

    def decay_epsilon(self):
        """Decay epsilon according to Equation 9."""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def save(self, filepath: str):
        """Save Q-table and agent parameters to file."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_acceptance_rates': self.episode_acceptance_rates,
            'params': {
                'num_ues': self.num_ues,
                'num_rus': self.num_rus,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'prb_bins': self.prb_bins
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load Q-table and agent parameters from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
        self.epsilon = data['epsilon']
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_acceptance_rates = data.get('episode_acceptance_rates', [])
        print(f"Agent loaded from {filepath}")


def train(env: NguyenEnv, agent: QLearningAgent, num_episodes: int,
          steps_per_ru: int = 10, verbose: bool = True):
    """
    Train the Q-learning agent.

    Implements Algorithm 1 from the paper.

    Args:
        env: Nguyen environment
        agent: Q-learning agent
        num_episodes: Number of training episodes
        steps_per_ru: Steps per RU per episode (step_ru in Algorithm 1)
        verbose: Print training progress
    """
    print(f"\n{'='*60}")
    print(f"Starting Q-Learning Training")
    print(f"Episodes: {num_episodes}, Steps per RU: {steps_per_ru}")
    print(f"Alpha: {agent.alpha}, Gamma: {agent.gamma}")
    print(f"Epsilon: {agent.epsilon} -> {agent.epsilon_min}")
    print(f"{'='*60}\n")

    for episode in range(num_episodes):
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0

        # Flatten observation if nested
        if isinstance(obs, tuple) and len(obs) == 1:
            obs = obs[0]

        # Algorithm 1: Loop over each RU
        for ru_id in range(agent.num_rus):

            # Loop over steps for this RU
            for step in range(steps_per_ru):
                # Get state for this RU
                state = agent.get_state_for_ru(obs, ru_id)
                discretized_state = agent.discretize_state(state)

                # Select action using epsilon-greedy
                action = agent.select_action(discretized_state, ru_id)

                # Execute action
                if action == -1:
                    # No-op action
                    env_action = (0, 0)  # Same UE, no transfer
                else:
                    k_from = action // agent.num_ues
                    k_to = action % agent.num_ues
                    env_action = (k_from, k_to)

                # Take step in environment
                next_obs, reward, terminated, truncated, info = env.step(env_action)

                # Flatten observation if nested
                if isinstance(next_obs, tuple) and len(next_obs) == 1:
                    next_obs = next_obs[0]

                # Get next state for this RU
                next_state = agent.get_state_for_ru(next_obs, ru_id)
                discretized_next_state = agent.discretize_state(next_state)

                # Update Q-table (Eq. 12)
                agent.update(discretized_state, action, reward,
                            discretized_next_state, ru_id)

                episode_reward += reward
                episode_steps += 1
                obs = next_obs

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        # Decay epsilon (Eq. 9)
        agent.decay_epsilon()

        # Get metrics
        metrics = env.get_metrics()
        agent.episode_rewards.append(episode_reward)
        agent.episode_acceptance_rates.append(metrics['acceptance_rate'])

        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-10:])
            avg_acceptance = np.mean(agent.episode_acceptance_rates[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (10): {avg_reward:.2f} | "
                  f"Acceptance: {metrics['acceptance_rate']:.1f}% | "
                  f"Avg Accept (10): {avg_acceptance:.1f}% | "
                  f"Epsilon: {agent.epsilon:.4f}")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Avg Reward (100): {np.mean(agent.episode_rewards[-100:]):.2f}")
    print(f"Final Avg Acceptance (100): {np.mean(agent.episode_acceptance_rates[-100:]):.1f}%")
    print(f"Q-table size: {len(agent.q_table)} states")
    print(f"{'='*60}\n")


def evaluate(env: NguyenEnv, agent: QLearningAgent, num_episodes: int = 10):
    """
    Evaluate trained agent.

    Args:
        env: Nguyen environment
        agent: Trained Q-learning agent
        num_episodes: Number of evaluation episodes
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Agent ({num_episodes} episodes)")
    print(f"{'='*60}\n")

    # Disable exploration during evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    rewards = []
    acceptance_rates = []
    throughputs = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0

        if isinstance(obs, tuple) and len(obs) == 1:
            obs = obs[0]

        done = False
        steps = 0

        while not done and steps < 100:
            # Use RU 0 for evaluation (could alternate)
            ru_id = steps % agent.num_rus
            state = agent.get_state_for_ru(obs, ru_id)
            discretized_state = agent.discretize_state(state)

            action = agent.select_action(discretized_state, ru_id)

            if action == -1:
                env_action = (0, 0)
            else:
                k_from = action // agent.num_ues
                k_to = action % agent.num_ues
                env_action = (k_from, k_to)

            obs, reward, terminated, truncated, info = env.step(env_action)

            if isinstance(obs, tuple) and len(obs) == 1:
                obs = obs[0]

            episode_reward += reward
            steps += 1
            done = terminated or truncated

        metrics = env.get_metrics()
        rewards.append(episode_reward)
        acceptance_rates.append(metrics['acceptance_rate'])
        throughputs.append(metrics['sum_throughput'])

        print(f"Episode {episode + 1}: "
              f"Reward={episode_reward:.2f}, "
              f"Acceptance={metrics['acceptance_rate']:.1f}%, "
              f"Throughput={metrics['sum_throughput']:.2f} Mbps")

    # Restore epsilon
    agent.epsilon = original_epsilon

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"  Average Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Average Acceptance Rate: {np.mean(acceptance_rates):.1f}% +/- {np.std(acceptance_rates):.1f}%")
    print(f"  Average Throughput: {np.mean(throughputs):.2f} +/- {np.std(throughputs):.2f} Mbps")
    print(f"{'='*60}\n")

    return {
        'rewards': rewards,
        'acceptance_rates': acceptance_rates,
        'throughputs': throughputs
    }


def main():
    parser = argparse.ArgumentParser(
        description="Q-Learning for PRB Allocation in O-RAN (Nguyen et al. ATC 2025)")

    # Environment arguments
    parser.add_argument("--config", type=str,
                        default="src/environments/scenario_configurations/nguyen_use_case.json",
                        help="Path to configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Output folder for simulation data")
    parser.add_argument("--ns3_path", type=str,
                        default="/home/elioth/Documentos/artigoJussi/ns-3-mmwave-oran",
                        help="Path to ns-3 mmWave O-RAN")
    parser.add_argument("--optimized", action="store_true",
                        help="Run ns-3 in optimized mode")

    # Training arguments
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--steps_per_ru", type=int, default=10,
                        help="Steps per RU per episode")
    parser.add_argument("--ues", type=int, default=None,
                        help="Override number of UEs")

    # Q-learning hyperparameters (from Section VI.A)
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.8,
                        help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument("--epsilon_min", type=float, default=0.01,
                        help="Minimum exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.001,
                        help="Epsilon decay rate")
    parser.add_argument("--prb_bins", type=int, default=10,
                        help="Number of bins for PRB discretization")

    # Other arguments
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save trained agent")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load pre-trained agent")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate, don't train")

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config) as f:
            scenario_configuration = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration")
        scenario_configuration = {
            "simTime": [10],
            "ues": [15],
            "RngRun": [1],
            "indicationPeriodicity": [0.100],
            "useSemaphores": [1],
            "embbDataRate": [15.0],
            "controlFileName": ["nguyen_actions.csv"]
        }

    # Override UEs if specified
    if args.ues is not None:
        scenario_configuration["ues"] = [args.ues]

    num_ues = scenario_configuration.get("ues", [15])[0]

    print(f"\n{'='*60}")
    print(f"Nguyen Q-Learning for O-RAN PRB Allocation")
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Number of UEs: {num_ues}")
    print(f"NS-3 Path: {args.ns3_path}")
    print(f"{'='*60}\n")

    # Create environment
    print("Creating environment...")
    env = NguyenEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=args.optimized
    )
    print("Environment created!")

    # Create or load agent
    agent = QLearningAgent(
        num_ues=num_ues,
        num_rus=NguyenEnv.NUM_RU,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        prb_bins=args.prb_bins
    )

    if args.load_path and os.path.exists(args.load_path):
        agent.load(args.load_path)

    # Train or evaluate
    if not args.eval_only:
        train(env, agent, args.episodes, args.steps_per_ru)

        # Save agent
        if args.save_path:
            agent.save(args.save_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"nguyen_agent_{num_ues}ues_{timestamp}.pkl"
            agent.save(default_path)

    # Evaluate
    if args.eval_episodes > 0:
        evaluate(env, agent, args.eval_episodes)


if __name__ == "__main__":
    main()
