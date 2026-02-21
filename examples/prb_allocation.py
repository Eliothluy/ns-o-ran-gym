"""
Exemplo de uso do NguyenPrbEnv com Q-Learning

Implementa o algoritmo Q-Learning do artigo Nguyen et al.
para alocacao dinamica de PRBs em O-RAN.

Uso:
    python prb_allocation.py --ns3_path /caminho/para/ns-3-mmwave-oran
"""

import argparse
import json
import numpy as np
from environments.nguyen_env import NguyenPrbEnv


class QLearningAgent:
    """Agente Q-Learning para alocacao de PRBs."""

    def __init__(self, alpha=0.15, gamma=0.8, epsilon=1.0, epsilon_min=0.01, delta_decay=0.005):
        """
        Inicializa o agente Q-Learning.

        Args:
            alpha: Taxa de aprendizado
            gamma: Fator de desconto
            epsilon: Taxa de exploracao inicial
            epsilon_min: Taxa de exploracao minima
            delta_decay: Decaimento linear do epsilon
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.delta_decay = delta_decay

        # Q-table por RU: {ru_id: {state_str: {action: q_value}}}
        self.q_tables = {}

        # Estados e acoes anteriores para update
        self.last_states = {}
        self.last_actions = {}

    def choose_action(self, state_str: str, ues_list: list, ru_id: int) -> tuple:
        """
        Escolhe acao usando politica epsilon-greedy.

        Args:
            state_str: String representando o estado
            ues_list: Lista de IMSIs do RU
            ru_id: ID do RU

        Returns:
            Tupla (imsi_from, imsi_to)
        """
        if len(ues_list) < 2:
            return (ues_list[0], ues_list[0]) if ues_list else (0, 0)

        # Inicializar Q-table para RU se necessario
        if ru_id not in self.q_tables:
            self.q_tables[ru_id] = {}

        if np.random.rand() < self.epsilon:
            # Exploracao: acao aleatoria
            k_from = np.random.choice(ues_list)
            k_to = np.random.choice(ues_list)
            while k_from == k_to:
                k_to = np.random.choice(ues_list)
            return (k_from, k_to)
        else:
            # Explotacao: melhor acao conhecida
            if state_str not in self.q_tables[ru_id] or not self.q_tables[ru_id][state_str]:
                # Estado desconhecido: acao aleatoria
                k_from = np.random.choice(ues_list)
                k_to = np.random.choice(ues_list)
                while k_from == k_to:
                    k_to = np.random.choice(ues_list)
                return (k_from, k_to)
            else:
                # Melhor acao para este estado
                return max(
                    self.q_tables[ru_id][state_str],
                    key=self.q_tables[ru_id][state_str].get
                )

    def update(self, ru_id: int, state: str, action: tuple, reward: float, next_state: str):
        """
        Atualiza Q-table usando equacao de Bellman.

        Args:
            ru_id: ID do RU
            state: Estado anterior
            action: Acao tomada
            reward: Recompensa recebida
            next_state: Novo estado
        """
        if ru_id not in self.q_tables:
            self.q_tables[ru_id] = {}
        if state not in self.q_tables[ru_id]:
            self.q_tables[ru_id][state] = {}
        if action not in self.q_tables[ru_id][state]:
            self.q_tables[ru_id][state][action] = 0.0
        if next_state not in self.q_tables[ru_id]:
            self.q_tables[ru_id][next_state] = {}

        # Max Q(s', a')
        max_next_q = max(self.q_tables[ru_id][next_state].values()) \
            if self.q_tables[ru_id][next_state] else 0.0

        # Bellman update
        current_q = self.q_tables[ru_id][state][action]
        self.q_tables[ru_id][state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

    def decay_epsilon(self):
        """Decai epsilon linearmente."""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.delta_decay)


def main():
    parser = argparse.ArgumentParser(description="PRB Allocation with Q-Learning")
    parser.add_argument("--config", type=str,
                        default="/home/eliothluy/Documentos/artigoJussi/ns-o-ran-gym/src/environments/scenario_configurations/nguyen_use_case.json",
                        help="Path to configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Path to output folder")
    parser.add_argument("--ns3_path", type=str,
                        default="/home/eliothluy/Documentos/artigoJussi/ns-3-mmwave-oran",
                        help="Path to ns-3 mmWave O-RAN")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to run")
    parser.add_argument("--optimized", action="store_true",
                        help="Use optimized ns-3 build")

    args = parser.parse_args()

    # Carregar configuracao
    try:
        with open(args.config) as f:
            scenario_config = json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        return

    print("Creating NguyenPrbEnv...")
    env = NguyenPrbEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_config,
        output_folder=args.output_folder,
        optimized=args.optimized
    )
    print("Environment created!")

    # Criar agente Q-Learning
    agent = QLearningAgent()

    print("Launching reset...", end=' ', flush=True)
    obs, info = env.reset()
    print("done")

    print(f"Initial observation: {obs}")
    print(f"Info: {info}")

    # Loop de treinamento
    for step in range(1, args.num_steps + 1):
        actions = {}

        # Para cada RU, escolher acao
        for ru_id, ru_data in obs.items():
            state_str = ru_data.get('state_str', '')
            ues_list = list(ru_data['ues'].keys())

            # Atualizar Q-table do passo anterior
            if ru_id in agent.last_states:
                reward = ru_data['total_throughput']  # Reward local do RU
                agent.update(
                    ru_id,
                    agent.last_states[ru_id],
                    agent.last_actions[ru_id],
                    reward,
                    state_str
                )

            # Escolher nova acao
            action = agent.choose_action(state_str, ues_list, ru_id)
            actions[ru_id] = action

            # Salvar para proximo update
            agent.last_states[ru_id] = state_str
            agent.last_actions[ru_id] = action

        # Executar step
        print(f"Step {step}", end=' ', flush=True)
        obs, reward, terminated, truncated, info = env.step(actions)
        print("done")

        # Log
        print(f"  Actions: {actions}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Epsilon: {agent.epsilon:.4f}")

        for ru_id, ru_data in obs.items():
            print(f"  RU {ru_id}: throughput={ru_data['total_throughput']:.2f} Mbps, "
                  f"QoS satisfied={ru_data['qos_satisfied']}/{len(ru_data['ues'])}")

        # Decair epsilon
        agent.decay_epsilon()

        if terminated:
            print("Simulation terminated")
            break

        if truncated:
            print("Episode truncated")
            break

    print("\nTraining complete!")
    print(f"Final epsilon: {agent.epsilon:.4f}")

    # Estatisticas finais
    total_entries = sum(
        len(states) for states in agent.q_tables.values()
    )
    print(f"Q-table entries: {total_entries}")


if __name__ == '__main__':
    main()
