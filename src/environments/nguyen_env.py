"""
Nguyen PRB Allocation Environment

Implementa o artigo "Resource Allocation for Open Radio Access Networks
Using Reinforcement Learning" (Nguyen et al., ATC 2025) como ambiente Gymnasium.

Comunicacao:
- Le: rl_state.csv (escrito pelo NS-3)
- Escreve: nguyen_actions.csv (lido pelo NS-3)
"""

from typing_extensions import override
import numpy as np
import pandas as pd
from nsoran.ns_env import NsOranEnv
import glob
import csv
import os
import math


class NguyenPrbEnv(NsOranEnv):
    """
    Ambiente Gymnasium para alocacao de PRBs baseado no artigo Nguyen et al.

    Estado: Para cada UE - (prbs_allocated, snr_db, throughput, qos_satisfied)
    Acao: Para cada RU - (imsi_from, imsi_to) indicando transferencia de delta PRBs
    Reward: Equacao 11 do artigo - maximiza throughput com restricao QoS
    """

    # Schema da tabela rl_state no datalake
    rl_state_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "ru_cellid": "INTEGER",
        "distance_m": "REAL",
        "snr_db": "REAL"
    }

    # Schema da tabela prb_allocation
    prb_allocation_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "ru_cellid": "INTEGER",
        "prbs_allocated": "INTEGER"
    }

    # Schema para grafana/metricas
    grafana_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "step": "INTEGER",
        "total_throughput": "REAL",
        "qos_satisfied_count": "INTEGER",
        "total_ues": "INTEGER",
        "reward": "REAL",
        "epsilon": "REAL"
    }

    def __init__(self, ns3_path: str, scenario_configuration: dict, output_folder: str, optimized: bool):
        """
        Inicializa o ambiente Nguyen PRB.

        Args:
            ns3_path: Caminho para o NS-3 mmWave O-RAN
            scenario_configuration: Dict com parametros do cenario
            output_folder: Pasta para outputs
            optimized: Se True, usa build otimizado do NS-3
        """
        super().__init__(
            ns3_path=ns3_path,
            scenario='scenario-nguyen',
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=['timestamp', 'ru_cellid', 'imsi_from', 'imsi_to', 'delta_prb'],
            log_file='NguyenActions.txt',
            control_file='nguyen_actions.csv'
        )

        self.folder_name = "Simulation"
        self.ns3_simulation_time = scenario_configuration['simTime'] * 1000  # ms

        # Parametros do artigo Nguyen
        self.num_rus = scenario_configuration.get('numRus', 2)
        self.prbs_per_ru = scenario_configuration.get('prbsPerRu', 79)
        self.delta_prb = scenario_configuration.get('deltaPrb', 2)
        self.ues_per_ru = scenario_configuration.get('ues', 15)

        # Constantes fisicas do artigo
        self.W_r_mbps = 0.18  # Banda por PRB em MHz [cite: 241]
        self.R_min = 10.0     # QoS: 10 Mbps [cite: 241]
        self.T_min = 1.0      # Fator de escala
        self.delta_reward = 0.9999  # Prioridade para satisfacao QoS [cite: 247]
        self.lambda_param = 0.05    # Ajuste fino

        # Estado interno: alocacao de PRBs {ru_id: {imsi: prbs}}
        self.prb_allocation = {}

        # Lista de UEs por RU {ru_id: [imsi1, imsi2, ...]}
        self.ues_by_ru = {}

        # Ultimo estado para cada RU (para Q-learning externo)
        self.last_states = {}

        # Contador de steps
        self.num_steps = 0

        # Cache do ultimo rl_state lido
        self.current_rl_state = None

    def _calculate_throughput(self, prbs: int, snr_db: float) -> float:
        """
        Equacao 2 do Artigo: R_k(t) = x_k^i * W_r * log2(1 + SNR)

        Args:
            prbs: Numero de PRBs alocados
            snr_db: SNR em dB

        Returns:
            Throughput em Mbps
        """
        if prbs <= 0:
            return 0.0
        snr_linear = 10 ** (snr_db / 10.0)
        return prbs * self.W_r_mbps * math.log2(1 + snr_linear)

    def _calculate_reward_from_throughputs(self, throughput_dict: dict) -> float:
        """
        Equacao 11 do Artigo: Reward function

        Args:
            throughput_dict: {imsi: throughput_mbps}

        Returns:
            Reward escalar
        """
        T_t = sum(throughput_dict.values())  # Throughput total
        pi_sum = 0  # Contador de UEs com QoS satisfeito
        penalty_bonus_sum = 0

        for R_k in throughput_dict.values():
            if R_k >= self.R_min:
                pi_sum += 1  # pi_k(t) = 1 [cite: 98, 103]
            penalty_bonus_sum += (R_k - self.R_min) / self.T_min  # [cite: 181]

        reward = (
            (1 - self.delta_reward) * (T_t / self.T_min) +
            (self.delta_reward * pi_sum) +
            (self.lambda_param * penalty_bonus_sum)
        )  # [cite: 179]

        return reward

    @override
    def _compute_action(self, action):
        """
        Converte acao do agente para formato NS-3.

        Formato de entrada (action):
            Lista de tuplas [(ru_id, imsi_from, imsi_to), ...]
            ou dict {ru_id: (imsi_from, imsi_to), ...}

        Formato de saida:
            [[ru_id, imsi_from, imsi_to, delta_prb], ...]
        """
        actions_list = []

        if isinstance(action, dict):
            # Formato: {ru_id: (imsi_from, imsi_to)}
            for ru_id, (imsi_from, imsi_to) in action.items():
                actions_list.append([ru_id, imsi_from, imsi_to, self.delta_prb])
        elif isinstance(action, list):
            for act in action:
                if len(act) == 3:
                    # (ru_id, imsi_from, imsi_to)
                    ru_id, imsi_from, imsi_to = act
                    actions_list.append([ru_id, imsi_from, imsi_to, self.delta_prb])
                elif len(act) == 4:
                    # (ru_id, imsi_from, imsi_to, delta_prb)
                    actions_list.append(list(act))

        # Atualizar alocacao interna de PRBs
        for ru_id, imsi_from, imsi_to, delta in actions_list:
            if ru_id in self.prb_allocation:
                if imsi_from in self.prb_allocation[ru_id]:
                    if self.prb_allocation[ru_id][imsi_from] >= delta:
                        self.prb_allocation[ru_id][imsi_from] -= delta
                        self.prb_allocation[ru_id][imsi_to] = \
                            self.prb_allocation[ru_id].get(imsi_to, 0) + delta

        return actions_list

    @override
    def _get_obs(self):
        """
        Constroi observacao a partir do rl_state.csv e alocacao de PRBs.

        Retorna:
            Tupla com observacao para cada RU:
            {ru_id: {
                'ues': {imsi: {'prbs': int, 'snr': float, 'throughput': float, 'qos': bool}},
                'total_throughput': float,
                'qos_satisfied': int,
                'state_str': str  # Para uso com Q-table
            }}
        """
        # Ler rl_state do datalake
        rl_state_data = self.datalake.read_table('rl_state')

        if not rl_state_data:
            return self._get_empty_obs()

        # Filtrar pelo ultimo timestamp
        latest_timestamp = max(row[0] for row in rl_state_data)
        current_data = [row for row in rl_state_data if row[0] == latest_timestamp]

        # Organizar por RU
        obs = {}
        for row in current_data:
            timestamp, imsi, ru_id, distance, snr = row

            if ru_id not in obs:
                obs[ru_id] = {'ues': {}, 'total_throughput': 0.0, 'qos_satisfied': 0}

            # Inicializar alocacao se necessario
            if ru_id not in self.prb_allocation:
                self.prb_allocation[ru_id] = {}
            if imsi not in self.prb_allocation[ru_id]:
                # Distribuir PRBs igualmente na primeira vez
                num_ues = len([r for r in current_data if r[2] == ru_id])
                prbs_per_ue = self.prbs_per_ru // max(1, num_ues)
                self.prb_allocation[ru_id][imsi] = prbs_per_ue

            # Atualizar lista de UEs
            if ru_id not in self.ues_by_ru:
                self.ues_by_ru[ru_id] = []
            if imsi not in self.ues_by_ru[ru_id]:
                self.ues_by_ru[ru_id].append(imsi)

            prbs = self.prb_allocation[ru_id][imsi]
            throughput = self._calculate_throughput(prbs, snr)
            qos_satisfied = throughput >= self.R_min

            obs[ru_id]['ues'][imsi] = {
                'prbs': prbs,
                'snr': snr,
                'distance': distance,
                'throughput': throughput,
                'qos': qos_satisfied
            }
            obs[ru_id]['total_throughput'] += throughput
            if qos_satisfied:
                obs[ru_id]['qos_satisfied'] += 1

        # Gerar state_str para cada RU (compativel com Q-table)
        for ru_id in obs:
            prb_tuple = tuple(sorted(self.prb_allocation[ru_id].items()))
            pi_tuple = tuple(sorted(
                (imsi, 1 if data['qos'] else 0)
                for imsi, data in obs[ru_id]['ues'].items()
            ))
            obs[ru_id]['state_str'] = str((prb_tuple, pi_tuple))

        self.current_rl_state = obs
        return obs

    def _get_empty_obs(self):
        """Retorna observacao vazia quando nao ha dados."""
        return {}

    @override
    def _compute_reward(self):
        """
        Calcula reward usando Equacao 11 do artigo.
        Combina rewards de todos os RUs.
        """
        if not self.current_rl_state:
            return 0.0

        total_reward = 0.0
        total_throughput = 0.0
        total_qos = 0
        total_ues = 0

        for ru_id, ru_data in self.current_rl_state.items():
            throughput_dict = {
                imsi: data['throughput']
                for imsi, data in ru_data['ues'].items()
            }
            total_reward += self._calculate_reward_from_throughputs(throughput_dict)
            total_throughput += ru_data['total_throughput']
            total_qos += ru_data['qos_satisfied']
            total_ues += len(ru_data['ues'])

        # Salvar metricas no grafana
        self._save_metrics(total_throughput, total_qos, total_ues, total_reward)

        return total_reward

    def _save_metrics(self, throughput: float, qos: int, ues: int, reward: float):
        """Salva metricas no datalake para visualizacao."""
        db_row = {
            'timestamp': self.last_timestamp,
            'ueImsiComplete': None,
            'step': self.num_steps,
            'total_throughput': throughput,
            'qos_satisfied_count': qos,
            'total_ues': ues,
            'reward': reward,
            'epsilon': 0.0  # Sera atualizado pelo agente externo se necessario
        }
        self.datalake.insert_data("grafana", db_row)

    @override
    def _init_datalake_usecase(self):
        """Inicializa tabelas especificas do caso de uso."""
        self.datalake._create_table("rl_state", self.rl_state_keys)
        self.datalake._create_table("prb_allocation", self.prb_allocation_keys)
        self.datalake._create_table("grafana", self.grafana_keys)
        return super()._init_datalake_usecase()

    @override
    def _fill_datalake_usecase(self):
        """Le rl_state.csv e insere no datalake."""
        rl_state_path = os.path.join(self.sim_path, 'rl_state.csv')

        if not os.path.exists(rl_state_path):
            return

        try:
            with open(rl_state_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    timestamp = int(row['timestamp_ms'])
                    if timestamp >= self.last_timestamp:
                        db_row = {
                            'timestamp': timestamp,
                            'ueImsiComplete': int(row['ue_imsi']),
                            'ru_cellid': int(row['ru_cellid']),
                            'distance_m': float(row['distance_m']),
                            'snr_db': float(row['snr_db'])
                        }
                        self.datalake.insert_data("rl_state", db_row)
                        self.last_timestamp = timestamp
        except (FileNotFoundError, KeyError, ValueError) as e:
            pass  # Arquivo ainda nao existe ou formato incorreto

    def get_ues_for_ru(self, ru_id: int) -> list:
        """Retorna lista de IMSIs para um RU especifico."""
        return self.ues_by_ru.get(ru_id, [])

    def get_prb_allocation(self, ru_id: int = None) -> dict:
        """
        Retorna alocacao atual de PRBs.

        Args:
            ru_id: Se especificado, retorna apenas para este RU

        Returns:
            Dict com alocacao {ru_id: {imsi: prbs}} ou {imsi: prbs}
        """
        if ru_id is not None:
            return self.prb_allocation.get(ru_id, {})
        return self.prb_allocation

    def get_state_str(self, ru_id: int) -> str:
        """Retorna string de estado para uso com Q-table."""
        if self.current_rl_state and ru_id in self.current_rl_state:
            return self.current_rl_state[ru_id].get('state_str', '')
        return ''

    def step(self, action):
        """
        Executa um passo no ambiente.

        Args:
            action: Acao do agente (dict ou lista de tuplas)

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        self.num_steps += 1
        return super().step(action)
