"""
Nguyen Environment - Resource Allocation for Open RAN Using Reinforcement Learning

Implementation of the Q-learning environment from:
"Resource Allocation for Open Radio Access Networks Using Reinforcement Learning"
Nguyen et al., ATC 2025

This environment implements:
- State: PRB allocation profile and throughput satisfaction status (Eq. 10)
- Action: Transfer PRBs from one UE to another (Section V.B.2)
- Reward: Weighted sum of throughput and user satisfaction (Eq. 11)
"""

from typing_extensions import override
import numpy as np
import pandas as pd
from nsoran.ns_env import NsOranEnv
import glob
import csv
import os


class NguyenEnv(NsOranEnv):
    """
    Gymnasium environment for PRB allocation in O-RAN using Q-learning.
    Based on Nguyen et al. ATC 2025 paper.
    """

    # Article parameters (Table I)
    NUM_RU = 2                      # Number of Radio Units
    NUM_PRB_PER_RU = 79             # PRBs per RU
    R_MIN_MBPS = 10.0               # Minimum QoS requirement (Mbps)
    T_MIN_MBPS = 1.0                # Scaling factor for throughput
    PRB_BANDWIDTH_MHZ = 0.18        # Bandwidth per PRB (MHz)
    TX_POWER_W = 0.01               # Power per PRB (W)

    # Q-learning hyperparameters (Section VI.A)
    DELTA = 0.9999                  # Weight factor (prioritizes user satisfaction)
    LAMBDA = 0.1                    # Penalty/bonus coefficient
    DELTA_PRB = 1                   # PRBs transferred per action

    def __init__(self, ns3_path: str, scenario_configuration: dict,
                 output_folder: str, optimized: bool):
        """
        Initialize the Nguyen environment.

        Args:
            ns3_path: Path to ns-3 mmWave O-RAN installation
            scenario_configuration: Dictionary with simulation parameters
            output_folder: Output folder for simulation data
            optimized: If True, run ns-3 in optimized mode
        """
        super().__init__(
            ns3_path=ns3_path,
            scenario='scenario-nguyen',
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=['timestamp', 'ueId', 'prbAllocation'],
            log_file='NguyenActions.txt',
            control_file='nguyen_actions.csv'
        )

        self.folder_name = "Simulation"
        self.ns3_simulation_time = scenario_configuration.get('simTime', [10])[0] * 1000

        # Number of UEs (K in the article)
        self.num_ues = scenario_configuration.get('ues', [15])[0]

        # State columns: PRBs allocated per UE + satisfaction status per UE
        # x_k^i(t) for each UE k and RU i, plus pi_k(t) for each UE k
        self.prb_columns = [f'prb_ue_{k}' for k in range(self.num_ues)]
        self.satisfied_columns = [f'satisfied_ue_{k}' for k in range(self.num_ues)]
        self.throughput_columns = [f'throughput_ue_{k}' for k in range(self.num_ues)]

        self.columns_state = self.prb_columns + self.satisfied_columns
        self.columns_reward = ['total_throughput', 'num_satisfied', 'throughput_gap']

        # Action space: (k_from, k_to) pairs
        # Action i = k_from * num_ues + k_to means transfer PRBs from k_from to k_to
        self.num_actions = self.num_ues * self.num_ues

        # Internal state tracking
        self.prb_allocation = {}  # {ue_id: num_prbs}
        self.ue_throughput = {}   # {ue_id: throughput_mbps}
        self.observations = None
        self.num_steps = 0

        # RU assignment (which UEs belong to which RU)
        self.ue_to_ru = {}
        ues_per_ru = self.num_ues // self.NUM_RU
        for ue in range(self.num_ues):
            self.ue_to_ru[ue] = ue // ues_per_ru if ues_per_ru > 0 else 0

    @override
    def _compute_action(self, action):
        """
        Convert agent action to ns-O-RAN format.

        Action is an integer representing (k_from, k_to) pair:
        - action = k_from * num_ues + k_to
        - Transfers DELTA_PRB PRBs from UE k_from to UE k_to

        Args:
            action: Integer action index or tuple (k_from, k_to)

        Returns:
            List of tuples [(ue_id, prb_allocation), ...]
        """
        if isinstance(action, (list, tuple)) and len(action) == 2:
            k_from, k_to = action
        else:
            k_from = action // self.num_ues
            k_to = action % self.num_ues

        # Validate action
        if k_from == k_to:
            # No-op action
            return self._get_current_allocation_actions()

        # Check if UEs are in the same RU (article constraint)
        if self.ue_to_ru.get(k_from, 0) != self.ue_to_ru.get(k_to, 0):
            # Cross-RU transfer not allowed, return current allocation
            return self._get_current_allocation_actions()

        # Transfer PRBs
        current_from = self.prb_allocation.get(k_from, 0)
        current_to = self.prb_allocation.get(k_to, 0)

        if current_from >= self.DELTA_PRB:
            self.prb_allocation[k_from] = current_from - self.DELTA_PRB
            self.prb_allocation[k_to] = current_to + self.DELTA_PRB

        return self._get_current_allocation_actions()

    def _get_current_allocation_actions(self):
        """Get current PRB allocation as action list for ns-3."""
        actions = []
        for ue_id in range(self.num_ues):
            prbs = self.prb_allocation.get(ue_id, 0)
            actions.append([ue_id, prbs])
        return actions

    @override
    def _get_obs(self):
        """
        Get current observation (state).

        State representation (Eq. 10):
        s_i = {x_k^i(t) | k in K}, {pi_k(t) | k in K}

        Returns:
            Tuple of state values
        """
        # Read KPMs from datalake
        kpms_raw = ["ueImsiComplete", "nrCellId", "RRU.PrbUsedDl",
                    "DRB.PdcpSduBitRateDl.UEID"]

        try:
            ue_kpms = self.datalake.read_kpms(self.last_timestamp, kpms_raw)
        except Exception:
            ue_kpms = []

        # Process KPMs to extract PRB allocation and throughput
        self._process_kpms(ue_kpms)

        # Build state vector
        state = []

        # PRB allocation for each UE: x_k^i(t)
        for ue in range(self.num_ues):
            prbs = self.prb_allocation.get(ue, 0)
            state.append(prbs)

        # Satisfaction status for each UE: pi_k(t)
        for ue in range(self.num_ues):
            throughput = self.ue_throughput.get(ue, 0)
            satisfied = 1 if throughput >= self.R_MIN_MBPS else 0
            state.append(satisfied)

        self.observations = state
        return tuple([tuple(state)])

    def _process_kpms(self, ue_kpms):
        """
        Process KPMs to extract PRB allocation and throughput per UE.

        Args:
            ue_kpms: List of KPM tuples from datalake
        """
        # Reset tracking
        temp_prb = {}
        temp_throughput = {}

        for kpm in ue_kpms:
            if len(kpm) >= 4:
                ue_imsi = kpm[0]
                cell_id = kpm[1]
                prb_used = kpm[2] if kpm[2] is not None else 0
                throughput = kpm[3] if kpm[3] is not None else 0

                # Map IMSI to UE index (0-based)
                ue_idx = (ue_imsi - 1) % self.num_ues if ue_imsi > 0 else 0

                # Accumulate PRBs and throughput
                temp_prb[ue_idx] = temp_prb.get(ue_idx, 0) + prb_used
                temp_throughput[ue_idx] = temp_throughput.get(ue_idx, 0) + throughput

        # Update internal state
        if temp_prb:
            self.prb_allocation = temp_prb
        if temp_throughput:
            self.ue_throughput = temp_throughput

        # Initialize missing UEs with fair allocation
        if not self.prb_allocation:
            prbs_per_ue = self.NUM_PRB_PER_RU // (self.num_ues // self.NUM_RU)
            for ue in range(self.num_ues):
                self.prb_allocation[ue] = prbs_per_ue

    @override
    def _compute_reward(self):
        """
        Compute reward according to Equation 11.

        r_i = (1-delta) * T(t)/T_min + delta * sum(pi_k(t))
              + lambda * sum((R_k(t) - R_min,k) / T_min)

        Returns:
            Float reward value
        """
        self.num_steps += 1

        # Calculate total throughput T(t)
        total_throughput = sum(self.ue_throughput.values())

        # Calculate number of satisfied users sum(pi_k(t))
        num_satisfied = sum(
            1 for ue in range(self.num_ues)
            if self.ue_throughput.get(ue, 0) >= self.R_MIN_MBPS
        )

        # Calculate throughput gap sum((R_k(t) - R_min,k) / T_min)
        throughput_gap = sum(
            (self.ue_throughput.get(ue, 0) - self.R_MIN_MBPS) / self.T_MIN_MBPS
            for ue in range(self.num_ues)
        )

        # Compute reward (Eq. 11)
        reward = (
            (1 - self.DELTA) * (total_throughput / self.T_MIN_MBPS) +
            self.DELTA * num_satisfied +
            self.LAMBDA * throughput_gap
        )

        # Store for logging
        self._log_reward_components(total_throughput, num_satisfied, throughput_gap, reward)

        return reward

    def _log_reward_components(self, throughput, satisfied, gap, reward):
        """Log reward components to datalake for analysis."""
        try:
            db_row = {
                'timestamp': self.last_timestamp,
                'ueImsiComplete': None,
                'step': self.num_steps,
                'total_throughput': float(throughput),
                'num_satisfied': int(satisfied),
                'throughput_gap': float(gap),
                'reward': float(reward),
                'acceptance_rate': float(satisfied / self.num_ues * 100) if self.num_ues > 0 else 0
            }
            self.datalake.insert_data("nguyen_metrics", db_row)
        except Exception:
            pass  # Logging failure should not break the environment

    @override
    def _init_datalake_usecase(self):
        """Initialize datalake tables for Nguyen use case."""
        # Metrics table for tracking performance
        metrics_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "step": "INTEGER",
            "total_throughput": "REAL",
            "num_satisfied": "INTEGER",
            "throughput_gap": "REAL",
            "reward": "REAL",
            "acceptance_rate": "REAL"
        }

        # PRB allocation history
        prb_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "ueId": "INTEGER",
            "prbAllocation": "INTEGER",
            "throughput": "REAL"
        }

        self.datalake._create_table("nguyen_metrics", metrics_keys)
        self.datalake._create_table("nguyen_prb", prb_keys)

        return super()._init_datalake_usecase()

    @override
    def _fill_datalake_usecase(self):
        """Fill datalake with use-case specific data."""
        # Log current PRB allocation
        for ue_id, prbs in self.prb_allocation.items():
            try:
                db_row = {
                    'timestamp': self.last_timestamp,
                    'ueImsiComplete': None,
                    'ueId': ue_id,
                    'prbAllocation': prbs,
                    'throughput': self.ue_throughput.get(ue_id, 0)
                }
                self.datalake.insert_data("nguyen_prb", db_row)
            except Exception:
                pass

    def get_state_for_ru(self, ru_id: int):
        """
        Get state representation for a specific RU.
        Used for multi-agent Q-learning where each RU is an agent.

        Args:
            ru_id: RU identifier (0 or 1)

        Returns:
            Tuple of (prb_allocation, satisfaction_status) for UEs in this RU
        """
        state = []

        for ue in range(self.num_ues):
            if self.ue_to_ru.get(ue, 0) == ru_id:
                prbs = self.prb_allocation.get(ue, 0)
                throughput = self.ue_throughput.get(ue, 0)
                satisfied = 1 if throughput >= self.R_MIN_MBPS else 0
                state.extend([prbs, satisfied])

        return tuple(state)

    def get_valid_actions_for_ru(self, ru_id: int):
        """
        Get valid actions for a specific RU.
        Only allows transfers between UEs in the same RU.

        Args:
            ru_id: RU identifier

        Returns:
            List of valid action indices
        """
        valid_actions = []
        ues_in_ru = [ue for ue in range(self.num_ues) if self.ue_to_ru.get(ue, 0) == ru_id]

        for k_from in ues_in_ru:
            for k_to in ues_in_ru:
                if k_from != k_to and self.prb_allocation.get(k_from, 0) >= self.DELTA_PRB:
                    action = k_from * self.num_ues + k_to
                    valid_actions.append(action)

        return valid_actions

    def get_metrics(self):
        """
        Get current performance metrics for evaluation.

        Returns:
            Dictionary with acceptance_rate, sum_throughput, objective_value
        """
        num_satisfied = sum(
            1 for ue in range(self.num_ues)
            if self.ue_throughput.get(ue, 0) >= self.R_MIN_MBPS
        )

        total_throughput = sum(self.ue_throughput.values())

        # Objective function (Eq. 7)
        objective = (
            (1 - self.DELTA) * (total_throughput / self.T_MIN_MBPS) +
            self.DELTA * num_satisfied
        )

        return {
            'acceptance_rate': num_satisfied / self.num_ues * 100 if self.num_ues > 0 else 0,
            'sum_throughput': total_throughput,
            'num_satisfied': num_satisfied,
            'objective_value': objective
        }
