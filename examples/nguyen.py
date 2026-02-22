"""
Nguyen Environment - Example Script

Simple example script for running the Nguyen PRB allocation environment.
Based on the paper: "Resource Allocation for Open Radio Access Networks
Using Reinforcement Learning" (Nguyen et al., ATC 2025)

This script demonstrates:
- How to create and configure the environment
- How to interact with the environment (reset, step)
- How to collect observations and rewards

For Q-learning implementation, see nguyen_qlearning.py
"""

import argparse
import json
from environments.nguyen_env import NguyenEnv


if __name__ == '__main__':
    #######################
    # Parse arguments
    #######################
    parser = argparse.ArgumentParser(description="Run the Nguyen PRB allocation environment")
    parser.add_argument("--config", type=str,
                        default="/home/eliothluy/Documentos/artigoJussi/ns-o-ran-gym/src/environments/scenario_configurations/nguyen_use_case.json",
                        help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str,
                        default="/home/eliothluy/Documentos/artigoJussi/ns-3-mmwave-oran/",
                        help="Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of steps to run in the environment")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")

    args = parser.parse_args()

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    num_steps = args.num_steps
    optimized = args.optimized

    # Load configuration
    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, using default configuration")
        params = json.dumps({
            "simTime": [10],
            "ues": [15],
            "RngRun": [1],
            "indicationPeriodicity": [0.100],
            "useSemaphores": [1],
            "embbDataRate": [15.0],
            "controlFileName": ["nguyen_actions.csv"]
        })

    scenario_configuration = json.loads(params)

    print('='*60)
    print('Nguyen PRB Allocation Environment')
    print('='*60)
    print(f'Configuration: {configuration_path}')
    print(f'UEs: {scenario_configuration.get("ues", [15])[0]}')
    print(f'Simulation Time: {scenario_configuration.get("simTime", [10])[0]}s')
    print('='*60)

    print('\nCreating Nguyen Environment')
    env = NguyenEnv(
        ns3_path=ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=output_folder,
        optimized=optimized
    )
    print('Environment Created!')

    print('\nLaunching reset... ', end='', flush=True)
    obs, info = env.reset()
    print('done')

    print(f'\nInitial observations: {obs}')
    print(f'Info: {info}')

    # Get number of UEs for action generation
    num_ues = scenario_configuration.get("ues", [15])[0]

    # Main loop
    for step in range(2, num_steps):
        # Simple heuristic action: no transfer (k_from = k_to = 0)
        # This is a no-op action that maintains current allocation
        # Replace this with your own policy/algorithm
        model_action = (0, 0)  # No PRB transfer

        # Alternative: random action
        # import random
        # k_from = random.randint(0, num_ues - 1)
        # k_to = random.randint(0, num_ues - 1)
        # model_action = (k_from, k_to)

        print(f'\nStep {step} ', end='', flush=True)
        obs, reward, terminated, truncated, info = env.step(model_action)
        print('done')

        # Get performance metrics
        metrics = env.get_metrics()

        print(f'\n--- Status t = {step} ---')
        print(f'Action: {env._compute_action(model_action)}')
        print(f'Observations: {obs}')
        print(f'Reward: {reward:.4f}')
        print(f'Terminated: {terminated}')
        print(f'Truncated: {truncated}')
        print(f'Metrics:')
        print(f'  - Acceptance Rate: {metrics["acceptance_rate"]:.1f}%')
        print(f'  - Sum Throughput: {metrics["sum_throughput"]:.2f} Mbps')
        print(f'  - Satisfied UEs: {metrics["num_satisfied"]}/{num_ues}')
        print(f'  - Objective Value: {metrics["objective_value"]:.4f}')

        # If the environment is over, exit
        if terminated:
            print('\nSimulation terminated.')
            break

        # If the episode is up (environment still running), then start another one
        if truncated:
            print('\nEpisode truncated.')
            break

    print('\n' + '='*60)
    print('Simulation Complete')
    print('='*60)
