"""Runner script for reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm.

Usage
    python train2.py EXP_CONFIG
"""
import argparse
import json
import sys
import shutil
import os

from datetime import datetime

from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.tune import Callback
from ray.tune.registry import register_env


class CopyCallback(Callback):
    def __init__(self, config_file):
        super().__init__()
        self.config_file = config_file

    def on_trial_start(self, iteration, trials, trial, **info):
        shutil.copy(self.config_file, f"{trial.logdir}/flow_params.py")

def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')

    return parser.parse_known_args(args)[0]

def get_trial_name(exp_tag):

    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    def get_trial_dir(trial):
        return(date_time + "_" + exp_tag + "_" + trial.trial_id)
    return get_trial_dir

def setup_exps_rllib(submodule):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    submodule : module
        the loaded config as an python module
    Returns
    -------
    dict
        training configuration parameters
    """
    config = PPOConfig()

    flow_params = submodule.flow_params
    n_cpus = submodule.N_CPUS
    n_rollouts = submodule.N_ROLLOUTS
    gamma = submodule.GAMMA

    horizon = flow_params['env'].horizon
    batch_size = horizon * n_rollouts
    config.training(gamma=gamma, train_batch_size=batch_size, lambda_=0.97, use_gae=True, kl_target=0.02, num_sgd_iter=10)
    config.model.update({'fcnet_hiddens': [32, 32, 32]})
    config.horizon = horizon

    config.rollouts(num_rollout_workers=n_cpus).resources(num_cpus_per_worker=1)

    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config.env_config['flow_params'] = flow_json

    create_env, gym_name = make_create_env(params=flow_params)

    config.environment(gym_name)

    # Register as rllib env
    register_env(gym_name, create_env)
    return config

def train_rllib(submodule, flags):
    """Train policies using the PPO algorithm in RLlib."""

    config = setup_exps_rllib(submodule)

    ray.init(address='auto')
    config.framework("tf")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    start_callback = CopyCallback(f"{dir_path}/exp_configs/rl/singleagent/{flags.exp_config}.py")

    tune.run("PPO",
             config = config.to_dict(),
             checkpoint_freq = 10,
             checkpoint_at_end = True,
             stop = {
                 "training_iteration": 1000,
             },
             trial_dirname_creator=get_trial_name(submodule.flow_params['exp_tag']),
             callbacks=[start_callback]
         )

def main(args):
    """Perform the training operations."""
    flags = parse_args(args)

    # Import relevant information from the exp_config script.
    module = __import__("exp_configs.rl.singleagent", fromlist=[flags.exp_config])

    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
    else:
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    train_rllib(submodule, flags)

if __name__ == "__main__":
    main(sys.argv[1:])
