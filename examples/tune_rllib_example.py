import random
import os
import shutil

import ray
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining

from ray.tune import Callback
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy

from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env, make_create_env_rllib

class CopyCallback(Callback):
    def __init__(self, config_file):
        super().__init__()
        self.config_file = config_file

    def on_trial_start(self, iteration, trials, trial, **info):
        shutil.copy(self.config_file, f"{trial.logdir}/flow_params.py")

def setup_exps_rllib(submodule):
    flow_params = submodule.flow_params

    create_env, gym_name = make_create_env_rllib(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return flow_params, gym_name

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent')
    args, _ = parser.parse_known_args()

    module = __import__("exp_configs.rl.singleagent", fromlist=[args.exp_config])

    if hasattr(module, args.exp_config):
        submodule = getattr(module, args.exp_config)
    else:
        raise ValueError("Unable to find experiment config.")

    # Stop when we've either reached 1000 training iterations or reward=15
    stopping_criteria = {"training_iteration": 1000, "episode_reward_mean": 15}

    config, gym_name  = setup_exps_rllib(submodule)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    start_callback = CopyCallback(f"{dir_path}/exp_configs/rl/singleagent/{args.exp_config}.py")

    ray.init(address='auto')

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            # num_samples=2,
        ),
        param_space={
            "env": gym_name,
            "env_config": {"flow_params": config},
            "kl_target": 0.02,
            "num_workers": 3,
            "num_cpus": 1,  # number of CPUs to use per trial
            "num_gpus": 0,  # number of GPUs to use per trial
            "model": {"free_log_std": True},
            # These params are tuned from a fixed starting value.
            "framework": "tf",
            "lambda": 0.97,
            "use_gae": True,
            # "lr": 1e-4,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": 10,
            # "sgd_minibatch_size": tune.grid_search([128, 512, 2048]),
            "train_batch_size": tune.grid_search([15000, 18000, 21000]),
        },
        run_config=air.RunConfig(
            stop=stopping_criteria,
            callbacks=[start_callback],
            checkpoint_config= air.CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end=True),
            ),
    )
    results = tuner.fit()
