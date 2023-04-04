"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""
import argparse
import json
import sys

from datetime import datetime

from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env

from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune

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

    # optional input parameters
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=1000,
        help='How many steps are in a training batch.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')

    return parser.parse_known_args(args)[0]

def get_trial_name(exp_tag):

    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    def get_trial_dir(trial):
        return(date_time + "_" + exp_tag + "_" + trial.trial_id)
    return get_trial_dir

def setup_exps_rllib(flow_params,
                     n_cpus,
                     n_rollouts):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_cpus : int
        number of CPUs to run the experiment over
    n_rollouts : int
        number of rollouts per training iteration
    Returns
    -------
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    from ray import tune
    from ray.tune.registry import register_env

    config = PPOConfig()

    # TODO make the training parameters setable via experiment config parameters
    horizon = flow_params['env'].horizon
    batch_size = horizon * n_rollouts
    config.training(gamma=0.999, train_batch_size=batch_size, lambda_=0.97, use_gae=True, kl_target=0.02, num_sgd_iter=10)
    config.model.update({'fcnet_hiddens': [32, 32, 32]})
    config.horizon = horizon

    config.rollouts(num_rollout_workers=n_cpus).resources(num_cpus_per_worker=1)

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config.env_config['flow_params'] = flow_json

    create_env, gym_name = make_create_env(params=flow_params)

    # Register as rllib env
    register_env(gym_name, create_env)
    return gym_name, config

def train_rllib(submodule, flags):
    """Train policies using the PPO algorithm in RLlib."""
    import ray
    from ray.tune import run_experiments

    flow_params = submodule.flow_params
    n_cpus = submodule.N_CPUS
    n_rollouts = submodule.N_ROLLOUTS

    gym_name, config = setup_exps_rllib(flow_params, n_cpus, n_rollouts)

    ray.init(address='auto')
    config.environment(gym_name)
    config.framework("tf2")

    tune.run("PPO",
             config = config.to_dict(),
             checkpoint_freq = 2,
             checkpoint_at_end = True,
             stop = {
                 "training_iteration": flags.num_steps,
             },
             trial_dirname_creator=get_trial_name(flow_params['exp_tag'])
         )

def main(args):
    """Perform the training operations."""
    flags = parse_args(args)

    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[flags.exp_config])

    # Import the sub-module containing the specified exp_config and determine
    # whether the environment is single agent or multi-agent.
    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
    elif hasattr(module_ma, flags.exp_config):
        submodule = getattr(module_ma, flags.exp_config)
    else:
        raise ValueError("Unable to find experiment config.")

    # Perform the training operation.
    train_rllib(submodule, flags)

if __name__ == "__main__":
    main(sys.argv[1:])
