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
import pickle

from datetime import datetime
from copy import deepcopy

from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env

import ray
from ray import air, tune
from ray.tune import Callback
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class CopyCallback(Callback):
    def __init__(self, config_file):
        super().__init__()
        self.config_file = config_file

    def on_trial_start(self, iteration, trials, trial, **info):
        shutil.copy(self.config_file, f"{trial.logdir}/flow_params.py")


class FlowParamsSaverCallback(BaseCallback):
    def __init__(self, config_file):
        super().__init__()
        self.config_file = config_file

    def _on_training_start(self):
        shutil.copy(self.config_file, f"{self.model.logger.dir}/flow_params.py")

    def _on_step(self):
        pass


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

    # optional parameters
    parser.add_argument(
        '--eval', type=str, default=None,
        help='Directory for the experiment output containing the policy to evaluate')

    parser.add_argument(
        '--num_runs', type=int, default=50,
        help='Number of episodes to perform for evaluation')

    parser.add_argument(
        '--load_state', type=int, default=-1,
        help='Index of initial state to load')

    parser.add_argument( '--render', action='store_true',
        help='Wether to render the the output during evaluation or not')
    parser.add_argument( '--no-render', dest='render', action='store_false',
        help='Wether to render the the output during evaluation or not')
    parser.set_defaults(render=True)

    return parser.parse_known_args(args)[0]
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
    return config, gym_name

def make_env(create_env):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = create_env()
        return env

    return _init

def setup_exps_tb3(submodule):
    flow_params = submodule.flow_params
    n_rollouts = submodule.N_ROLLOUTS
    gamma = submodule.GAMMA

    horizon = flow_params['env'].horizon
    batch_size = horizon * n_rollouts

    config = {}

    config["horizon"] = horizon
    config["batch_size"] = batch_size
    config["gamma"] = gamma
    config["exp_tag"] = flow_params["exp_tag"]

    create_env, _ = make_create_env(params=flow_params)

    return create_env, config

def train_tb3(submodule, flags):
    create_env, config = setup_exps_tb3(submodule)

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
      save_freq=10000,
      save_path=f"~/tb3_results/checkpoints/{config['exp_tag']}",
      name_prefix="rl_model",
      save_replay_buffer=True,
      save_vecnormalize=True,
    )

    # vec_env = SubprocVecEnv([make_env(create_env) for i in range(8)])
    vec_env = create_env()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    params_saver_callback = FlowParamsSaverCallback(f"{dir_path}/exp_configs/rl/singleagent/{flags.exp_config}.py")

    model = PPO("MlpPolicy", vec_env, tensorboard_log="~/tb3_results/", verbose=1)
    model.batch_size = config["batch_size"]
    model.gamma = config["gamma"]
    model.horizon = config["horizon"]
    model.target_kl = 0.02
    model.policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
    model.learn(total_timesteps=10_000_000, progress_bar=True, tb_log_name=config["exp_tag"], log_interval=1, callback=[checkpoint_callback, params_saver_callback])

def train_rllib(submodule, flags):
    """Train policies using the PPO algorithm in RLlib."""

    config, gym_name  = setup_exps_rllib(submodule)

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


def setup_eval_env(params, render):
    # override the rendering relevant render parameters in case the
    # use specified to render the output
    params['sim'].render = render
    params['sim'].restart_instance = render

    create_env, gym_name = make_create_env(params=params)
    register_env("myMergeEnv", create_env)
    env = create_env()

    return env

def eval_policy(env, num_runs, checkpoint, load_state):
    num_steps = env.env_params.horizon

    config = PPOConfig()
    config.environment(env="myMergeEnv")
    config.rollouts(num_rollout_workers=0)
    config.model.update({'fcnet_hiddens': [32, 32, 32]})
    config.explore = False
    config.framework("tf")

    alg = config.build(use_copy=False)
    alg.restore(checkpoint)

    success = 0
    failed = 0
    failed_states = []

    if load_state > -1:
        with open('failed_states.pkl', 'rb') as f:
            failed_states = pickle.load(f)
            num_runs = len(failed_states)

    for i in range(num_runs):
        ret = 0

        if load_state > -1:
            print(f"loading state {failed_states[i]}")
            env.set_initial_state(deepcopy(failed_states[i]))

        state = env.reset()

        for j in range(num_steps):
            action = alg.compute_single_action(state)
            state, reward, done, _ = env.step(action)
            ret += reward

            # print(state)

            if done:
                if reward < 0:
                    failed += 1
                    failed_states.append(deepcopy(env.initial_state))
                    print(f"failed with {env.initial_state}")
                elif reward > 10:
                    success += 1
                break

        print("Round {0}, return: {1} steps: {2}".format(i, ret, j))

    print(f"Success {success} failed {failed}")
    print(f"{len(failed_states)} failed initial states")

    # with open('failed_states.pkl', 'wb') as f:
        # pickle.dump(failed_states, f)


def main(args):
    """Perform the training operations."""
    flags = parse_args(args)

    if flags.eval is None: # training
        # Import relevant information from the exp_config script.
        module = __import__("exp_configs.rl.singleagent", fromlist=[flags.exp_config])

        if hasattr(module, flags.exp_config):
            submodule = getattr(module, flags.exp_config)
        else:
            raise ValueError("Unable to find experiment config.")

        # Perform the training operation.
        # train_rllib(submodule, flags)
        train_tb3(submodule, flags)

    else: # evaluation
        eval_folder = flags.eval
        sys.path.append(os.path.dirname(os.path.expanduser(eval_folder)))

        try:
            import flow_params
        except ModuleNotFoundError:
            print("Can't load config file. Is the path correctly specified? (Add trailing slash?)")
            exit(1)

        env = setup_eval_env(flow_params.flow_params, flags.render)

        # get the newest checkpoint
        folder_content = os.listdir(eval_folder)
        folder_content = list(filter(lambda x: 'checkpoint' in x, folder_content))
        # checkpoint = max(folder_content)
        # checkpoint = "checkpoint_002020"
        checkpoint = "checkpoint_000400"
        print(f"Loading checkpoint {checkpoint}")

        # pol = Policy.from_checkpoint(f"{eval_folder}/{checkpoint}")['default_policy']

        # start the evaluation process
        eval_policy(env, flags.num_runs, f"{eval_folder}/{checkpoint}", flags.load_state)

if __name__ == "__main__":
    main(sys.argv[1:])
