""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on an OpenAI Gym environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

"""
from typing import Any
from typing import Dict

import os
import sys
import argparse

from flow.utils.registry import make_create_env

import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(5e6)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}

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

def sample_ppo_params(trial: optuna.Trial, submodule) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    # gamma = 1.0 - trial.suggest_float("gamma", 0.001, 0.01, 0.0025)
    gamma = 0.995
    # learning_rate = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
    learning_rate = 5e-5 # use default lr for now
    # net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])

    # horizon = trial.suggest_int("horizon", 500, 700, 50)
    # n_rollouts = trial.suggest_int("n_rollouts", 20, 30, 5)
    horizon = 700
    n_rollouts = 30

    # Display true values.
    # trial.set_user_attr("gamma_", gamma)

    # net_arch = [
        # {"pi": [64, 64], "vf": [64, 64]} if net_arch == "tiny" else {"pi": [64, 64, 64], "vf": [64, 64, 64]}
    # ]

    net_arch = [
        {"pi": [32, 32, 32], "vf": [32, 32, 32]}
    ]

    flow_params = submodule.flow_params
    # n_rollouts = submodule.N_ROLLOUTS

    # horizon = flow_params['env'].horizon
    batch_size = horizon * n_rollouts

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "target_kl": 0.02,
        "gae_lambda": 0.97,
        "policy_kwargs": {
            "net_arch": net_arch,
        },
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial, submodule) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial, submodule))

    create_env, _ = make_create_env(params=submodule.flow_params)
    env = create_env()

    # Create the RL model.
    model = PPO(env=env, tensorboard_log=os.path.expanduser("~/tb3_results/"), verbose=1, **kwargs)
    # model = PPO("MlpPolicy", vec_env, tensorboard_log=os.path.expanduser("~/tb3_results/"), verbose=1)

    # Create env used for evaluation.
    eval_env = create_env()

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, tb_log_name=str(kwargs), log_interval=10, callback=eval_callback)
        # model.learn(total_timesteps=10_000_000, progress_bar=True, tb_log_name=config["exp_tag"], log_interval=1, callback=[checkpoint_callback, params_saver_callback])
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def main(args):
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    flags = parse_args(args)
    module = __import__("exp_configs.rl.singleagent", fromlist=[flags.exp_config])

    if hasattr(module, flags.exp_config):
        submodule = getattr(module, flags.exp_config)
    else:
        raise ValueError("Unable to find experiment config.")

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize", storage="sqlite:///db.sqlite3")
    # study = optuna.load_study(study_name="optuna_flow_test", storage="sqlite:///db_new.sqlite3")
    try:
        # study.optimize(objective, n_trials=N_TRIALS, timeout=600)
        study.optimize(lambda trial: objective(trial, submodule), n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main(sys.argv[1:])
