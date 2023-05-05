import json
import argparse

import ray
from ray import air, tune

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import ray.rllib.algorithms.ppo as ppo

from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env, make_create_env_rllib

def setup_exps_rllib(params):
    create_env, gym_name = make_create_env_rllib(params=params)
    # Register as rllib env
    register_env("myMergeEnv", create_env)
    return params, gym_name

def objective(config): 
    #convert the dict back to the actual python objects
    flow_config = get_flow_params(config)
    config["env_config"]["flow_params"] = flow_config

    algo = ppo.PPO(config, env="myMergeEnv")

    while True:
        train_res = algo.train()
        tune.report(**train_res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent')
    args, _ = parser.parse_known_args()

    # Load the matching flow config
    module = __import__("exp_configs.rl.singleagent", fromlist=[args.exp_config])

    if hasattr(module, args.exp_config):
        submodule = getattr(module, args.exp_config)
    else:
        raise ValueError("Unable to find experiment config.")

    flow_config, gym_name  = setup_exps_rllib(submodule.flow_params)

    ray.init(address='auto')

    # Configure the algorithm
    batch_size = 200
    config = (
            PPOConfig()
            .environment("myMergeEnv")
            .rollouts(num_rollout_workers=3)
            .resources(num_cpus_per_worker=1)
            .training(
                gamma=0.999,
                train_batch_size=batch_size,
                lambda_=0.97,
                use_gae=True,
                kl_target=0.02,
                num_sgd_iter=10,
                model={'fcnet_hiddens': [32, 32, 32]}
            )
    )

    config = config.to_dict()

    #convert the the flow parameters to a dict to be able to search through them
    flow_json = json.dumps(flow_config, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    flow_dict = json.loads(flow_json)
    config['env_config']['flow_params'] = flow_dict

    ######
    # Define our search space
    ######
    config["env_config"]['flow_params']["sim"]["sim_step"] = tune.grid_search([0.1, 0.2])
    config["env_config"]['flow_params']["veh"][0]["num_vehicles"] = tune.grid_search([10, 12])

    # Stop when we've either reached 1000 training iterations or reward=15
    stopping_criteria = {"training_iteration": 10, "episode_reward_mean": 15}

    tuner = tune.Tuner(
        tune.with_resources(objective, ppo.PPO.default_resource_request(config)),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
        ),
        param_space=config,
        run_config=air.RunConfig(
            stop=stopping_criteria,
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end=True),
            ),
    )
    results = tuner.fit()
