from flow.utils.registry import make_create_env
import time

from flow.controllers import IDMController, ContinuousRouter, SimCarFollowingController, RLController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.ring.merge import MergePOEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.loop_merge import TwoLoopsOneMergingScenario, ADDITIONAL_NET_PARAMS

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

######
# META CONFIG
#####
TRAIN = False
EVAL = False
USE_GPU = False
CHECKPOINT_FREQ = 10

# time horizon of a single rollout
HORIZON = 1000
# number of rollouts per training iteration
N_ROLLOUTS = 20

class Experiment:

    def __init__(self):

        # Standard idm cars for the inner circle
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm-inner",
            acceleration_controller=(SimCarFollowingController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
                min_gap=0.2,
                max_speed=10,
                ),
            # num_vehicles=14)
            num_vehicles=10)

        # add our rl agent car
        # this needs to be added after the idm cars to spawn on the outer ring
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
                max_speed=50
                ),
            color='red',
            num_vehicles=1)

        flow_params = dict(
            # name of the experiment
            exp_tag="merge_ring",

            # name of the flow environment the experiment is running on
            env_name=MergePOEnv,

            # name of the network class the experiment is running on
            network=TwoLoopsOneMergingScenario,

            # simulator that is used by the experiment
            simulator='traci',

            # sumo-related parameters (see flow.core.params.SumoParams)
            sim=SumoParams(
                sim_step=0.1,
                render= not TRAIN,
                # render= False,
                restart_instance= not TRAIN,
                # restart_instance= False,
            ),

            # environment related parameters (see flow.core.params.EnvParams)
            env=EnvParams(
                horizon=HORIZON,
                additional_params=ADDITIONAL_ENV_PARAMS
            ),

            # network-related parameters (see flow.core.params.NetParams and the
            # network's documentation or ADDITIONAL_NET_PARAMS component)
            net=NetParams(
                    additional_params=ADDITIONAL_NET_PARAMS.copy(),
                 ),

            # vehicles to be placed in the network at the start of a rollout (see
            # flow.core.params.VehicleParams)
            veh=vehicles,

            # parameters specifying the positioning of vehicles upon initialization/
            # reset (see flow.core.params.InitialConfig)
            initial=InitialConfig(
                bunching=0,
                spacing="custom",
                # edges_distribution={"top": 0, "bottom": 0, "left": 7, "center": 7, "right": 0},
                edges_distribution={"top": 0, "bottom": 0, "left": 5, "center": 5, "right": 0},
                min_gap=0.5, # add a minimum gap of 0.5m between the spawning vehicles so no erros occur
                perturbation=20,
                shuffle = True,
            ),
        )

        create_env, gym_name = make_create_env(params=flow_params)
        self.env = create_env()

        register_env("myMergeEnv", create_env)

    def run(self, num_runs, rl_actions=None, convert_to_csv=False):

        if TRAIN:
            ray.init(address='auto')
            config = PPOConfig().environment(env="myMergeEnv").rollouts(num_rollout_workers=7).resources(num_cpus_per_worker=1)
            # config.exploration(explore=True, exploration_config={
                                                # "type": "EpsilonGreedy",
                                                # "initial_epsilon": 1.0,
                                                # "final_epsilon": 0.02,
                                            # })
            batch_size = HORIZON * N_ROLLOUTS
            config.training(gamma=0.999, train_batch_size=batch_size, lambda_=0.97, use_gae=True, kl_target=0.02, num_sgd_iter=10)
            config.model.update({'fcnet_hiddens': [32, 32, 32]})
            config.horizon = HORIZON

            if USE_GPU:
                config.resources(num_gpus=1)

            algo = config.build(use_copy=False)

            for i in range(num_runs):
                print(algo.train())

                if i % CHECKPOINT_FREQ == 0:
                    algo.save()

            algo.save()

        elif EVAL:
            config = PPOConfig().environment(env="myMergeEnv").rollouts(num_rollout_workers=0)

            batch_size = HORIZON * N_ROLLOUTS
            config.training(gamma=0.999, train_batch_size=batch_size, lambda_=0.97, use_gae=True, kl_target=0.02, num_sgd_iter=10)
            config.model.update({'fcnet_hiddens': [32, 32, 32]})
            config.horizon = HORIZON

            alg = config.build(use_copy=False)
            # alg.restore("/home/paul/checkpoint_000400")
            alg.restore("/home/paul/checkpoint_000800")

            # pol = Policy.from_checkpoint("/home/paul/checkpoint_000171/")['default_policy']

            num_steps = self.env.env_params.horizon
            print(f"{num_steps=}")

            for i in range(num_runs):
                ret = 0
                state = self.env.reset()

                for j in range(num_steps):
                    action = alg.compute_single_action(state)
                    state, reward, done, _ = self.env.step(action)
                    ret += reward

                    if done:
                        break

                print(f"Run took {j} steps")
                print("Round {0}, return: {1}".format(i, ret))

            self.env.terminate()

        else:
            num_steps = self.env.env_params.horizon

            for i in range(num_runs):
                ret = 0
                state = self.env.reset()

                start = time.time()
                for j in range(num_steps):
                    state, reward, done, _ = self.env.step(None)

                    ret += reward
                    if done:
                        break

                end = time.time()
                print(f"Episode took {end-start} with {j} steps")
                print("Round {0}, return: {1}".format(i, ret))

            self.env.terminate()

if __name__ == "__main__":
    exp = Experiment()
    exp.run(num_runs=10)

