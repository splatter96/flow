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
            num_vehicles=14)

        # add our rl agent car
        # this needs to be added after the idm cars to spawn on the outer ring
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
                max_speed=20
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
                render=not TRAIN,
                restart_instance=not TRAIN,
            ),

            # environment related parameters (see flow.core.params.EnvParams)
            env=EnvParams(
                horizon=1500,
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
                bunching=20,
                spacing="random",
                edges_distribution={"top": 0, "bottom": 0, "left": 7, "center": 7, "right": 1},
                min_gap=0.5 # add a minimum gap of 0.5m between the spawning vehicles so no erros occur
            ),
        )

        create_env, gym_name = make_create_env(params=flow_params)
        self.env = create_env()

        register_env("myMergeEnv", create_env)

    def run(self, num_runs, rl_actions=None, convert_to_csv=False):

        if TRAIN:
            ray.init(address='auto')
            config = PPOConfig().environment(env="myMergeEnv").rollouts(num_rollout_workers=1).resources(num_cpus_per_worker=7)
            algo = config.build(use_copy=False)

            if USE_GPU:
                config.resources(num_gpus=1)

            for i in range(num_runs):
                print(algo.train())

                if i % CHECKPOINT_FREQ == 0:
                    algo.save()

            algo.save()

        elif EVAL:
            pol = Policy.from_checkpoint("/home/paul/checkpoint_000400/")['default_policy']

            num_steps = self.env.env_params.horizon

            for i in range(num_runs):
                ret = 0
                state = self.env.reset()

                for j in range(num_steps):
                    action = pol.compute_single_action(state)[0]
                    state, reward, done, _ = self.env.step(action)
                    ret += reward
                    if done:
                        break

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
    exp.run(num_runs=50)

