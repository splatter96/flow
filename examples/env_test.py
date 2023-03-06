from flow.utils.registry import make_create_env
import time

from flow.controllers import IDMController, ContinuousRouter, SimCarFollowingController, RLController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
# from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.envs.ring.merge import MergePOEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
# from flow.networks.double_ring import DoubleRingNetwork, ADDITIONAL_NET_PARAMS
from flow.networks.loop_merge import TwoLoopsOneMergingScenario, ADDITIONAL_NET_PARAMS

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

class Experiment:

    def __init__(self):

        # Standard idm cars for the inner circle
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm-inner",
            # acceleration_controller=(IDMController, {'T':0.1}),
            acceleration_controller=(SimCarFollowingController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
                # speed_mode="obey_safe_speed",
                min_gap=0.2,
                max_speed=10,
                ),
            num_vehicles=14)

        # add our rl agent car
        # this needs to be added after the idm cars to spawn on the outer ring
        vehicles.add(
            veh_id="rl",
            # acceleration_controller=(IDMController, {}),
            # acceleration_controller=(SimCarFollowingController, {}),
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
                # speed_mode="obey_safe_speed",
                max_speed=20
                ),
            color='red',
            num_vehicles=1)

        """
        # network = DoubleRingNetwork(
        network = TwoLoopsOneMergingScenario(
                name='ring',
                vehicles=vehicles,
                net_params=NetParams(
                    additional_params=ADDITIONAL_NET_PARAMS.copy(),
                ),
                initial_config=InitialConfig(
                    bunching=20,
                    spacing="random",
                    # x0 = 10,
                    # edges_distribution={"top": 1, "left": 1, "bottom": 5, "right": 5, "right_outer1": 1}
                    edges_distribution={"top": 0, "bottom": 0, "left": 8, "center": 8, "right": 1}
                ),
        )

        self.env = MergePOEnv(
                env_params=EnvParams(
                    horizon=1500,
                    additional_params=ADDITIONAL_ENV_PARAMS,
                ),
                network = network,
                sim_params=SumoParams(
                    # render=True,
                    render=False,
                    sim_step=0.1,
                ),
        )
        """



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
                # render=True,
                render=False,
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
                edges_distribution={"top": 0, "bottom": 0, "left": 7, "center": 7, "right": 1}
            ),
        )

        create_env, gym_name = make_create_env(params=flow_params)
        self.env = create_env()

        register_env("myMergeEnv", create_env)

        ray.init()
        config = PPOConfig().environment(env="myMergeEnv").rollouts(num_rollout_workers=1)
        self.algo = config.build(use_copy=False)

        for _ in range(5):
            print(self.algo.train())


    def run(self, num_runs, rl_actions=None, convert_to_csv=False):
        num_steps = self.env.env_params.horizon

        # time profiling information
        t = time.time()

        for i in range(num_runs):
            ret = 0
            state = self.env.reset()

            for j in range(num_steps):
                state, reward, done, _ = self.env.step(None)

                # print(state)
                # print(all(elem <= 1 and elem >= 0 for elem in state))
                # print(reward)

                ret += reward

                if done:
                    break

            print("Round {0}, return: {1}".format(i, ret))

        print("Total time:", time.time() - t)
        self.env.terminate()

        return

if __name__ == "__main__":
    exp = Experiment()
    # exp.run(num_runs=10)

