from flow.utils.registry import make_create_env
import time

from flow.controllers import IDMController, ContinuousRouter, SimCarFollowingController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS
from flow.networks.double_ring import DoubleRingNetwork, ADDITIONAL_NET_PARAMS


class Experiment:

    def __init__(self):

        # Standard idm cars for the inner circle
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm-inner",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive"
                # speed_mode="obey_safe_speed"
                ),
            num_vehicles=16)

        # add our rl agent car
        # this needs to be added after the idm cars to spawn on the outer ring
        vehicles.add(
            veh_id="rl",
            acceleration_controller=(IDMController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive"
                # speed_mode="obey_safe_speed"
                ),
            num_vehicles=1)

        network = DoubleRingNetwork(
                name='ring',
                vehicles=vehicles,
                net_params=NetParams(
                    additional_params=ADDITIONAL_NET_PARAMS.copy(),
                ),
                initial_config=InitialConfig(
                    bunching=20,
                    spacing="uniform",
                    x0 = 10,
                    edges_distribution={"top": 4, "left": 4, "bottom": 4, "right": 4, "right_outer1": 1}
                ),
        )

        self.env = AccelEnv(
                env_params=EnvParams(
                    horizon=1500,
                    additional_params=ADDITIONAL_ENV_PARAMS,
                ),
                network = network,
                sim_params=SumoParams(
                    render=True,
                    # render=False,
                    sim_step=0.1,
                ),
        )

    def run(self, num_runs, rl_actions=None, convert_to_csv=False):
        num_steps = self.env.env_params.horizon

        # time profiling information
        t = time.time()

        for i in range(num_runs):
            ret = 0
            state = self.env.reset()

            for j in range(num_steps):
                state, reward, done, _ = self.env.step(None)

                ret += reward

                if done:
                    break

            print("Round {0}, return: {1}".format(i, ret))

        print("Total time:", time.time() - t)
        self.env.terminate()

        return

if __name__ == "__main__":
    exp = Experiment()
    exp.run(num_runs=10)

