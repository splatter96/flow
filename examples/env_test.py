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

        # Add 22 normal vehicles to our network
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="idm",
            acceleration_controller=(IDMController, {}),
            # acceleration_controller=(SimCarFollowingController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(speed_mode="aggressive"),
            num_vehicles=5)

        network = DoubleRingNetwork(
                name='ring',
                vehicles=vehicles,
                net_params=NetParams(
                    additional_params=ADDITIONAL_NET_PARAMS.copy(),
                ),
                initial_config=InitialConfig(
                    bunching=20,
                    spacing="uniform"
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

