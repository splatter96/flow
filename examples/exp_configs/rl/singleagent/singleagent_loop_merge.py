"""
Merge example.

Trains a single autonomous vehicle to merge successfully into a ring network
"""
from flow.controllers import IDMController, ContinuousRouter, SimCarFollowingController, RLController, IDMController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.ring.merge import MergePOEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.loop_merge import TwoLoopsOneMergingScenario, ADDITIONAL_NET_PARAMS

# number of parallel workers
N_CPUS = 7

# time horizon of a single rollout
HORIZON = 1000
# number of rollouts per training iteration
N_ROLLOUTS = 20

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id="idm-inner",
    # acceleration_controller=(IDMController, {}),
    acceleration_controller=(SimCarFollowingController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="aggressive",
        min_gap=0.2,
        max_speed=10,
        # tau=0.1,
        ),
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
        # render= not TRAIN,
        # restart_instance= not TRAIN,
        render=False,
        restart_instance=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params= {
            # maximum acceleration for autonomous vehicles, in m/s^2
            "max_accel": 8,
            # maximum deceleration for autonomous vehicles, in m/s^2
            "max_decel": 8,
            # specifies whether vehicles are to be sorted by position during a
            # simulation step. If set to True, the environment parameter
            # self.sorted_ids will return a list of all vehicles sorted in accordance
            # with the environment
            'sort_vehicles': True,

            # if this param is true then only a partial observable environment is used
            # aka only 5 vehicles are in the state space
            'PO_env': True,

            #position at which the agents needs to be to consider a merge a successs
            'success_pos': 250
        }
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
            additional_params= {
                # radius of the loops
                "ring_radius": 50,
                # length of the straight edges connected the outer loop to the inner loop
                "lane_length": 50,
                # number of lanes in the inner loop
                "inner_lanes": 1,
                # number of lanes in the outer loop
                "outer_lanes": 1,
                # max speed limit in the network
                "speed_limit": 50,
                # resolution of the curved portions
                "resolution": 40,
                # startposition of the agent in amount of the straight lane
                "agent_start_pos": 0.9
            },
         ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        bunching=0,
        spacing="custom",
        edges_distribution={"top": 0, "bottom": 0, "left": 5, "center": 5, "right": 0},
        min_gap=0.5, # add a minimum gap of 0.5m between the spawning vehicles so no erros occur
        perturbation=20,
        shuffle = True,
    ),
)
