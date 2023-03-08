"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.base import Env
from flow.core import rewards

from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
}


class MergePOEnv(Env):
    """Partially observable merge environment.

    This environment is used to train autonomous vehicles to safely
    merge in an ring merge network.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2

    States
        The observation consists of the locations and speeds of all
        vehicles in the system.

    Actions
        The action space consists of the bounded accelerations for the
        autonomous vehicle. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

    Rewards
        -1 for collision 1 for successfull merge

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))


        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1, ),
            dtype=np.float32)

    @property
    # def observation_space(self):
        # """See class definition."""
        # return Box(low=0, high=1, shape=(5 * 1, ), dtype=np.float32)
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return Box(
            low=0,
            high=1,
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)


    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        # self.leader = []
        # self.follower = []

        # # normalizing constants
        # max_speed = self.k.network.max_speed()
        # max_length = self.k.network.length()

        # observation = [0 for _ in range(5 * 1)]
        # for i, rl_id in enumerate(self.rl_veh):
            # this_speed = self.k.vehicle.get_speed(rl_id)
            # lead_id = self.k.vehicle.get_leader(rl_id)
            # follower = self.k.vehicle.get_follower(rl_id)

            # if lead_id in ["", None]:
                # # in case leader is not visible
                # lead_speed = max_speed
                # lead_head = max_length
            # else:
                # self.leader.append(lead_id)
                # lead_speed = self.k.vehicle.get_speed(lead_id)
                # lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    # - self.k.vehicle.get_x_by_id(rl_id) \
                    # - self.k.vehicle.get_length(rl_id)

            # if follower in ["", None]:
                # # in case follower is not visible
                # follow_speed = 0
                # follow_head = max_length
            # else:
                # self.follower.append(follower)
                # follow_speed = self.k.vehicle.get_speed(follower)
                # follow_head = self.k.vehicle.get_headway(follower)

            # observation[5 * i + 0] = this_speed / max_speed
            # observation[5 * i + 1] = (lead_speed - this_speed) / max_speed
            # observation[5 * i + 2] = lead_head / max_length
            # observation[5 * i + 3] = (this_speed - follow_speed) / max_speed
            # observation[5 * i + 4] = follow_head / max_length

        # return observation

        speed = [max(self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed(), 0.)
                 for veh_id in self.k.vehicle.get_ids()]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.k.vehicle.get_ids()]

        # for veh_id in self.k.vehicle.get_ids():
            # print(self.k.vehicle.get_edge(veh_id))
            # if self.k.vehicle.get_x_by_id(veh_id) < 0:
                # print(self.k.vehicle.get_edge(veh_id))
                # print(self.k.vehicle.get_position(veh_id))
                # print(veh_id)

        return np.array(speed + pos)
        # return np.array(pos)


    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # TODO

        # penalty of -1 in case of collision
        if self.k.simulation.check_collision():
            return -1

        # rl_id = self.k.vehicle.get_rl_ids()[0] #assume single rl agent
        if self.k.vehicle.get_edge("rl_0") == 'left': # agent reached host lane
            return 1

        return 0 

        # if self.env_params.evaluate:
            # return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        # else:
            # # return a reward of 0 if a collision occurred
            # if kwargs["fail"]:
                # return 0

            # # reward high system-level velocities
            # cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

            # # penalize small time headways
            # cost2 = 0
            # t_min = 1  # smallest acceptable time headway
            # for rl_id in self.rl_veh:
                # lead_id = self.k.vehicle.get_leader(rl_id)
                # if lead_id not in ["", None] \
                        # and self.k.vehicle.get_speed(rl_id) > 0:
                    # t_headway = max(
                        # self.k.vehicle.get_headway(rl_id) /
                        # self.k.vehicle.get_speed(rl_id), 0)
                    # cost2 += min((t_headway - t_min) / t_min, 0)

            # # weights for cost1, cost2, and cost3, respectively
            # eta1, eta2 = 1.00, 0.10

            # return max(eta1 * cost1 + eta2 * cost2, 0)

    def compute_dones(self):
        """
        Override default done condition.
        We also want the episode to end when the agent has
        sucessfully merged.
        """

        success = self.k.vehicle.get_edge('rl_0') == 'left'

        return super().compute_dones() or success

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < 1:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []
        return super().reset()
