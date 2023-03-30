"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.base import Env
from flow.core import rewards

from gym.spaces.box import Box

from libsumo import INVALID_DOUBLE_VALUE

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 4,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 4,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': True
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


        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        self.prev_rl_dist = 0

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
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return Box(
            low=-1,
            high=1,
            # shape=(2 * self.initial_vehicles.num_vehicles, ),
            shape=(2 * 5, ),
            dtype=np.float32)


    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # for i, rl_id in enumerate(self.rl_veh):
            # # ignore rl vehicles outside the network
            # if rl_id not in self.k.vehicle.get_rl_ids():
                # continue
            # self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])

        rl_id = self.k.vehicle.get_rl_ids()[0] #assume single rl agent
        self.k.vehicle.apply_acceleration(rl_id, rl_actions)

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        """
        #old approach
        speed = [max(self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed(), 0.)
                 for veh_id in self.sorted_ids]
        pos = [(max(self.k.vehicle.get_driving_distance(veh_id, "left", 200) - 200, -200)) / 200
                for veh_id in self.sorted_ids]

        rl_id = self.k.vehicle.get_rl_ids()[0] #assume single rl agent
        speed.append(max(self.k.vehicle.get_speed(rl_id) / self.k.network.max_speed(), 0.))
        pos.append(max(self.k.vehicle.get_driving_distance(rl_id, "left", 200) - 200, -200) / 200)
        """

        # new approach
        left_ids = list(filter(lambda x: (self._get_abs_position(x) <= 0), self.sorted_ids))
        right_ids = list(filter(lambda x: (self._get_abs_position(x) > 0), self.sorted_ids))

        if len(left_ids) < 2: # not enough vehicles on left half
            # ids = left_ids + right_ids[-(2-len(left_ids)):] + right_ids[0:2]
            left_ids = ['placeholder' for i in range(2-len(left_ids))] + left_ids

        if len(right_ids) < 2: # not enough vehicles in right half
            # ids = left_ids[-2:] + right_ids + left_ids[:2-len(right_ids)]
            right_ids = right_ids + ['placeholder' for i in range(2-len(right_ids))]

        left_ids = left_ids[-2:]
        right_ids = right_ids[:2]

        pos = []

        for veh_id in left_ids:
            veh_pos = self.k.vehicle.get_driving_distance(veh_id, "left", 200)
            if veh_pos == INVALID_DOUBLE_VALUE:
                pos.append(-1)
            else:
                pos.append((veh_pos - 200)/200)

        for veh_id in right_ids:
            veh_pos = self.k.vehicle.get_driving_distance(veh_id, "left", 200)
            if veh_pos == INVALID_DOUBLE_VALUE:
                pos.append(1)
            else:
                pos.append((veh_pos - 200)/200)

        rl_id = self.k.vehicle.get_rl_ids()[0] #assume single rl agent

        pos.append((self.k.vehicle.get_driving_distance(rl_id, "left", 200) - 200) / 200)
        print(pos)

        speed = [max(self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed(), 0.)
                 for veh_id in (*left_ids, *right_ids,  rl_id)]
        print(speed)

        return np.array(speed + pos)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        reward = 0

        # penalty of -10 in case of collision
        if self.k.simulation.check_collision():
            reward += -10
            return reward

        # reward for successfull merge
        if self._merge_success():
            reward += 10

        rl_id = self.k.vehicle.get_rl_ids()[0] #assume single rl agent
        dist = self.k.vehicle.get_distance(rl_id)

        if dist > 0: #avaid error case
            reward += 0.05 * (dist - self.prev_rl_dist)
            self.prev_rl_dist = dist

        return reward

    def compute_dones(self):
        """
        Override default done condition.
        We also want the episode to end when the agent has
        sucessfully merged.
        """
        return super().compute_dones() or self._merge_success()

    def _merge_success(self):
        rl_id = self.k.vehicle.get_rl_ids()[0] #assume single rl agent
        agent_pos = self.k.vehicle.get_x_by_id(rl_id)

        return agent_pos > 230 and agent_pos < 300

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            # return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
            return sorted(self.k.vehicle.get_human_ids(), key=self._get_abs_position)
        else:
            # return self.k.vehicle.get_ids()
            return self.k.vehicle.get_human_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        # return self.absolute_position.get(veh_id, -1001)
        return self.k.vehicle.get_driving_distance(veh_id, 'left', 200) - 200

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        self.prev_rl_dist = 0

        return super().reset()
