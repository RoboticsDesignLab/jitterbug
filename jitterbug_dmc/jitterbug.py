"""A Jitterbug dm_control Reinforcement Learning domain

Copyright 2018 The authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections

import numpy as np

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import rewards
from dm_control.utils import containers
from dm_control.utils import io as resources
from dm_control.mujoco.wrapper.mjbindings import mjlib



# Constants
SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets"""
    return (
        resources.GetResource(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "jitterbug.xml"
        )),
        common.ASSETS
    )


@SUITE.add("benchmarking", "easy")
def move_from_origin(
        time_limit=_DEFAULT_TIME_LIMIT,
        random=None,
        environment_kwargs=None
):
    """Move the Jitterbug away from the origin"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="move_from_origin")
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        **environment_kwargs
    )


@SUITE.add("benchmarking", "easy")
def move_in_direction(
        time_limit=_DEFAULT_TIME_LIMIT,
        random=None,
        environment_kwargs=None
):
    """Move the Jitterbug in a certain direction"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="move_in_direction")
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        **environment_kwargs
    )


@SUITE.add("benchmarking", "hard")
def move_to_position(
        time_limit=_DEFAULT_TIME_LIMIT,
        random=None,
        environment_kwargs=None
):
    """Move the Jitterbug to a certain XYZ position"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="move_to_position")
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        **environment_kwargs
    )


@SUITE.add("benchmarking", "hard")
def move_to_pose(
        time_limit=_DEFAULT_TIME_LIMIT,
        random=None,
        environment_kwargs=None
):
    """Move the Jitterbug to a certain XYZRPY pose"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="move_to_pose")
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features"""

    def jitterbug_position(self):
        """Get the full jitterbug pose vector"""
        return self.named.data.qpos["root"]

    def jitterbug_position_xyz(self):
        """Get the XYZ position of the Jitterbug"""
        return self.jitterbug_position()[:3]

    def jitterbug_position_quat(self):
        """Get the orientation of the Jitterbug"""
        return self.jitterbug_position()[3:]

    def jitterbug_position_yaw(self):
        """Get the yaw angle of the Jitterbug in radians

        Returns:
            (float): Yaw angle of the Jitterbug in radians on the range
                [-pi, pi]
        """
        mat = np.zeros((9))
        mjlib.mju_quat2Mat(mat, self.jitterbug_position_quat())
        mat = mat.reshape((3, 3))
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        return yaw

    def jitterbug_velocity(self):
        """Get the full jitterbug velocity vector"""
        return self.named.data.qvel["root"]

    def jitterbug_velocity_xyz(self):
        """Get the XYZ velocity of the Jitterbug"""
        return self.jitterbug_velocity()[:3]

    def jitterbug_velocity_rpy(self):
        """Get the angular velocity of the Jitterbug"""
        return self.jitterbug_velocity()[3:]

    def target_position(self):
        """Get the full target pose vector"""
        return np.concatenate((
                self.target_position_xyz(),
                self.target_position_quat()
            ),
            axis=0
        )

    def target_position_xyz(self):
        """Get the XYZ position of the target"""
        return self.named.data.geom_xpos["target"]

    def target_position_quat(self):
        """Get the orientation of the target"""
        return self.named.data.xquat["target"]

    def target_position_yaw(self):
        """Get the yaw angle of the target in radians

        Returns:
            (float): Yaw angle of the target in radians on the range
                [-pi, pi]
        """
        mat = np.zeros((9))
        mjlib.mju_quat2Mat(mat, self.target_position_quat())
        mat = mat.reshape((3, 3))
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        return yaw

    def vec_jitterbug_to_target(self):
        """Gets an XYZ vector from jitterbug to the target"""
        return self.target_position_xyz() - self.jitterbug_position_xyz()

    def angle_jitterbug_to_target(self):
        """Gets the relative yaw angle from Jitterbug heading to the target

        Returns:
            (float): The relative angle in radians from the target to the
                Jitterbug on the range [-pi, pi]
        """
        angle = self.target_position_yaw() - self.jitterbug_position_yaw()
        while angle > np.pi:
            angle -= 2*np.pi
        while angle <= -np.pi:
            angle += 2*np.pi
        return angle


class Jitterbug(base.Task):
    """A jitterbug `Task`"""

    def __init__(self, random=None, task="move_from_origin"):
        """Initialize an instance of the `Jitterbug` domain

        Args:
            random (numpy.random.RandomState): Options are;
                - numpy.random.RandomState instance
                - An integer seed for creating a new `RandomState`
                - None to select a seed automatically (default)
            task (str): Specifies which task to configure. Options are;
                - move_from_origin
                - move_in_direction
                - move_to_position
                - move_to_pose
        """

        assert task in [
            "move_from_origin",
            "move_in_direction",
            "move_to_position",
            "move_to_pose"
        ]

        self.task = task
        super(Jitterbug, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode
        """

        # Configure target based on task
        angle = self.random.uniform(0, 2 * np.pi)
        radius = self.random.uniform(.05, 0.4)
        yaw = np.random.uniform(0, 2 * np.pi)

        if self.task == "move_from_origin":

            # Hide the target as it is not needed for this task
            physics.named.model.geom_rgba["target", 3] = 0
            physics.named.model.geom_rgba["targetPointer", 3] = 0

        elif self.task == "move_in_direction":

            # Randomize target orientation
            physics.named.model.body_quat["target"] = np.array([
                np.cos(yaw / 2), 0, 0, 1 * np.sin(yaw / 2)
            ])

        elif self.task == "move_to_position":

            # Hide the target orientation indicator as it is not needed
            physics.named.model.geom_rgba["targetPointer", 3] = 0

            # Randomize target position
            physics.named.model.body_pos["target", "x"] = radius * np.sin(angle)
            physics.named.model.body_pos["target", "y"] = radius * np.cos(angle)

        elif self.task == "move_to_pose":

            # Randomize full target pose
            radius = 0.01
            physics.named.model.body_pos["target", "x"] = radius * np.sin(angle)
            physics.named.model.body_pos["target", "y"] = radius * np.cos(angle)
            physics.named.model.body_quat["target"] = np.array([
                np.cos(yaw / 2), 0, 0, 1 * np.sin(yaw / 2)
            ])

        else:
            raise ValueError("Invalid task {}".format(self.task))

        super(Jitterbug, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state and the target position
        """

        obs = collections.OrderedDict()
        obs['position'] = physics.jitterbug_position()
        obs['velocity'] = physics.jitterbug_velocity()

        if self.task == "move_from_origin":

            # Jitterbug position is a sufficient observation for this task
            pass

        elif self.task == "move_in_direction":

            # Store the relative goal direction vector X, Y components
            target_yaw = physics.angle_jitterbug_to_target()
            obs['target_direction'] = np.array([
                np.cos(target_yaw),
                np.sin(target_yaw)
            ])

        elif self.task == "move_to_position":

            # Store the goal X, Y position
            obs['target_position'] = physics.target_position_xyz()[:2]

        elif self.task == "move_to_pose":

            # Store the goal X, Y position and yaw
            obs['target_position'] = np.concatenate((
                physics.target_position_xyz()[:2],
                [physics.target_position_yaw()]
                ),
                axis=0
            )

        else:
            raise ValueError("Invalid task {}".format(self.task))

        return obs

    def get_reward(self, physics):

        if self.task == "move_from_origin":

            # Reward is bounded by [0, 1] and grows larger as you move away
            # from the origin, asymptoting to 1 at infinite distance
            # https://www.desmos.com/calculator/021e5m4cwr
            dist_from_origin = np.linalg.norm(
                physics.jitterbug_position_xyz()[:2]
            )
            return 1 - 1 / (20 * dist_from_origin + 1)

        elif self.task == "move_in_direction":

            raise NotImplementedError("Not yet implemented")

        elif self.task == "move_to_position":

            # Reward is bounded by [0, 1] and grows larger as you approach the
            # target, asymptoting to 0 as infinite distance
            # https://www.desmos.com/calculator/r1ijdhubd3
            dist_to_target = np.linalg.norm(physics.vec_jitterbug_to_target())
            return 1 / (5 * dist_to_target + 1)

        elif self.task == "move_to_pose":

            dist_to_target = np.linalg.norm(physics.vec_jitterbug_to_target())
            angle_to_target = physics.angle_jitterbug_to_target()

            # Distance is rewarded on [0, 1] as you approach the target
            # https://www.desmos.com/calculator/r1ijdhubd3
            dist_reward = 1 / (5 * dist_to_target + 1)

            # Angle is rewarded on [0, 1] as you face the target direction
            # https://www.desmos.com/calculator/iaczzkaplq

            angle_to_target = physics.angle_jitterbug_to_target()
            angle_reward = 2 / (np.abs(angle_to_target) / np.pi + 1) - 1

            # Combine the two rewards
            #return dist_reward * angle_reward
            return 0.5 * (dist_reward + angle_reward)

        else:
            raise ValueError("Invalid task {}".format(self.task))


def demo():
    """Demonstrate the Jitterbug domain"""

    # Get some imports
    from dm_control import suite
    from dm_control import viewer

    # Add the jitterbug tasks to the suite
    import jitterbug_dmc

    env = suite.load(
        domain_name="jitterbug",
        task_name="move_to_pose",
        visualize_reward=True
    )
    policy = lambda ts: 0.8

    # Run the viewer with a constant policy
    viewer.launch(env, policy=policy)


if __name__ == '__main__':
    demo()
