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
import sys
import inspect
import collections

# Uncomment to disable GPU training in tensorflow (must be before keras imports)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import rewards
from dm_control.utils import containers
from dm_control.utils import io as resources
from dm_control.mujoco.wrapper.mjbindings import mjlib

import autoencoder
import pickle

# Load the suite so we can add to it
SUITE = containers.TaggedTasks()

# Task constants
DEFAULT_TIME_LIMIT = 10
DEFAULT_CONTROL_TIMESTEP = 0.01
TARGET_SPEED = 0.1


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
        time_limit=DEFAULT_TIME_LIMIT,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
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
        control_timestep=control_timestep,
        **environment_kwargs
    )


@SUITE.add("benchmarking", "easy")
def face_direction(
        time_limit=DEFAULT_TIME_LIMIT,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
        random=None,
        environment_kwargs=None
):
    """Move the Jitterbug to face a certain yaw angle"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="face_direction")
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=control_timestep,
        **environment_kwargs
    )


@SUITE.add("benchmarking", "easy")
def move_in_direction(
        time_limit=DEFAULT_TIME_LIMIT,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
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
        control_timestep=control_timestep,
        **environment_kwargs
    )


@SUITE.add("benchmarking", "hard")
def move_to_position(
        time_limit=DEFAULT_TIME_LIMIT,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
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
        control_timestep=control_timestep,
        **environment_kwargs
    )


@SUITE.add("benchmarking", "hard")
def move_to_pose(
        time_limit=DEFAULT_TIME_LIMIT,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
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
        control_timestep=control_timestep,
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

    def jitterbug_direction_yaw(self):
        """Get the yaw angle of the Jitterbug in radians

        Returns:
            (float): Yaw angle of the Jitterbug in radians on the range
                [-pi, pi]
        """
        mat = np.zeros(9)
        mjlib.mju_quat2Mat(mat, self.jitterbug_position_quat())
        mat = mat.reshape((3, 3))
        yaw = np.arctan2(mat[1, 0], mat[0, 0])

        # Jitterbug model faces the -Y direction, so we rotate 90deg CW to
        # align its face with the +X axis
        yaw -= np.pi / 2

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

    def motor_position(self):
        """Get the motor angular position

        NB: This function artificially rotates the motor frame so that 0deg is
        facing forwards on the Jitterbug

        Returns:
            (float): The motor position in radians on the range [-pi, pi]
        """

        # Offset motor so 0deg is facing forwards on the jitterbug
        angle = self.named.data.qpos["jointMass"] + np.pi / 2

        while angle > np.pi:
            angle -= 2 * np.pi
        while angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    def motor_velocity(self):
        """Get the motor angular velocity"""
        return self.named.data.qvel["jointMass"]

    def target_position(self):
        """Get the full target pose vector"""
        return np.concatenate((
            self.target_position_xyz(),
            self.target_position_quat()#data gathering while learning the task move_in_direction with ddpg
        ),
            axis=0
        )

    def target_position_xyz(self):
        """Get the XYZ position of the target"""
        return self.named.data.geom_xpos["target"]

    def target_position_quat(self):
        """Get the orientation of the target"""
        return self.named.data.xquat["target"]

    def target_direction_yaw(self):
        """Get the yaw angle of the target in radians

        Returns:
            (float): Yaw angle of the target in radians on the range
                [-pi, pi]
        """
        mat = np.zeros(9)
        mjlib.mju_quat2Mat(mat, self.target_position_quat())
        mat = mat.reshape((3, 3))
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        return yaw

    def target_position_in_jitterbug_frame(self):
        """Find XYZ position of the target in the Jitterbug frame

        NB: +X in Jitterbug frame is to the LHS, +Y in Jitterbug frame is
        backwards
        """

        # Find relative target position
        target_pos = self.target_position_xyz() - self.jitterbug_position_xyz()

        # Get the Jitterbug frame rotation matrix
        jitterbug_rot_mat = np.zeros(9)
        mjlib.mju_quat2Mat(jitterbug_rot_mat, self.jitterbug_position_quat())

        # Apply inverse transform to put target in JB frame
        return np.linalg.inv(jitterbug_rot_mat.reshape((3, 3))) @ target_pos

    def jitterbug_velocity_in_target_frame(self):
        """Find the XYZ velocity of the Jitterbug in the target frame"""

        # Get the Jitterbug global frame velocity
        jitterbug_vel = self.named.data.sensordata['jitterbug_framelinvel']

        # Get the target frame rotation matrix
        target_rot_mat = np.zeros(9)
        mjlib.mju_quat2Mat(target_rot_mat, self.target_position_quat())

        # Apply inverse rotation to put velocity in target frame
        return np.linalg.inv(target_rot_mat.reshape((3, 3))) @ jitterbug_vel

    def angle_jitterbug_to_target(self):
        """Gets the relative yaw angle from Jitterbug heading to the target

        Returns:
            (float): The relative angle in radians from the target to the
                Jitterbug on the range [-pi, pi]
        """
        angle = self.target_direction_yaw() - self.jitterbug_direction_yaw()
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle <= -np.pi:
            angle += 2 * np.pi
        return np.array([angle])


class Jitterbug(base.Task):
    """A jitterbug `Task`"""

    def __init__(
            self,
            random=None,
            task="move_from_origin",
            random_pose=True,
    ):
        """Initialize an instance of the `Jitterbug` domain

        Args:
            random (numpy.random.RandomState): Options are;
                - numpy.random.RandomState instance
                - An integer seed for creating a new `RandomState`
                - None to select a seed automatically (default)
            task (str): Specifies which task to configure. Options are;
                - move_from_origin
                - face_direction
                - move_in_direction
                - move_to_position
                - move_to_pose
            random_pose (bool): If true, initialize the Jitterbug with a random
                pose to break symmetries
        """

        # Reflect to get task names from the current module
        self.task_names = [
            obj[0]
            for obj in inspect.getmembers(sys.modules[__name__])
            if inspect.isfunction(obj[1]) and obj[0] in SUITE._tasks
        ]
        assert task in self.task_names, \
            "Invalid task {}, options are {}".format(task, self.task_names)

        self.task = task
        self.random_pose = random_pose
        super(Jitterbug, self).__init__(random=random)

        #self.pickleFile = open("observations.pkl", "wb")
        self.principalVectors = np.array([[0.0049, 0.0171, -0.0001, -0.0001, 0.0003, 0, 0, 0],
                                          [0.0242, -0.0042, -0.0002, 0.0001, 0.0001, 0, 0, 0],
                                          [-0.0002, -0.0001, 0.0003, 0, 0, 0, 0, 0],
                                          [0.1224, 0.9907, 0.0519, -0.0072, 0.0094, -0.0001, 0.0001, 0.0001],
                                          [-0.0019, 0.0201, 0.0014, -0.0006, 0.0003, 0, -0.0002, 0],
                                          [0.0224, -0.0022, -0.0009, 0, -0.0001, -0.0003, 0, 0],
                                          [0.9918, -0.1215, -0.0179, 0.0052, 0.0022, 0, -0.0001, -0.0002],
                                          [0.0014, 0.0043, 0.0001, 0.0003, 0.0001, 0.0002, 0.0001, 0],
                                          [0.0066, -0.0011, 0.0001, -0.0002, 0, 0.0001, -0.0001, 0],
                                          [0.0001, 0, -0.0002, 0.0001, -0.0001, 0.0012, 0, 0],
                                          [-0.0002, 0, 0.0004, 0.0284, 0.0096, -0.9995, -0.0035, -0.0003],
                                          [-0.0003, 0.0004, -0.0014, 0.0766, 0.0141, -0.0011, 0.997, 0.0007],
                                          [0.0038, 0.0067, 0.0441, 0, -0.9988, -0.0096, 0.0142, -0.0046],
                                          [0.0043, -0.0079, 0.0022, -0.9966, 0.0014, -0.0286, 0.0765, 0],
                                          [0.0002, -0.0004, 0.0057, -0.0001, -0.0044, -0.0003, -0.0007, 1],
                                          [-0.0113, 0.054, -0.9975, -0.0027, -0.0438, -0.0009, -0.0006, 0.0055]])

        self.principalVectors4dim = np.array([[0.0003, 0, 0, 0],
                                              [0.0001, 0, 0, 0],
                                              [0, 0, 0, 0],
                                              [0.0094, -0.0001, 0.0001, 0.0001],
                                              [0.0003, 0, -0.0002, 0],
                                              [-0.0001, -0.0003, 0, 0],
                                              [0.0022, 0, -0.0001, -0.0002],
                                              [0.0001, 0.0002, 0.0001, 0],
                                              [0, 0.0001, -0.0001, 0],
                                              [-0.0001, 0.0012, 0, 0],
                                              [0.0096, -0.9995, -0.0035, -0.0003],
                                              [0.0141, -0.0011, 0.997, 0.0007],
                                              [-0.9988, -0.0096, 0.0142, -0.0046],
                                              [0.0014, -0.0286, 0.0765, 0],
                                              [-0.0044, -0.0003, -0.0007, 1],
                                              [-0.0438, -0.0009, -0.0006, 0.0055]])

        self.use_autoencoder = False
        self.use_several_autoencoders = False

        if self.use_autoencoder:
            sess = tf.get_default_session()
            if sess == None:
                sess = tf.Session()
            self.session = sess
            self.jitterbug_autoencoder = autoencoder.Autoencoder(feature_dimension=16,
                                                                 lr=0.0005,
                                                                 sess=self.session
                                                                 )
            i=19
            self.jitterbug_autoencoder.load_autoencoder(f"./autoencoder_model{i}.ckpt")
            print(f"load autoencoder {i} from file ./autoencoder_model{i}.ckpt")


        if self.use_several_autoencoders:
            self.index_list = [11,15,16,17,18]
            self.num_autoencoders = len(self.index_list)
            self.autoencoder_list = []
            self.session_list = []
            for i in self.index_list:
                #Generate a session for each autoencoder
                g_i = tf.Graph()
                with g_i.as_default():
                    session_i = tf.Session(graph=g_i)
                    self.session_list.append(session_i)
                    jitterbug_autoencoder_i = autoencoder.Autoencoder(feature_dimension=16,
                                                                      lr=0.0005,
                                                                      sess=session_i
                                                                      )
                    print(f"load autoencoder {i} from file ./autoencoder_model{i}.ckpt")
                    jitterbug_autoencoder_i.load_autoencoder(f"./autoencoder_model{i}.ckpt")
                    self.autoencoder_list.append(jitterbug_autoencoder_i)


        self.extremum = np.array([[float('Inf'),-float('Inf')]]*16)
        self.N_features =len(self.extremum)




    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode
        """

        # Use reset context to ensure changes are applied immediately
        with physics.reset_context():

            # Configure target based on task
            angle = self.random.uniform(0, 2 * np.pi)
            radius = self.random.uniform(.05, 0.2)
            yaw = np.random.uniform(0, 2 * np.pi)

            if self.task == "move_from_origin":

                # Hide the target orientation as it is not needed for this task
                physics.named.model.geom_rgba["targetPointer", 3] = 0

            elif self.task == "face_direction":

                # Randomize target orientation
                physics.named.model.body_quat["target"] = np.array([
                    np.cos(yaw / 2), 0, 0, 1 * np.sin(yaw / 2)
                ])

            elif self.task == "move_in_direction":

                # Randomize target orientation
                physics.named.model.body_quat["target"] = np.array([
                    np.cos(yaw / 2), 0, 0, 1 * np.sin(yaw / 2)
                ])

            elif self.task == "move_to_position":

                # Hide the target orientation indicator as it is not needed
                physics.named.model.geom_rgba["targetPointer", 3] = 0

                # Randomize target position
                physics.named.model.body_pos["target", "x"] = radius * np.cos(angle)
                physics.named.model.body_pos["target", "y"] = radius * np.sin(angle)

            elif self.task == "move_to_pose":

                # Randomize full target pose
                physics.named.model.body_pos["target", "x"] = radius * np.cos(angle)
                physics.named.model.body_pos["target", "y"] = radius * np.sin(angle)
                physics.named.model.body_quat["target"] = np.array([
                    np.cos(yaw / 2), 0, 0, 1 * np.sin(yaw / 2)
                ])

            else:
                raise ValueError("Invalid task {}".format(self.task))

            if self.random_pose:
                # Randomize Jitterbug orientation to break symmetries
                rotation_angle = np.random.random() * 2 * np.pi
                rotation_axis = np.concatenate((
                    np.random.random(size=2) * 0.05 - 0.025,
                    (1.0,)
                ))
                rotation_axis /= np.linalg.norm(rotation_axis)
                physics.named.data.qpos["root"][3:] = np.concatenate((
                    (np.cos(rotation_angle / 2),),
                    np.sin(rotation_angle / 2) * rotation_axis
                ))

        super(Jitterbug, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state and the target position
        """
        obs = collections.OrderedDict()
        obs['position'] = physics.jitterbug_position()
        obs['velocity'] = physics.jitterbug_velocity()
        obs['motor_position'] = physics.motor_position()
        obs['motor_velocity'] = physics.motor_velocity()
        if self.task == "move_from_origin":

            # Jitterbug position is a sufficient observation for this task
            pass

        elif self.task == "face_direction":

            # Store the relative target yaw angle
            obs['angle_to_target'] = physics.angle_jitterbug_to_target()

        elif self.task == "move_in_direction":

            # Store the relative target yaw angle
            obs['angle_to_target'] = physics.angle_jitterbug_to_target()

            # Store the speed in the target frame
            obs['speed_in_target_frame'] = (
                physics.jitterbug_velocity_in_target_frame()[0]
            )

        elif self.task == "move_to_position":

            # Store the relative target XYZ position in JB frame
            obs['target_in_jitterbug_frame'] = (
                physics.target_position_in_jitterbug_frame()
            )

        elif self.task == "move_to_pose":

            # Store the relative target XYZ position in JB frame
            obs['target_in_jitterbug_frame'] = (
                physics.target_position_in_jitterbug_frame()
            )

            # Store the relative target yaw angle
            obs['angle_to_target'] = physics.angle_jitterbug_to_target()

        else:
            raise ValueError("Invalid task {}".format(self.task))
        #pickle.dump(obs, self.pickleFile)
        # new_obs = self.PCA(obs)
        if self.use_autoencoder or self.use_several_autoencoders:
            obs = self.encode_obs(obs)
        return obs

    def heading_reward(self, physics):
        """Compute a reward for facing a certain direction

        Returns:
            (float): Angular reward on [0, 1]
        """
        return rewards.tolerance(
            physics.angle_jitterbug_to_target()[0],
            bounds=(0, 0),
            margin=np.pi / 2,
            value_at_margin=0,
            sigmoid='cosine'
        )

    def velocity_reward(self, physics):
        """Compute a reward for moving in a certain direction

        Returns:
            (float): Velocity reward on [0, 1]
        """
        return rewards.tolerance(
            physics.jitterbug_velocity_in_target_frame()[0],
            bounds=(TARGET_SPEED, float('inf')),
            margin=TARGET_SPEED,
            value_at_margin=0,
            sigmoid='linear'
        )

    def position_reward(self, physics):
        """Compute a reward for moving to a certain position

        Returns:
            (float): Position reward on [0, 1]
        """
        return rewards.tolerance(
            np.linalg.norm(
                physics.target_position_in_jitterbug_frame()
            ),
            bounds=(0, 0),
            margin=0.05
        )

    def upright_reward(self, physics):
        """Reward Jitterbug for remaining upright"""
        return rewards.tolerance(
            # Dot product of the Jitterbug Z axis with the global Z
            physics.named.data.xmat['jitterbug', 'zz'],
            bounds=(1, 1),
            margin=0.5
        )

    def get_reward(self, physics):

        r = 0

        if self.task == "move_from_origin":

            r = (1 - self.position_reward(physics))

        elif self.task == "face_direction":

            r = self.heading_reward(physics)

        elif self.task == "move_in_direction":

            r = self.velocity_reward(physics)

        elif self.task == "move_to_position":

            r = self.position_reward(physics)

        elif self.task == "move_to_pose":

            # Use multiplicative reward
            r = (
                    self.position_reward(physics) *
                    self.heading_reward(physics)
            )

        else:
            raise ValueError("Invalid task {}".format(self.task))

        # Reward Jitterbug for staying upright
        r *= self.upright_reward(physics)
        # print(r)
        return r

    def encode_obs(self, obs):
        obsArray = np.concatenate(
            (obs['position'], obs['velocity'], obs['motor_position'], obs['motor_velocity'], obs['angle_to_target']))
        if self.use_autoencoder:
            norm_obs = [np.array(self.jitterbug_autoencoder.normalize_obs(obsArray))]
            #print("###############")
            #print(norm_obs)
            encoded_obs = self.jitterbug_autoencoder.encode(norm_obs)
            #print(encoded_obs)
            encoded_obs_dict = {'observations': np.array(encoded_obs)}
        elif self.use_several_autoencoders:
            encoded_list = []
            norm_obs = [np.array(self.autoencoder_list[0].normalize_obs(obsArray))]
            #print("###############")
            #print(norm_obs)
            for i in range(self.num_autoencoders):
                encoded_list.append(np.array(self.autoencoder_list[i].encode(norm_obs)))
            encoded_obs = sum(encoded_list)/self.num_autoencoders
            #print(encoded_obs)
            encoded_obs_dict = {'observations': encoded_obs}
        return encoded_obs_dict



    def PCA(self, obs):
        obsArray = np.concatenate(
            (obs['position'], obs['velocity'], obs['motor_position'], obs['motor_velocity'], obs['angle_to_target']))
        return {'observations': np.dot(obsArray, self.principalVectors4dim)}

def demo():
    """Demonstrate the Jitterbug domain"""

    # Get some imports
    from dm_control import suite
    from dm_control import viewer

    # Add the jitterbug tasks to the suite
    import jitterbug_dmc

    # Load the Jitterbug domain
    env = suite.load(
        domain_name="jitterbug",
        task_name="move_from_origin",
        visualize_reward=True,
        task_kwargs={
            # "time_limit": float("inf")
        }
    )

    # Use a constant policy
    policy = lambda ts: 0.8
    # Dance, jitterbug, dance!
    viewer.launch(
        env,
        policy=policy,
        title="Jitterbug Demo"
    )


if __name__ == '__main__':
    demo()
