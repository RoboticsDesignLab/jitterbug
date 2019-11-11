import os
import sys

import numpy as np
import xml.etree.ElementTree as ET

from dm_control.utils import containers
from dm_control.rl import control
from dm_control.utils import io as resources
from dm_control.suite import common

# Load the suite so we can add to it
SUITE = containers.TaggedTasks()

# sys.path.insert(0, os.path.join(
#     os.path.dirname(os.path.realpath(__file__)),
#     ".."
# ))

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from jitterbug_dmc.jitterbug import Jitterbug, Physics

# Load the suite so we can add to it
SUITE = containers.TaggedTasks()

# Task constants
DEFAULT_TIME_LIMIT = 10
DEFAULT_CONTROL_TIMESTEP = 0.01
TARGET_SPEED = 0.1


def str2array(string):
    """Convert the string containing coordinates into an array.

     Args:
         string (str): string containing coordinates to be converted"""
    arr_str = string.split(" ")
    N = len(arr_str)
    arr_float = np.zeros(N)
    for i in range(N):
        arr_float[i] = float(arr_str[i])

    return arr_float


def array2str(arr):
    """Convert the array containing coordinates into a string.

    Args:
        arr (np array): array containing the coordinates to be converted"""
    arr_str = []
    for el in arr:
        arr_str.append(str(el))

    return " ".join(arr_str)


def update_features(string, d_array):
    """Update the string that contains the geometry/joint coordinates with the values
    in d_array.

    Args:
        string (str): string containing the geometry coordinates to be updated
        d_array (Numpy array): array containing the noise to be added to the coordinates

    """
    # Convert the string into array of floats
    arr_float = str2array(string)

    # Update the coordinates with d_array

    arr_float += d_array

    # Convert the updated array back into a string
    return array2str(arr_float)


def fromto2vect(string):
    """Calculate a normalized vector directing the cylinder described in string.

    Args:
        string (str): contains the cylinder coordinates
    """
    fromto_arr = str2array(string)
    v_1 = fromto_arr[:3]
    v_2 = fromto_arr[3:]
    v = v_2 - v_1
    # Normalize vector
    v /= np.linalg.norm(v)

    return v


def augment_Jitterbug(modify_legs=False,
                      sd_legs=np.array([0.003, 0.003, 0.002]),
                      modify_mass=False,
                      sd_mass_pos=np.array([0.0015, 0.002, 0.001]),
                      sd_mass_size=np.array([0.0007, 0.0007, 0.0007]),
                      modify_coreBody1=False,
                      sd_coreBody1_density=10.,
                      modify_coreBody2=False,
                      sd_coreBody2_density=80.,
                      modify_global_density=False,
                      sd_global_density=200.,
                      modify_gear=False,
                      sd_gear=0.001,
                      original_file='jitterbug.xml',
                      new_file='augmented_jitterbug.xml'
                      ):
    """Create an augmented version of Jitterbug by adding some random noise to some of the key parameters
        defined in jitterbug.xml

        Args:
            modify_legs (bool): whether to add noise to the position of the legs extremities or not
            sd_legs (Numpy array): array containing the standard deviations of the random normal noise to add to
                                   the extremities of each leg. It has 3 components: [sd_x, sd_y, sd_z]
            modify_mass (bool): whether to add noise to the position and size of the mass or not
            sd_mass_pos (Numpy array): array containing the standard deviations of the random normal noise to add to
                                   the position of the mass
            sd_mass_size (Numpy array): array containing the standard deviations of the random normal noise to add
                                        to the size of the mass
            modify_coreBody1 (bool): whether to add noise to the coreBody1 density or not
            sd_coreBody1_density (Float): standard deviation of the random normal noise to add to the coreBody1
                                          density
            modify_coreBody2 (bool): whether to add noise to the coreBody2 density or not
            sd_coreBody2_density (Float): standard deviation of the random normal noise to add to the coreBody2
                                          density
            modify_global_density (bool): whether to add noise to the standard density or not. This density is used
                                          in the legs, the mass and the screw.
            sd_global_density (Float): standard deviation of the random normal noise to add to the standard density
            modify_gear (bool): whether to add noise to the actuator gear or not
            sd_gear (Float): standard deviation of the random normal noise to add to the actuator gear
            original_file (str): the path of the original Jitterbug file
            new_file (str): the path of the file of the augmented Jitterbug

        """

    # Open original Jitterbug file
    original_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), original_file)
    tree = ET.parse(original_path)
    root = tree.getroot()

    if modify_global_density:
        for deflt in root.findall('default'):  # Find default to modify density
            for geom in deflt.findall('geom'):
                d_global_density = np.array([np.random.normal(0, sd_global_density)])
                geom.attrib['density'] = update_features(geom.attrib['density'], d_global_density)
    for wb in root.findall('worldbody'):  # Find worldbody
        for bd in wb.findall('body'):  # Find direct worldbody children: jitterbug, target
            if bd.attrib['name'] == "jitterbug":  # Focus on jitterbug child
                for geom in bd.findall('geom'):  # Focus on coreBody{i}
                    if geom.attrib['name'] == "coreBody1" and modify_coreBody1:
                        # Modify density of the box
                        dd = np.array([np.random.normal(0, sd_coreBody1_density)])
                        geom.attrib['density'] = update_features(geom.attrib['density'], dd)

                    elif geom.attrib['name'] == "coreBody2" and modify_coreBody2:
                        dd = np.array([np.random.normal(0, sd_coreBody2_density)])
                        geom.attrib['density'] = update_features(geom.attrib['density'], dd)

                for child_bd in bd.findall('body'):  # Find direct jitterbug children bodies: upper{i}leg, mass
                    if child_bd.attrib['name'] in [f"leg{i}upper" for i in range(1, 5)] and modify_legs:
                        # Modify upper legs with random noise
                        dx_u = np.random.normal(0, sd_legs[0])
                        dy_u = np.random.normal(0, sd_legs[1])
                        dz_u = np.random.normal(0, sd_legs[2])
                        for geom in child_bd.findall('geom'):
                            if geom.attrib['type'] == "cylinder":
                                d_array = np.array([dx_u, dy_u, dz_u, 0., 0., 0.])
                                geom_str = geom.attrib['fromto']
                                geom_str_updated = update_features(geom_str, d_array)
                                geom.attrib['fromto'] = geom_str_updated
                                v_u = fromto2vect(geom_str_updated)

                            elif geom.attrib['type'] == 'sphere':
                                d_array = np.array([dx_u, dy_u, dz_u])
                                geom_str = geom.attrib['pos']
                                geom.attrib['pos'] = update_features(geom_str, d_array)

                        # Modify joint between upper leg and core body
                        for joint in child_bd.findall('joint'):
                            i_z = np.array([0., 0., 1.])
                            axis_updated = np.cross(v_u, i_z)
                            axis_updated /= np.linalg.norm(axis_updated)
                            joint.attrib['axis'] = array2str(axis_updated)

                        # Modify lower legs with random noise
                        for leg_lower in child_bd.findall('body'):
                            dx_l = np.random.normal(0, sd_legs[0])
                            dy_l = np.random.normal(0, sd_legs[1])
                            dz_l = np.random.normal(0, sd_legs[2])
                            for geom in leg_lower.findall('geom'):
                                if geom.attrib['type'] == "cylinder":
                                    d_array = np.array([dx_l, dy_l, dz_l, dx_u, dy_u, dz_u])
                                    geom_str = geom.attrib['fromto']
                                    geom_str_updated = update_features(geom_str, d_array)
                                    geom.attrib['fromto'] = geom_str_updated
                                    v_l = fromto2vect(geom_str_updated)

                                elif geom.attrib['type'] == 'sphere':
                                    d_array = np.array([dx_l, dy_l, dz_l])
                                    geom_str = geom.attrib['pos']
                                    geom.attrib['pos'] = update_features(geom_str, d_array)

                            for joint in leg_lower.findall('joint'):
                                # Modify joint position
                                d_array = np.array([dx_u, dy_u, dz_u])
                                joint_pos_str = joint.attrib['pos']
                                joint.attrib['pos'] = update_features(joint_pos_str, d_array)

                                # Modify joint axis
                                axis_updated = np.cross(v_u, v_l)
                                axis_updated /= np.linalg.norm(axis_updated)
                                joint.attrib['axis'] = array2str(axis_updated)

                    elif child_bd.attrib['name'] == "mass" and modify_mass:
                        dp_x = np.random.normal(0, sd_mass_pos[0])
                        dp_y = np.clip(np.random.normal(0, sd_mass_pos[1]), -0.001, 1.)
                        dp_z = np.clip(np.random.normal(0, sd_mass_pos[2]), -0.001, 1.)
                        dp_array = np.array([dp_x, dp_y, dp_z])
                        for geom in child_bd.iter('geom'):
                            if geom.attrib['name'] == "threadMass":
                                d_array = np.array([dp_x, dp_y, 0., dp_x, dp_y, dp_z])
                                geom.attrib['fromto'] = update_features(geom.attrib['fromto'], d_array)

                            elif geom.attrib['name'] == "mass_geom":
                                # Modify size of the mass
                                ds_x = np.random.normal(0, sd_mass_size[0])
                                ds_y = np.random.normal(0, sd_mass_size[1])
                                ds_z = np.random.normal(0, sd_mass_size[2])
                                ds_array = np.array([ds_x, ds_y, ds_z])
                                geom_size_str = geom.attrib['size']
                                geom.attrib['size'] = update_features(geom_size_str, ds_array)

                                # Modify position of the mass
                                geom_pos_str = geom.attrib['pos']
                                geom.attrib['pos'] = update_features(geom_pos_str, dp_array)

                        for joint in child_bd.iter('joint'):
                            joint.attrib['pos'] = update_features(joint.attrib['pos'], dp_array)

    if modify_gear:
        for ac in root.findall('actuator'):
            for ger in ac.findall('general'):
                gear = np.array(np.random.normal(0, sd_gear))
                ger.attrib['gear'] = update_features(ger.attrib['gear'], gear)

    # Save augmented Jitterbug in a new XML file
    new_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), new_file)
    tree.write(new_path)

    # Print the changes made
    print_changes(modify_legs,
                  sd_legs,
                  modify_mass,
                  sd_mass_pos,
                  sd_mass_size,
                  modify_coreBody1,
                  sd_coreBody1_density,
                  modify_coreBody2,
                  sd_coreBody2_density,
                  modify_global_density,
                  sd_global_density,
                  modify_gear,
                  sd_gear
                  )


def print_changes(modify_legs=False,
                  sd_legs=np.array([0.003, 0.003, 0.002]),
                  modify_mass=False,
                  sd_mass_pos=np.array([0.0015, 0.002, 0.001]),
                  sd_mass_size=np.array([0.0007, 0.0007, 0.0007]),
                  modify_coreBody1=False,
                  sd_coreBody1_density=10.,
                  modify_coreBody2=False,
                  sd_coreBody2_density=80.,
                  modify_global_density=False,
                  sd_global_density=200.,
                  modify_gear=False,
                  sd_gear=0.001,
                  ):
    """Print the standard deviation used to modify each component."""

    print("Changes made on Jitterbug:")
    if modify_legs:
        print("Legs: ", sd_legs)
    if modify_mass:
        print("Mass : pos", sd_mass_pos, " size ", sd_mass_size)
    if modify_global_density:
        print("Global density: ", sd_global_density)
    if modify_coreBody1:
        print("Core body 1 density: ", sd_coreBody1_density)
    if modify_coreBody2:
        print("Core body 2 density: ", sd_coreBody2_density)
    if modify_gear:
        print("Gear: ", sd_gear)
    if not (modify_legs or modify_mass or modify_global_density or modify_coreBody1 or modify_coreBody2 or modify_gear):
        print("None")

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
        environment_kwargs=None,
        **kwargs
):
    """Move the Jitterbug away from the origin"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="move_from_origin", **kwargs)
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
        environment_kwargs=None,
        **kwargs
):
    """Move the Jitterbug to face a certain yaw angle"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="face_direction", **kwargs)
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
        environment_kwargs=None,
        **kwargs
):
    """Move the Jitterbug in a certain direction"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="move_in_direction", **kwargs)
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
        environment_kwargs=None,
        **kwargs
):
    """Move the Jitterbug to a certain XYZ position"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="move_to_position", **kwargs)
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
        environment_kwargs=None,
        **kwargs
):
    """Move the Jitterbug to a certain XYZRPY pose"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jitterbug(random=random, task="move_to_pose", **kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=control_timestep,
        **environment_kwargs
    )


class Augmented_jitterbug(Jitterbug):
    "An augmented Jitterbug class"

    def __init__(self,
                 random=None,
                 task="move_from_origin",
                 random_pose=True,
                 norm_obs=False):
        """Initialize an instance of the `Jitterbug` domain

        Args:
            random (numpy.random.RandomState): Options are;
                - numpy.random.RandomState instance
                - An integer seed for creating a new `RandomState`
                - None to select a seed automatically (default)
            task (str): Specifies which task to configure. Options are;
                - move_from_origin =
                - face_direction
                - move_in_direction
                - move_to_position
                - move_to_pose
            random_pose (bool): If true, initialize the Jitterbug with a random
                pose to break symmetries
                norm_obs (bool): If true, observations will be approximately normalized
                   to the range (-1, 1)
        """
        super(Augmented_jitterbug, self).__init__(random,
                                                  task,
                                                  random_pose,
                                                  norm_obs)


if __name__ == '__main__':
    augment_Jitterbug(modify_legs=True,
                      modify_mass=True)
