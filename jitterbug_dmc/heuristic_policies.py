"""A collection of heuristic policies that solve the Jitterbug tasks"""

import numpy as np


def face_direction(ts, *, angle_to_target=None):
    """Face jitterbug towards the target

    Args:
        ts (dm_control Timestep): Timestep object

        angle_to_target (float): Optional relative angle in radians to the
            desired heading
    """

    if angle_to_target is None:
        angle_to_target = ts.observation['angle_to_target']

    return 0.9 * max(
        min(
            3 * angle_to_target / np.pi,
            1
        ),
        -1
    )


def move_forward(ts, *, kick_angle=np.deg2rad(45), speed=0.3):
    """Move Jitterbug forwards in the direction it is facing

    Args:
        kick_angle (float): Motor angle magnitude in radians at which to kick
            the motor
        speed (float): Speed to drive the motor at
    """

    motor_angle = ts.observation['motor_position'][0]
    motor_vel = ts.observation['motor_velocity'][0]

    if motor_angle < -kick_angle:
        return speed
    elif motor_angle > kick_angle:
        return -speed
    else:
        if motor_vel > 0:
            return speed
        else:
            return -speed


def move_from_origin(ts):
    """Move Jitterbug away from the origin"""
    return move_forward(ts)


def move_in_direction(ts, *, angle_threshold=np.deg2rad(20)):
    """Move Jitterbug in a certain direction

    Args:
        angle_threshold (float): Threshold relative angle in radians at which to
            rotate towards the target or move towards the target
    """

    print(ts.observation['speed_in_target_frame'])

    angle_to_target = ts.observation['angle_to_target']

    if np.abs(angle_to_target) > angle_threshold:
        return face_direction(ts)
    else:
        return move_forward(ts)


def move_to_position(ts, *, angle_threshold=np.deg2rad(20)):
    """Move Jitterbug to a certain position

    Args:
        angle_threshold (float): Threshold relative angle in radians at which to
            rotate towards the target or move towards the target
    """

    dx, dy = ts.observation['target_in_jitterbug_frame'][0:2]
    angle_to_target = np.arctan2(dx, -dy)

    if np.abs(angle_to_target) > angle_threshold:
        return face_direction(ts, angle_to_target=angle_to_target)
    else:
        return move_forward(ts)


def move_to_pose(ts, *, angle_threshold=np.deg2rad(20)):
    """Move Jitterbug to a certain pose

    Args:
        angle_threshold (float): Threshold relative angle in radians at which to
            rotate towards the target or move towards the target
    """

    dx, dy = ts.observation['target_in_jitterbug_frame'][0:2]
    angle_to_target = np.arctan2(dx, -dy)

    if np.abs(angle_to_target) > angle_threshold:
        # Face towards the target
        return face_direction(ts, angle_to_target=angle_to_target)
    else:
        if np.linalg.norm([dx, dy]) > 0.01:
            # Move towards the target
            return move_forward(ts)
        else:
            # Face the target yaw direction
            return face_direction(ts)


def demo():
    """Demonstrate the hueristic policies
    """

    # Get some imports
    from dm_control import suite
    from dm_control import viewer

    # Add the jitterbug tasks to the suite
    import jitterbug_dmc

    viewer.launch(
        suite.load(
            domain_name="jitterbug",
            task_name="move_to_pose",
            visualize_reward=True
        ),
        policy=move_to_pose,
        title="Jitterbug Policy Demo"
    )


if __name__ == '__main__':
    demo()
