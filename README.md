# jitterbug-dmc

A 'Jitterbug' under-actuated continuous control Reinforcement Learning domain,
implemented using the [MuJoCo](http://mujoco.org/) physics engine and
distributed as an extension to the
[Deep Mind Control suite (`dm_control`)](https://github.com/deepmind/dm_control).

![Jitterbug model](figures/jitterbug.jpg)

## Installation

This package is not distributed on PyPI - you will have to install it from
source:

```bash
$> git clone github.com/aaronsnoswell/jitterbug-dmc
$> cd jitterbug-dmc
$> pip install .
```

To test the installation:

```bash
$> cd ~
$> python
>>> import jitterbug_dmc
>>> jitterbug_dmc.demo()
```

## Requirements

This package is designed for Python 3.6+ (but may also work with Python 3.5) 
under Windows, Mac or Linux.

The only pre-requisite package is
[`dm_control`](https://github.com/deepmind/dm_control).

## Usage

Upon importing `jitterbug_dmc`, the domain and tasks are added to the standard
[`dm_control`](https://github.com/deepmind/dm_control) suite.
For example, the `move_from_origin` task can be instantiated as follows;

```python
from dm_control import suite
from dm_control import viewer
import jitterbug_dmc
import numpy as np

env = suite.load(
    domain_name="jitterbug",
    task_name="move_from_origin",
    visualize_reward=True
)
action_spec = env.action_spec()

# Define a uniform random policy
def random_policy(time_step):
    return np.random.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
        size=action_spec.shape
    )

# Launch the viewer
viewer.launch(env, policy=random_policy)
```

## OpenAI Gym Interface

For convenience, we also provide an [OpenAI Gym](https://gym.openai.com/docs/)
compatible interface to this environment using the
[`dm2gym`](https://github.com/zuoxingdong/dm2gym) library.

```python
from dm_control import suite
import jitterbug_dmc

env = JitterbugGymEnv(
    suite.load(
        domain_name="jitterbug",
        task_name="move_from_origin",
        visualize_reward=True
    )
)

# Test the gym interface
env.reset()
for t in range(1000):
    observation, reward, done, info = env.step(
        env.action_space.sample()
    )
    env.render()
env.close()
```

## Heuristic Policies

We provide a heuristic reference policy for each task in the module
[`jitterbug_dmc.heuristic_policies`(jitterbug_dmc/heuristic_policies.py). 

## Tasks

This Reinforcement Learning domain contains several distinct tasks.
All tasks require the jitterbug to remain upright at all times.

 - `move_from_origin` (easy): The jitterbug must move away from the origin
 - `face_direction` (easy): The jitterbug must rotate to face a certain
   direction
 - `move_in_direction` (easy): The jitterbug must achieve a positive velocity in
   a certain direction
 - `move_to_position` (hard): The jitterbug must move to a certain cartesian
   position 
 - `move_to_pose` (hard): The jitterbug must move to a certain cartesian
   position and face in a certain direction 
