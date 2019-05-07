"""Provides an OpenAI Gym compatible env wrapper for the Jitterbug domain"""

import dm2gym


class JitterbugGymEnv(dm2gym.DMControlEnv):
    """A renderer with customized settings for the Jitterbug domain"""

    def __init__(self, env, *, render_every=1):
        """Constructor

        Args:
            env (dm_control Environment): The environment we are wrapping

            render_every (int): Draw a human-friendly visualisation of the
                Jitterbug every this many frames
        """
        self.frame_count = 0
        self.render_every = render_every
        super().__init__(env, render_window_mode="opencv")

    def render(self, mode='human', **kwargs):
        """Render with nicer settings for the Jitterbug domain"""

        self.frame_count += 1
        if self.frame_count % self.render_every == 0:

            if not kwargs:
                kwargs = {}

            if 'width' not in kwargs:
                kwargs["width"] = 1024
            if 'height' not in kwargs:
                kwargs["height"] = 768
            if 'camera_id' not in kwargs:
                kwargs["camera_id"] = 1

            super().render(mode, **kwargs)


def demo():
    """Demonstrate the Jitterbug gym interface"""

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
        observation, reward, done, info = env.step(0.9)
        env.render()

    return


if __name__ == '__main__':
    demo()
