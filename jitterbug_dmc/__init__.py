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

import inspect

from dm_control import suite

from jitterbug_dmc import jitterbug

# Add jitterbug to domain record
suite._DOMAINS.update({
    name: module
    for name, module in locals().items()
    if inspect.ismodule(module) and hasattr(module, 'SUITE')
})

# Update task lists
suite.ALL_TASKS = suite._get_tasks(tag=None)
suite.BENCHMARKING = suite._get_tasks('benchmarking')
suite.EASY = suite._get_tasks('easy')
suite.HARD = suite._get_tasks('hard')
suite.EXTRA = tuple(sorted(set(suite.ALL_TASKS) - set(suite.BENCHMARKING)))
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)

# Convenience imports
from jitterbug_dmc.jitterbug import (
    Jitterbug,
    demo
)

from jitterbug_dmc.gym_wrapper import JitterbugGymEnv
