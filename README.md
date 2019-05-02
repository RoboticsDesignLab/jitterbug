# jitterbug-dmc

A Jitterbug dm_control Reinforcement Learning domain

![Jitterbug model](figures/jitterbug.jpg)

# Installation

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

# Requirements

This package is designed for Python 3.6+ (but may also work with Python 3.5) 
under Windows, Mac or Linux.

The only pre-requisite package is
[`dm_control`](https://github.com/deepmind/dm_control).
