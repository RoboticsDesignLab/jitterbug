from setuptools import setup

setup(
    name='jitterbug_dmc',
    version='0.0.1a',
    description='A Jitterbug dm_control Reinforcement Learning domain',
    license='MIT',
    packages=['jitterbug_dmc'],
    install_requires=[
        "dm_control"
    ],
    include_package_data=True,
    author='Aaron Snoswell',
    author_email='aaron.snoswell@uqconnect.edu.au',
    url='https://github.com/aaronsnoswell/jitterbug-dm'
)
