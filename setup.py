from setuptools import setup

setup(
    name='jitterbug_dmc',
    version='0.0.1a',
    description='A Jitterbug dm_control Reinforcement Learning domain',
    license='MIT',
    python_requires='>=3.5',
    packages=['jitterbug_dmc'],
    install_requires=[
        "dm_control"
        "dm2gym @ git+https://github.com/aaronsnoswell/dm2gym@opencv-render-window"
    ],
    include_package_data=True,
    author='Aaron Snoswell',
    author_email='aaron.snoswell@uqconnect.edu.au',
    url='https://github.com/aaronsnoswell/jitterbug-dmc',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
