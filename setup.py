from setuptools import setup

setup(
    name='jitterbug_dmc',
    version='0.0.1a',
    description='A Jitterbug dm_control Reinforcement Learning domain',
    license='MIT',
    python_requires='>=3.5',
    packages=['jitterbug_dmc'],
    install_requires=[
        "dm_control @ git+https://github.com/deepmind/dm_control@09b27a0b7232c6a5a5045a2b7a608cbfc693f85a",
        "dm2gym @ git+https://github.com/zuoxingdong/dm2gym@e16048a33a875943556a62b69bbf63e28c7f1d3c"
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
