from setuptools import setup, find_packages

setup(
    name="mini1-rl",
    version="0.1.0",
    description="EECS 590 Mini-1: Reinforcement Learning repository with classical and deep RL on grid-world MDPs",
    author="Ismail",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.6",
        "scipy>=1.10",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    ],
    extras_require={
        "drl": ["torch>=2.0", "tensorboard>=2.13"],
    },
)
