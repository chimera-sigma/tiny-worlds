from setuptools import setup, find_packages

setup(
    name="tiny_world_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "hydra-core>=1.3.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.11",
)
