from setuptools import setup, find_packages

setup(
    name="AdversarialRulLib",
    version="0.1",
    description="This library aims to implement the different attacks and defense methods against predictive maintenance.",
    author="Pierre-Francois Maillard",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.1.1",
        "numpy>=1.24.1",
        "torch>=2.2.1",
        "tqdm>=4.66.1",
        "matplotlib>=3.7.3",
        "scipy>=1.11.2",
        "xgboost>=2.1.1",
        "scikit-learn>=1.1.3",
    ],
    python_requires='>=3.10.12',
    license="MIT",
    url="https://github.com/PF-Maillard/AdversarialRulLib",
)