from setuptools import setup, find_packages

setup(
    name="subgroup-rulesets",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy", 
        "scikit-learn",
        "scipy",
        "matplotlib",
    ],
    author="Albert Chiu",
    description="A package for discovering interpretable subgroups in data using rulesets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/albert-chiu/subgroup-ruleset",
)