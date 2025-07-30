from setuptools import setup, find_packages

setup(
    name="subgroup-ruleset",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        pandas, numpy, string, itertools, collections, bisect, math, copy,
        random, skslearn, scipy,
        matplotlib,
        time, operator, 
    ],
    author="Your Name",
    description="A package for subgroup ruleset operations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/subgroup-ruleset",
)