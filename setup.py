"""Minimal setuptools config: installs the repo's packages as `point-in-time-llm` (dependencies are in requirements.txt).

Usage: pip install -e .
"""
from setuptools import setup, find_packages

setup(name="point-in-time-llm", python_requires=">=3.11", packages=find_packages())
