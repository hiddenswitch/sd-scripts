from setuptools import setup, find_packages

setup(name="sd_scripts",
      install_requires=[open("requirements.txt").readlines()],
      packages=find_packages())
