from setuptools import setup, find_packages

setup(name="sd_scripts",
      install_requires=[open("requirements.txt").readlines()],
      packages=find_packages(),
      package_data={
            'sd_scripts.finetune.blip': ['med_config.json'],
      },
)
