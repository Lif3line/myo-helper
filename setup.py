"""Utiltiy functions for working with Myo Armband data."""

from setuptools import setup, find_packages


setup(name='myo_helper',
      version='0.1',
      description='Utiltiy functions for working with Myo Armband data',
      author='Lif3line',
      author_email='adamhartwell2@gmail.com',
      license='MIT',
      packages=find_packages(),
      url='https://github.com/Lif3line/myo_helper',  # use the URL to the github repo
      install_requires=[
          'scipy',
          'sklearn',
          'numpy'
      ],
      keywords='myo emg')
