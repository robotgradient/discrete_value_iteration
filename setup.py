from setuptools import setup
from codecs import open
from os import path


from value_iteration import __version__


setup(name='value_iteration',
      version=__version__,
      description='Discrete Value Iteration: A simple Pytorch implementation of value iteration on graphs',
      author='Julen Urain',
      author_email='julen@robot-learning.de',
      packages=['value_iteration'],
      )