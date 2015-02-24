#!/usr/bin/env python
# Copyright (C) 2015 Tadej Stajner <tadej@tdj.si>
# License: 3-clause BSD

from setuptools import setup

setup(name='autokit',
      version='0.1',
      description='autokit - machine learning for busy people',
      author='Tadej Stajner',
      author_email='tadej@tdj.si',
      url='https://github.com/tadejs/autokit',
      packages=['autokit'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
      ],
      platforms=['Linux', 'OS-X', 'Windows'],
      dependency_links = ['https://github.com/hyperopt/hyperopt-sklearn/tarball/master#egg=hyperopt-sklearn-0.0.1'],
      install_requires = [
        'numpy',
        'scipy',
        'scikit-learn',
        'networkx',
        'hyperopt'],
    )
