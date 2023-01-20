# -*- coding: utf-8 -*-
"""
setup.py - Luke Bouma (luke@astro.caltech.edu) - Apr 2019

Stolen from the astrobase setup.py
"""
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        import pytest

        if not self.pytest_args:
            targs = []
        else:
            targs = shlex.split(self.pytest_args)

        errno = pytest.main(targs)
        sys.exit(errno)


def readme():
    with open('README.md') as f:
        return f.read()

INSTALL_REQUIRES = [
    'numpy>=1.4.0',
    'scipy',
    'astropy>=1.3',
    'matplotlib',
]

EXTRAS_REQUIRE = {
    'all':[
        'emcee==3.0rc1',
        'h5py',
        'batman-package',
        'corner'
    ]
}



###############
## RUN SETUP ##
###############

# run setup.
version = 0.3
setup(
    name='cdips',
    version=version,
    description=('Python modules and scripts used for CDIPS project.'),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    keywords='astronomy',
    url='https://github.com/lgbouma/cdips',
    download_url=f'https://github.com/lgbouma/cdips/archive/refs/tags/v{str(version).replace(".","")}.tar.gz',
    author='Luke Bouma',
    author_email='luke@astro.caltech.edu',
    license='MIT',
    packages=[
        'cdips',
        'cdips.lcproc',
        'cdips.catalogbuild',
        'cdips.plotting',
        'cdips.tests',
        'cdips.utils',
        'cdips.pipetrextuning'
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=['pytest==3.8.2',],
    cmdclass={'test':PyTest},
    include_package_data=True,
    zip_safe=False,
)
