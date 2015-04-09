#!/usr/bin/env python

from distutils.core import setup

setup(
	name = 'nonconformist',
	packages = ['nonconformist'],
	version = '1.0.0',
	description = 'Python implementation of the conformal prediction framework.',
	author = 'Henrik Linusson',
	author_email = 'henrik.linusson@gmail.com',
	url = 'https://github.com/donlnz/nonconformist',
	download_url = 'https://github.com/donlnz/nonconformist/tarball/1.0.0',
	install_requires = ['numpy', 'scikit-learn'],
	keywords = ['conformal prediction',
	            'machine learning',
	            'classification',
	            'regression'],
	classifiers=['Intended Audience :: Science/Research',
	             'Intended Audience :: Developers',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 ],
)

# Authors: Henrik Linusson
