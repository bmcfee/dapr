from setuptools import setup, find_packages

import imp

version = imp.load_source('dapr.version', 'dapr/version.py')
description = 'Deep architecture plot rendering',

setup(
    name='dapr',
    version=version.version,
    description=description,
    author='Brian McFee',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/bmcfee/dapr',
    download_url='http://github.com/bmcfee/dapr/releases',
    packages=find_packages(),
    long_description=description,
    classifiers=[
        "License :: OSI Approved :: MIT License (MIT)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
    keywords='machine learning',
    license='ISC',
    install_requires=[
        'matplotlib >= 1.5',
        'six',
    ],
    extras_require={
        'docs': ['numpydoc']
    }
)
