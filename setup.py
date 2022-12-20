#!/usr/bin/env python

import sys
from setuptools import setup, find_packages

ROOT_DIR_NAME = "kbqa"

sys.path.append(ROOT_DIR_NAME)

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name=ROOT_DIR_NAME,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(
        where=".", exclude=("deploy", "experiments", "subgraphs_dataset")
    ),
)
