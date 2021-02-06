"""Dummy setup.py

Taken from https://packaging.python.org/tutorials/packaging-projects/
"""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rfi-gcsk",  # TBD
    version="0.0.1",  # TBD
    author="Gunnar Koenig",
    author_email="gunnar.koenig.edu@pm.me",
    description="Relative Feature Importance Package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gcskoenig/rfi",
    packages=setuptools.find_packages(),
    classifiers=[  # TBD
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  # TBD
)
