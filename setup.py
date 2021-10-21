"""
molpal
active learning to accelerate virtual drug discovery
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    name="molpal",
    author="david graff",
    author_email="deg711@g.harvard.edu",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[],
    url="https://github.com/coleygroup/molpal",
    platforms=["Linux", "Mac OS-X", "Unix"],
    python_requires=">=3.7",
    install_requires=[
        "configargparse",
        "h5py",
        "numpy",
        "ray[default] >= 1.7",
        "scikit_learn",
        "tensorflow",
        "tensorflow_addons",
        "tqdm",
    ],
)
