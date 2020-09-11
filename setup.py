from setuptools import setup
import os

v_temp = {}
with open("gsa_framework/version.py") as fp:
    exec(fp.read(), v_temp)
version = ".".join((str(x) for x in v_temp["version"]))


setup(
    name="gsa_framework",
    version=version,
    packages=[
        "gsa_framework",
        "gsa_framework.sampling",
        "gsa_framework.sensitivity_analysis",
    ],
    author="Aleksanda Kim",
    author_email="aleksandra.kim@icloud.com",
    license="BSD-3-Clause",
    package_data={
        "gsa_framework": [os.path.join("sampling", "data", "directions.npy")]
    },
    install_requires=[
        "h5py",
        "numpy",
        "plotly",
        "scikit-learn",
        "scipy",
        "xgboost",
    ],
    url="https://github.com/aleksandra-kim/gsa_framework",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    description="Generic framework for global sensitivity analysis",
    classifiers=[
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
