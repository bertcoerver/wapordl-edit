from setuptools import find_packages, setup

setup(
    name="wapordl",
    version="0.13",
    packages=find_packages(include=["wapordl", "wapordl.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "requests",
        "pandas>=2.1.0,<3",
        "numpy>=1.15,<2",
        "gdal>=3.4.0,<4",
        "shapely>=2.0.0",
        "tqdm",
        "pydantic>=2",
    ],
)
