import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()


setuptools.setup(
    name="veloxbench",
    version="0.0.1",
    description="Velox Benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    entry_points={},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache 2 License",
    ],
    python_requires=">=3.8",
    maintainer="Velox Developers",
    url="https://github.com/facebookincubator/velox",
)
