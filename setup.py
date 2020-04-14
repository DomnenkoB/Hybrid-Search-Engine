import setuptools


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="hybrid_search_engine",
    version="1.0.0",
    author="Andrii Boichuk, Bohdan Domnenko",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DomnenkoB/Hybrid-Search-Engine",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.6"
)
