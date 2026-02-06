from setuptools import setup, find_packages

setup(
    name="dede",
    version="0.1.6",
    author="Zhiying Xu",
    author_email="xuzhiying9510@gmail.com",
    description="Decouple and Decompose: Scaling Resource Allocation with DeDe",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/illinois-nsai/dede",
    packages=find_packages(),
    install_requires=[
        "cvxpy>=1.4.0, <=1.4.3",
        "numpy>=1.26.0, <=1.26.4",
        "ray",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-xdist",
            "gurobipy==12.0.3",
            "ruff"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
