from setuptools import setup

__version__ = ""

with open("fedgraph/version.py", "r") as f:
    exec(f.read(), globals())


with open("README.md", "r") as f:
    README = f.read()

setup(
    name="fedgraph",
    version=__version__,
    author="Jiayu Chang, Yuhang Yao",
    author_email="jiayuc@andrew.cmu.edu, yuhangya@andrew.cmu.edu",
    description="Federated Graph Learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/FedGraph/fedgraph",
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["Graph Neural Networks", "Federated Learning"],
    python_requires=">=3.9",
    install_requires=[
        "torch-scatter>=2.0.9",
        "torch-sparse>=0.6.15",
        "torch-cluster>=1.6.0",
        "torch-spline-conv>=1.2.1",
        "torch-geometric>=2.1.0.post1",
        "omegaconf>=2.3.0",
        "ray>=2.6.3",
    ],
    extras_require={"dev": ["build", "mypy", "pre-commit", "pytest"]},
)
