from setuptools import setup

__version__ = ""

with open("fedgraph/version.py", "r") as f:
    exec(f.read(), globals())


with open("README.md", "r") as f:
    README = f.read()

setup(
    name="fedgraph",
    version=__version__,
    packages=["fedgraph"],
    author="Yuhang Yao",
    author_email="yuhangya@andrew.cmu.edu",
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
        "torch>=2.0.0",
        "torch-scatter>=2.0.9",
        "torch-sparse>=0.6.15",
        "torch-cluster>=1.6.0",
        "torch-spline-conv>=1.2.1",
        "torch-geometric>=2.1.0.post1",
        "omegaconf>=2.3.0",
        "ray[default]>=2.6.3",
        "PyYAML>=5.4.0",
        "attridict",
        "torchmetrics",
        "setuptools",
        "sphinx_rtd_theme",
        "sphinxcontrib-bibtex",
        "matplotlib",
        "sphinx_gallery",
        "tensorboard",
        "dtaidistance",
        "gdown",
        "pandas",
        "twine==5.0.0",
        "scikit-learn",
        "tenseal",
        "huggingface_hub",
        "ogb",
    ],
    extras_require={"dev": ["build", "mypy", "pre-commit", "pytest"]},
    include_package_data=True,
    package_data={
        "fedgraph": ["he_context.pkl"],
    },
)
