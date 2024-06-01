from importlib import util
from pathlib import Path
from setuptools import setup, find_packages

reqs_dir = Path("./requirements")


def read_requirements(filename: str):
    requirements_file = reqs_dir / filename
    if requirements_file.is_file():
        reqs = requirements_file.read_text().splitlines()
        return [
            req.strip()
            for req in reqs
            if not req.strip().startswith("#")
            and req != ""
            and not req.startswith("-r")
        ]
    else:
        return []


requirements_base = read_requirements("base.txt")


def get_version():
    spec = util.spec_from_file_location("metadata", "metadata.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.__version__


setup(
    name="hivex",
    version=get_version(),
    license="Apache 2.0",
    license_files=["LICENSE"],
    url="https://github.com/philippds/hivex-research/hivex",
    download_url="https://github.com/philippds/hivex-research/hivex",
    author="Philipp D Siedler",
    author_email="p.d.siedler@gmail.com",
    description=("A High-Impact Environment Suite for Multi-Agent Research"),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="multi-agent reinforcement-learning python machine-learning",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    install_requires=requirements_base,
)
