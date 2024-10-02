from importlib import util
from pathlib import Path
from setuptools import setup, find_packages
import io

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

requirements_stable_baselines3 = read_requirements("stable_baselines3.txt")
requirements_pettingzoo = read_requirements("pettingzoo.txt")
requirements_dm_env = read_requirements("dm_env.txt")
requirements_rllib = read_requirements("rllib.txt")
requirements_ml_agents = read_requirements("ml_agents.txt")


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
    url="[ANONYMIZED]",
    download_url="[ANONYMIZED]",
    author="[ANONYMIZED]",
    author_email="[ANONYMIZED]",
    description=("A High-Impact Environment Suite for Multi-Agent Research"),
    long_description=io.open("README.md", encoding="utf-8").read(),
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
    extras_require={
        "stable_baselines3": requirements_stable_baselines3,
        "pettingzoo": requirements_pettingzoo,
        "dm_env": requirements_dm_env,
        "rllib": requirements_rllib,
        "ml_agents": requirements_ml_agents,
    },
)
