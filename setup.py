from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="llrflibs-rl",
    version="0.1.0",
    maintainer="Ming Liu",
    maintainer_email="mliu1018@ustc.edu.cn",
    author="Ming Liu",
    author_email="mliu1018@ustc.edu.cn",
    description="Reinforcement-learning environments and examples for LLRF control.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iuming/LLRFLibs_RL",
    license="BSD 3-Clause",
    packages=find_packages(include=["llrflibs_rl", "llrflibs_rl.*"]),
    install_requires=["numpy", "scipy", "gymnasium", "stable-baselines3", "llrflibs>=1.0.2"],
    zip_safe=False,
)