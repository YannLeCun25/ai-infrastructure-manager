from setuptools import setup, find_packages

setup(
    name="ai-infrastructure-manager",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    author="Yann LeCun",
    description="Intelligent resource management and orchestration for large-scale AI training clusters.",
    url="https://github.com/YannLeCun25/ai-infrastructure-manager",
)
