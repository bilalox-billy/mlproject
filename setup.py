# the setup.py is responsible for packaging the application (My ML application)

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    """this function returns the list of requirements"""
    with open(file_path) as f:
       requirements = f.readlines()
       requirements = [req.replace("\n", "") for req in requirements]

       if HYPHEN_E_DOT in requirements:
           requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="ML project",
    version="0.0.1",
    author = "Bilal Ben Mahria",
    email ="bilal.benmahria.up@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)