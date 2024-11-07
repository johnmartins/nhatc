from setuptools import setup, find_packages

VERSION = '0.1.1'
DESCRIPTION = 'Library for running non-hierarchical multi-disciplinary optimization'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nhatc',
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy==2.0.*', 'scipy==1.14.*', 'cexprtk==0.4.*'],
    extras_require={
        "dev": [
            "pytest==8.*",
            "twine>=4.0.2"
        ]
    },
    url="https://github.com/johnmartins/nhatc",
    author="Julian Martinsson Bonde",
    author_email="johnmartins1992@gmail.com",
    license="MIT"
)