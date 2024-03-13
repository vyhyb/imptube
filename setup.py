from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

setup(
    name= "imptube",
    version= "0.0.3",
    description = "A package for impedance tube measurements.",
    package_dir={"": "imptube"},
    packages=find_packages(where="imptube"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= "https://github.com/vyhyb/imptube/tree/main",
    author="David Jun",
    author_email="David.Jun@vut.cz",
    licence="MIT",
    classifiers=[
        "Licence :: OSI Approved :: MIT Licence",
        "Programming Language :: Python :: 3.9", 
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "sounddevice",
        "soundfile", 
        "scipy", 
        "numpy", 
        "pandas",
        "logging"
    ],
    extras_require={
    "dev": ["pytest", "twine"],
    },
    python_requires=">=3.9"
)