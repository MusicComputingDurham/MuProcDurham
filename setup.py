import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="muprocdurham",
    version="0.0.7",
    author="Robert Lieck, Eamonn Bell",
    # author_email="",
    description="Python package for the Music Processing lecture and practicals at Durham University",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MusicComputingDurham/MuProcDurham",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

