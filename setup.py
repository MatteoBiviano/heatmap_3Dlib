from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='heatmap3D',
    packages=find_packages(include=['heatmap3Dlib']),
    version='0.1.0',
    description='3D heatmap plot library',
    author='Matteo Biviano',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3',
    install_requires=['matplotlib', 'pandas', 'numpy'],
)