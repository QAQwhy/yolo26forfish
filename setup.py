"""Install this source-only project with: python -m pip install ."""
from setuptools import find_packages, setup

setup(
    name='underwater26',
    version='0.1.0',
    description='Minimal underwater YOLO26 research extensions',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=['ultralytics==8.4.48', 'timm>=1.0'],
    license='AGPL-3.0',
)
