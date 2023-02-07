from setuptools import find_packages, setup
setup(
    name='llimcobe',
    packages=find_packages(include=['LCB']),
    version='0.1.0',
    description='Lossless Image Compression Benchmark',
    author='Xavier Fernàndez Mellado',
    license='MIT',
    install_requires=['matplotlib', 'numpy', 'typing'],
    setup_reqires=['matplotlib', 'numpy', 'typing'],
)