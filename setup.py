from setuptools import setup

setup(
    name='LightRL',
    version='0.0.1',
    packages=setuptools.find_packages(),
    description='a drl package',
    author='lintao',
    author_email='lintao209@outlook.com',
    install_requires=['numpy', 'torch'],
    python_requires='>=3.6',
)