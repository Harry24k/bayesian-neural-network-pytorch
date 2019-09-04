import setuptools

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
    name = 'torchbnn',
    version = '0.1',
    description = 'Bayesian Neural Network for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'Harry Kim <24k.harry@gmail.com>',
    packages = setuptools.find_packages(),
    keyword = ['bayesian', 'bnn'],
    python_requires = '>=3',
    zip_safe = False,
    license="MIT",
    url = 'https://github.com/Harry24k/bayesian-neural-network',
    
    classifiers = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 4 - Beta',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',

    # Pick your license as you wish (should match "license" above)
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.6',

    'Operating System :: OS Independent',
    ],
)