from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'A kNN classifier optimized by P systems'
LONG_DESCRIPTION = 'k-nearest neighbors (kNN) classifier optimized by P systems, called kNN-P, which can improve the performance of the original kNN classifier'

# Setting up
setup(
    name="kNNp",
    version=VERSION,
    author="Khushiyant",
    author_email="<khushiyant2002@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['knn', 'theoretical computer science', 'machine learning', 'p systems'],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)