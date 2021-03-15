import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textpy",
    version="1.1.1",
    author="Ritvik Rastogi",
    author_email="rastogiritvik99@gmail.com",
    description="NLP made Easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ritvik19",
    packages=setuptools.find_packages(
        exclude=[".git", ".idea", ".gitattributes", ".gitignore", ".github"]
    ),
    install_requires=[
        "Keras>=2.3.1",
        "gensim>=3.8.3",
        "numpy>=1.18.1",
        "pandas>=1.0.3",
        "scikit-learn>=0.23.2",
        "scipy>=1.4.1",
        "spacy>=2.3.0",
        "tensorflow-hub>=0.8.0",
        "tensorflow-text>=2.4.3",
        "tensorflow>=2.2.0",
        "tqdm>=4.46.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)