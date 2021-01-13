<h1>Recyclable Classifier - Final Project for Advanced Computer Vision </h1>

**Authors:** Casey Duncan <br />
**Date:** 1/12/2021 <br />
**Note:** This project utilizes the bulk of the Computer-Vision-Recyclable-Classifier on my Github, but utilizes Weights & Biases python library for managing important model artifacts, such as hyperparameters, performance metrics, etc.

<h2>OVERVIEW</h2>

The objective of this project is to provide insight on the efficacy of neural networks in classifying a variety of recyclables and distinguishing which general material category an item belongs to based entirely on relatively unstructured photographs of the objects. To this end, six classes are defined for all possible municipal solid waste items: Cardboard, Glass, Metal, Paper Plastic, and Trash. Given the impressive results demonstrated by both proven and emerging neural network research, we decided to leverage a transfer learning approach using a few popular, modern network architectures to benchmark and explore the accuracy we can achieve using modern techniques on our dataset. In order to both effectively train and evaluate each network's performance appropriated a canonical approach of splitting the dataset into training, validation, and test sets.

<h2>Installation and Usage</h2>

Installing this project can be done in a variety of ways but the suggested method relies on using a virtual environment 
provided through [Poetry](https://python-poetry.org/docs/).

It is also recommended that you use pyenv to manage your variety of python environments. Pyenv can be installed by 
executing:
```
$ curl https://pyenv.run | bash
```
Additional help for setting up pyenv can be found [here](https://realpython.com/intro-to-pyenv/).

Once you have pyenv installed add in your preferred version of Python 3.6, or simply default to Python 3.6.9 (the latest 
version before the release of 3.7) and set pyenv to use this version for this project. 

With the project cloned to you local system, change directory into the project's top level. In order to install all of 
the project dependencies to the virtual environment execute:
```
$ poetry install
```

And in order to run this project with default settings simply execute the following from the project folder's root top level directory (csci508-final):
```
$ poetry run python main.py
```

At the moment, the default setting currently does not do anything, so we recommend you run it from the provided command line interface (CLI) for safely managing data and training neural networks. For in console help simply execute
```
$ poetry run python main.py --help
```
or 
```
$ poetry run python main.py -h
```
The generated menu of options will help guide the user through using any of the pre-built tools.  
```
$ poetry run python main.py -h
usage: main.py [-h] [-a] [-s] [-e] [--train]

optional arguments:
  -h, --help          show this help message and exit
  -a, --augment-data  Allows user to choose augmentations (flip horizontally,
                      flip vertically, rotate, etc.)performed on training data
  -s, --split-data    Allows user to choose how to split Training, Validation,
                      & Test data
  -e, --epoch-qty     Allows user to choose the number of epochs for training
                      the selected model
  --train             Trains the chosen model to classify dataset (defined in
                      load_data.py)
```

By executing with the `--train` flag the user will be prompted to select any model architecture currently implemented, which include the following:
- ResNet18
- ResNet34
- ResNet50
- ResNet50-modnet
- ResNeXt101
- ResNext101-modnet

Note that none of these CLI options require additional arguments. Simply pass with the appropriate flag to initialize an
interface that will guide you through changing the settings. Please reference the .pdf file titled **RecyclableClassifier_FinalReport_DuncanThune** for more information on the currently implemented model architectures and functionality of the optional arguments.

<h2>Data Collection</h2>

In order to compare the performance of various network architectures and methods on our data we are collecting all of 
our results in a shared google sheets project which can be found and modified 
[here](https://drive.google.com/open?id=1LFFuCYt-rlyDO3pLFGtBgJ-dwgkvgq0lGIOQKgtxyao). Note that this is a fully mutable 
sheet as the number and variety of possible hyper-parameters varies with architectures. This may become more 
constrained at some point but as things evolve this flexibility is more critical than safety. 
