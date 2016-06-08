Deep learning for music information retrieval: CBMI2016 paper
-----------------------------
It is build using python on Lasagne-Theano for deep learning and Essentia for feature extraction.
Currently, MIRdl is for easily doing music classification using any deep learning architecture available on Lasagne-Theano.

**Installation**
 
Requires having Lasagne-Theano (http://lasagne.readthedocs.org/en/latest/user/installation.html) and Essentia (http://essentia.upf.edu/documentation/installing.html) installed.

Lasagne is already in a folder that you can download together with MIRdl, to install Theano do: 
> sudo pip install --upgrade https://github.com/Theano/Theano/archive/master.zip

Dependencies: numpy and scipy.

**Important folders**
- *./data/datasets*: the library expects to have the dataset divided by folders that represent the tag to be predicted. 
- *./data/preloaded*: this directory contains the pickle files storing the datasets in a format readable for the library. The name of the pickle file contains all the parameters used for computing it.
- *./data/results*: this directory stores the following files: **.result** (with training and test results), **.training** (having the training evolution, readable with utils.py!), **.param** (storing all the deep learning parameters used for each concrete experiment) and the **.npz** (where the best trained deep learning model is stored).
- the public [Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) dataset, that is also included in this repository.

**Important scripts**
- *runCBMI2016.py*: where the network architecture is selected, you can also set the input and training parameters.
- *buildArchitecture.py*: where the Lasagne-Theano network architecture is set.
- *load_datasets.py*: where audios are loaded, formatted and normalized to be fed into the net. 
- *MIRdl.py*: main part of the library where the training happens.

**Reproducing the paper**
- run: *runCBMI2016.py*. There, you can simply set the parameters and choose the architecture you want to use according to the paper: 'blackbox' for *Black-box*, 'time' for *Time*, 'frequency' for *Frequency*, 'mergeTimeFrequency' for *Time-Frequency* and 'loadMergeTimeFrequency' for *Time-FrequencyInit*. You will be able to reproduce all the results provided in the paper. The Ballroom dataset is also uploaded to this GitHub repository, after downloading it and installing the dependencies the experiments are ready to run. The *Time* and *Frequncy* models to initialize the *Time-FrequencyInit* architecture are also provided in *./data/preloaded*.

**Steps for using MIRdl**
- **0.0)** Install.

- **0.1)** Understand this tutorial: http://lasagne.readthedocs.org/en/latest/user/tutorial.html. This library is based on it!

- **1)** Download a dataset and copy it in *./data/datasets* (that repository already includes the Ballroom dataset). The library expects to have the dataset divided by folders that represent the tag to be predicted. 
For example, for the GTZAN dataset (http://marsyasweb.appspot.com/download/data_sets/) the library expects:
>./data/datasets/GTZAN/blues
>
>./data/datasets/GTZAN/classical
>
> (...)
>
>./data/datasets/GTZAN/rock
- **2)** Adapt the *load_datasets.py* function to work using your dataset. We recommend you to use first the GTZAN dataset (already implemented) to understand how it works.

- **3)** Set the *runCBMI2016.py* parameters and the deep learning architecture in *buildArchitecture.py*.

- **4)** Run *runCBMI2016.py*.

**Reference**

- Jordi Pons, Thomas Lidy & Xavier Serra (2016, June). "Experimenting with Musically Motivated Convolutional Neural Networks" in 14th International Workshop on Content-Based Multimedia Indexing (CBMI). Publisher: IEEE.

**License**

This code is Copyright 2016 - Music Technology Group, Universitat Pompeu Fabra. It is released under [Affero GPLv3 license](http://www.gnu.org/licenses/agpl.html) except for the third party libraries and datasets which have its own licenses.

This code is free software: you can redistribute it and/or modify it under the terms of the Affero GPLv3 as published by the Free Software Foundation. This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
