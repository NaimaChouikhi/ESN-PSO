# Particle swarm optimization of ESN weights

> ESN is a simple and powerful network. It is simple thanks to its non-complex architecture as well as
its training method. It is powerful thanks to the good results given in the field of machine learning. Moreover, It has a special topology
characterized by random parameters initialization especially those related to the reservoir and the weights.
Although this random initialization is followed by some pre-treatments such as the scaling ofthe reservoir matrix by its spectral
radius, it is still non sufficient to obtain satisfying results. As a remedy to this problem, PSO is used for the fine tuning of some of these
parameters. In fact, the studied approach consists of doing a PSO pre-training of a subset or subsets from the reservoir, input and
backward weight matrices. Hence, the network will not be tuned by fully aleatory variables.

## Getting started
The implemented code is designed for ESN-PSO and applied to the Mackey and Glass time series prediction.
Run the script with Matlab: training_esn_mg_pred.m. 

In order to apply it for other datasets, just upload your dataset in the main file and change the training and testing datasets.

In this version, the data is normalized and resized and the network parameters initialization is altered to minimize the testing error compared to the previous version, also the size of the washing-out data is altered too.