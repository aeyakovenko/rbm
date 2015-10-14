Restricted Boltzmann Machine
============================

Implementation of [Hinton's paper](docs/hinton_rbm_guide.pdf?raw=true) on rbms, and [Back Propagation](docs/rojas-backprop.pdf?raw=true).

run `make mnist_data` to generate the test data
run `make mnist` to test the mnist training.

After backprop there is very strong correlation between the output and the labels

label | correlation coeffecient
-------------------------------
    0 |      0.9693106100687761
    1 |      0.9758625066233139
    2 |      0.9292041170805803
    3 |      0.9318148520851495
    4 |      0.9299791737058097
    5 |      0.8615904996306167
    6 |      0.9533504333370227
    7 |      0.943650526245008
    8 |      0.914467807212961
    9 |      0.869197579806333

So its pretty good at picking the right number.

Monitoring Progress
-------------------

First layer of weights should approximate the input we are training on.  It can be seen here [dist/rbm1.gif](results/rbm1.gif?raw=true)

Second layer looks interesting, but not sure how to interpret it [dist/rbm1.gif](results/rbm2.gif?raw=true)

Thrid layer looks like it might be picking actual classes [dist/rbm1.gif](results/rbm3.gif?raw=true)

For backprop generated the output of the RBM run backwards after backprop training the classes.  I expected to what the network considered each number to look like, but they all look the same.

[dist/bp1.gif](results/bp1.gif?raw=true)
[dist/bp2.gif](results/bp2.gif?raw=true)
[dist/bp3.gif](results/bp3.gif?raw=true)

