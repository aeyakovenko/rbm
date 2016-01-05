Restricted Boltzmann Machine
============================

[![Build Status](https://travis-ci.org/aeyakovenko/rbm.svg?branch=master)](https://travis-ci.org/aeyakovenko/rbm)

Implementation of RBM training from [Hinton's paper](docs/hinton_rbm_guide.pdf?raw=true), and [Back Propagation](docs/rojas-backprop.pdf?raw=true).

run `make mnist_data` to generate the test data
run `make mnist` to test the mnist training.

After backprop there is very strong correlation between the output and the labels

* 25k minibatches of 5 at 0.01 learning rate

label|      correlation
-----|-----------------
   0 | 0.96931061006877  
   1 | 0.97586250662331  
   2 | 0.92920411708058  
   3 | 0.93181485208514  
   4 | 0.92997917370580  
   5 | 0.86159049963061  
   6 | 0.95335043333702  
   7 | 0.94365052624500  
   8 | 0.91446780721296  
   9 | 0.86919757980633  

* 250k minibatches of 5 at 0.01 learning rate

label|      correlation
-----|-----------------
   0 | 0.99341336376690
   1 | 0.99060696767070
   2 | 0.98038157977265
   3 | 0.98314542811599
   4 | 0.97051993597869
   5 | 0.97578985146789
   6 | 0.98018183041991
   7 | 0.97570483598546
   8 | 0.96970036917824
   9 | 0.97368923077333

As you can see, after 250k minibatches there is a significant improvement in digit recognition.

Monitoring Progress
-------------------

First layer of weights should approximate the input we are training on.  It can be seen here (its a large gif, so it takes a few seconds to load)

![dist/rbm1.gif](results/rbm1.gif?raw=true)

Second layer picks up some of the features of the first layer.

![dist/rbm1.gif](results/rbm2.gif?raw=true)

Thrid layer. My guess is that the box on the top left is related to the bias nodes.

![dist/rbm1.gif](results/rbm3.gif?raw=true)

For backprop generated the output of the RBM run backwards after backprop training the classes.  The gif represents about 250k minibatches of 5 images at 0.01 learning rate.   The network slowly converges on what looks like the numbers its trying to classify.

![dist/bp1.gif](results/bp13.gif?raw=true)
