Restricted Boltzmann Machine
============================

Implementation of [Hinton's paper](docs/hinton_rbm_guide.pdf) on rbms, and [Back Propagation](docs/rojas-backprop.pdf).

run `make mnist_data` to generate the test data
run `make mnist` to test the mnist training.

Monitoring Progress
-------------------

Open dist/rbm[1-3].gif.

These generated files contain what the 
open dist/bp[1-3].gif

experiments
-----------
* error and reconstruction is hard to observe

basically, any small shift in the data could cause the generated image look correct, but the actual error rate to be high and useless.

* learning rate recipe from Hinton's paper

so the learning rate should keep the update size to 0.001 of the weights, (sum weights)/(sum update) * (0.001)

* use reconstruction error intead of update amount as the stopping point

how much we update the weights is not a good indicator of how close we are to reconstructin the input

* initialize the weights to be +/- 0.01 of around 0

initial weights that are to big made the results unpredictable, and took to long to train

* mini batches should be no more then number of classes you are learning, and random

using larger batches doesn't train the rbm effectively.
