Restricted Boltzmann Machine
============================

Simple single layer implementation in haskell.  Take a look at [RBM/List.hs](RBM/List.hs).  

Repa based single layer implementation [RBM/Repa.hs](RBM/Repa.hs).

Multi layer implmentation [DBN/Repa.sh](DBN/Repa.hs).

run `make mnist` to test the mnist training.

Todo
----

* gpu based one using accelrate 
* console to monitor the recustructions, change rate etc..

experiments
-----------

* use hintons paper for a learning rate recipe

so the learning rate should keep the update size to 0.001 of the weights, (sum weights)/(sum update) * (0.001)

* use reconstruction error intead of update amount

how much we update the weights is a terrible indicator of how close we are to reconstructin the image

* train many mini batches at once

seems to overload the training.  i think this is due to the common features between images getting the most recognized.  no matter how i set the learning rate, the result ends up overtraining the layer and everything is 0.

do i need to constaintly randomize the mini batches?

Size of minibatches should be no more then the number of classes we are learning, and should contain one from every class.

* set the learning rate 10x lower, to 0.001

layer 1 is slowly moving towards a smaller error (from seconds to many minutes for a single minibatch), much much slower then with a rate of 0.01

has a hard time to get over the 0.50 mse mark

