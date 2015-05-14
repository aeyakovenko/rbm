Restricted Boltzmann Machine
============================

Simple single layer implementation in haskell.  Take a look at [RBM/List.hs](RBM/List.hs).  

Working on a multi layer implementation using haskells parallel cpu array library.  Take a look at [RBM/Repa.hs](RBM/Repa.hs). The matrix/vector operations are unrolled so its easy to play around with repa's parallelization.

experiments
-----------

* train many mini batches at once

seems to overload the training.  i think this is due to the common features between images getting the most recognized.  no matter how i set the learning rate, the result ends up overtraining the layer and everything is 0.

do i need to constaintly randomize the mini batches?

* set the learning rate 10x lower, to 0.001

layer 1 is slowly moving towards a smaller error (from seconds to many minutes for a single minibatch), much much slower then with a rate of 0.01

has a hard time to get over the 0.50 mse mark
