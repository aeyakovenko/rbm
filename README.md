Restricted Boltzmann Machine
============================

Implementation of [Hinton's paper](docs/hinton_rbm_guide.pdf?raw=true) on rbms, and [Back Propagation](docs/rojas-backprop.pdf?raw=true).

run `make mnist_data` to generate the test data
run `make mnist` to test the mnist training.

After backprop there is very strong correlation between the output and the labels

label | correlation coeffecient after 25k/5
-------------------------------------------
|   0 |                  0.96931061006877 |
|   1 |                  0.97586250662331 |
|   2 |                  0.92920411708058 |
|   3 |                  0.93181485208514 |
|   4 |                  0.92997917370580 |
|   5 |                  0.86159049963061 |
|   6 |                  0.95335043333702 |
|   7 |                  0.94365052624500 |
|   8 |                  0.91446780721296 |
|   9 |                  0.86919757980633 |
-------------------------------------------

label | correlation coeffecient after 125k/5
-------------------------------------------
|   0 |                  0.99044301435635 |
|   1 |                  0.98928921973885 |
|   2 |                  0.97034648760490 |
|   3 |                  0.97159955796703 |
|   4 |                  0.97142776042971 |
|   5 |                  0.95405431999714 |
|   6 |                  0.97616215725713 |
|   7 |                  0.96718188691733 |
|   8 |                  0.96738713964942 |
|   9 |                  0.96226386748393 |
-------------------------------------------


So its pretty good at picking the right number.

Monitoring Progress
-------------------

First layer of weights should approximate the input we are training on.  It can be seen here 

![dist/rbm1.gif](results/rbm1.gif?raw=true)

Second layer looks interesting, but not sure how to interpret it 

![dist/rbm1.gif](results/rbm2.gif?raw=true)

Thrid layer looks like it might be picking actual classes

![dist/rbm1.gif](results/rbm3.gif?raw=true)

For backprop generated the output of the RBM run backwards after backprop training the classes.  Each gif represents 25k minibatches of 5 images.  

![dist/bp1.gif](results/bp1.gif?raw=true)
![dist/bp2.gif](results/bp2.gif?raw=true)
![dist/bp3.gif](results/bp3.gif?raw=true)
![dist/bp4.gif](results/bp4.gif?raw=true)
![dist/bp5.gif](results/bp5.gif?raw=true)

