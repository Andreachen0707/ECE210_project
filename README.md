# ECE210_project

# This code is for ECE 210 final project: "Digit classification on MNIST dataset"
## main.py
This py is the main entry to the software. By using model that is defined in network, you can train and tune various of variables

## solver,py
This py is the solver for different network. The main functions are solver.train, solver.check_accuracy, solver._step
solver.train do the training process. solver.check_accuracy gives out the validation accuracy and training accuracy. solver._step do the optimization process.

## network.py
All the networks that we wrote, including LR, SVM, SVM with kernel, FCNet and CNN. 
To be specific, SVM with kernel do not use regular gradient descent method, so the training part for SVM_Kernel do in this class.

In CNN, the fast version of forward and backward calculation using cross-platform programming (C & Pytohn) is credited by Standford CS231n, as our own function took too much time to calculate.

## optims.py
All the optimization methods that we use, including SGD, SGD momentum and Adam are defended here.

##layers.py and layer_utils.py
Our version of different layer definations and gradient calculation.