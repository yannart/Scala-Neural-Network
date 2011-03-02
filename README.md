Neural Network
====

An implementation in Scala written by Yann Nicolas

This Scala implementation of a neural network, allows the execution and the training of a multi-layer perceptron.


Requirements
------------

To build and run the XOR example you need [Simple Build Tool][sbt] (sbt).


Running
-------

To compile and run the XOR example use 'sbt run'. 

- First, an new Perceptron will be created with two hidden layers, one of 5 neurons and another of 10 neurons.
- Then, 100 iteration will be used to train the perceptron to be able to solve XOR operations
- The perceptron configuration will be saved in the XOR.xml file

- The perceptron will be loaded from the XOR.xml file
- XOR operations will be run to verify the perceptron outputs


Notice
------

The use and distribution terms for this software are covered by the
Apache Software Foundation License 2.0 ([http://www.apache.org/licenses/LICENSE-2.0][apa])

[sbt]: http://code.google.com/p/simple-build-tool/
[apa]: http://www.apache.org/licenses/LICENSE-2.0