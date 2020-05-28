# neural_net_cpp

**A super simple neural net, from scratch, in C++.**

## Current Status

I never finished converting the code to work with CUDA, so that remains a project for another date.

~~Part of the way through transitioning to paralell processing, I realized that working with cross device memory efficiently is difficult.
I am currently researching existing strategies for optimization, such as in place matrix transposition (which is surprisingly difficult) and otherwise reconsidering life choices.~~

~~I've also been working on some side projects, so this one is somewhat on hold for the moment. Until then!~~

## Summary

This project was inspired by [iamtrask's toy neural net blogpost](https://iamtrask.github.io/2015/07/12/basic-python-network/). I played with it a few years back and learned alot from that, so I figured I would try to implement the same thing in C++ because it's speedy. Then, I would like to CUDA-fy it so I can run something on my gpu and feel cool (:. I'm aware that cognitivedemons has done this before both in [C++](https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/) and with [CUDA](https://cognitivedemons.wordpress.com/2017/09/02/a-neural-network-in-10-lines-of-cuda-c-code/), but I would like to try it out for myself. This should be a fun, managable, and educational project and I'm glad you decided to check it out! Let's goo!

### To run this project

**To run this project, clone the repo and run `make`.**

(I am also experimenting with a new git workflow for this project, so my commits will strive to make sense.)

## Plan

Here's the master plan for this project. It is divided into stages:

### Stage 1 - OOP Framework (Complete)

I plan to write this decently well, and from scratch, which means that I will need to create some classes. At the moment, I'm thinking of writing a `Matrix` class that can implement all the math, a `Layer` class that acts implements a fully connected layer with methods like `init` and `activate` and `update` (for backprop), and finally a `Network` class that handles all the interfacing. Specifically, I plan to have the following basic functionality for each class:

- [x] `Matrix`: Mathamatical workhorse
  - [x] Primary constructor `Matrix(vector<vector<long double> > &src)` until I figure out how to make it take initializer lists
  - [x] Transpose utility `Matrix matrix.transpose() const -> copy`
  - [x] Dot multiplication utility `Matrix matrix.dot(const Matrix &other) const -> copy`
  - [x] Exponent utility `Matrix matrix.exp() const -> copy`
- [x] `Layer`: Abstract layer-wise math
  - [x] Primary constructor `Layer(const int size)`
  - [x] Implement `Matrix layer.feed(const Matrix &activations) -> activations`
  - [x] Implement `Matrix layer.backprop(const Matrix &inp, const Matrix &out, const Matrix &err) -> delta` which will update the weights of that layer using backprop
- [x] `Network`: Interface for the other code
  - [x] Primary constructor `Network(const vector<int> &layer_sizes)`
  - [x] Single step wrapper `const Matrix& network.feed(const Matrix &input) -> output`
  - [x] Train wrapper (feed with backprop) `void network.train(const Matrix &input, const Matrix &expected, const int epochs)`

### Stage 2 - Basic Functionality (Complete)

With all the base code written, I plan to write a simple `int main()` driver that mimics the functionality of Part 2 of [iamtrask's original post](https://iamtrask.github.io/2015/07/12/basic-python-network/). If the Stage 1 was written well, then this should go smoothly.

### Stage 3 - CUDA Madness (On Hold)

Having a working CPU neural net in Python, I can finally complete the dream by converting my framework from Stage 1 to work with CUDA.

I am working on this step now, but I don't know how exactly I should go about assigning operations to blocks and threads, because there is so much data manipulation involved that data transfer will probably be the main bottleneck. I need to think about how to paralellize operations without so much data movement.

### Next Steps

- Fewer segfaults. Get the thing to actually work.
- Cuda!
- More activation functions
- Rewrite in rust so that I don't have to deal with memory leaks.

