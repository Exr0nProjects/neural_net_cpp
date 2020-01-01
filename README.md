# neural_net_cpp

**A super simple neural net, from scratch, in C++.**

## Summary

This project was inspired by [iamtrask's toy neural net blogpost](https://iamtrask.github.io/2015/07/12/basic-python-network/). I played with it a few years back and learned alot from that, so I figured I would try to implement the same thing in C++ because it's speedy. Then, I would like to CUDA-fy it so I can run something on my gpu and feel cool (:. I'm aware that cognitivedemons has done this before both in [C++](https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/) and with [CUDA](https://cognitivedemons.wordpress.com/2017/09/02/a-neural-network-in-10-lines-of-cuda-c-code/), but I would like to try it out for myself. This should be a fun, managable, and educational project and I'm glad you decided to check it out! Let's goo!

(I am also experimenting with a new git workflow for this project, so my commits will strive to make sense.)

## Plan

Here's the master plan for this project. It is divided into stages:

### Stage 1 - OOP Framework

I plan to write this decently well, and from scratch, which means that I will need to create some classes. At the moment, I'm thinking of writing a `Matrix` class that can implement all the math, a `Layer` class that acts implements a fully connected layer with methods like `init` and `activate` and `update` (for backprop), and finally a `Network` class that handles all the interfacing. Specifically, I plan to have the following basic functionality for each class:

- [ ] `Matrix`: Mathamatical workhorse
  - [x] Primary constructor `Matrix(vector<vector<int> > &src)` until I figure out how to make it take initializer lists
  - [ ] Transpose utility `const Matrix& matrix.transpose() const -> copy`
  - [ ] Dot multiplication utility `const Matrix& matrix.dot(const Matrix &other) const -> copy`
  - [ ] Exponent utility `const Matrix& matrix.exp() const -> copy`
- [ ] `Layer`: Abstract layer-wise math
  - [ ] Primary constructor `Layer(const int size)`
  - [ ] Implement `const Matrix& layer.feed(const Matrix &activations) -> activations`
  - [ ] Implement `const Matrix& layer.backprop(const Matrix &target) -> delta` which is to be used on the final layer of the network
  - [ ] Implement `const Matrix& layer.backprop(const Layer &previous) -> delta` which is to be used for intermediate layers
- [ ] `Network`: Interface for the other code
  - [ ] Primary constructor `Network(const vector<int> &layer_sizes)`
  - [ ] Single step wrapper `const Matrix& network.feed(const Matrix &input) -> output`
  - [ ] Backprop wrapper `void network.backprop(const Matrix &output, const Matrix &expected)`
  - [ ] Feed wrapper (combines single step and backprop) `void network.feed(const Matrix &input, const Matrix &expected)`
  - [ ] Train wrapper (loops feed wrapper) `void network.train(const Matrix &input, const Matrix &expected, const int epochs)`

### Stage 2 - Basic Functionality

With all the base code written, I plan to write a simple `int main()` driver that mimics the functionality of Part 2 of [iamtrask's original post](https://iamtrask.github.io/2015/07/12/basic-python-network/). If the Stage 1 was written well, then this should go smoothly

### Stage 3 - CUDA Madness

Having a working CPU neural net in Python, I can finally complete the dream by converting my framework from Stage 1 to work with CUDA. This will probably be a pain, so I will update this section of the README later.

All this being said, I will start fully considering the tasks that each class must perform and what needs to be implementd. I will update this README soon...
