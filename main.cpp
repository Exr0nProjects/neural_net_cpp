/**
 * A toy neural net in C++
 * @author Exr0n
 * @version 0.0.1
 */

#include <iostream>
#include <vector>

#pragma once
#include "src/matrix.cpp"
#include "src/activation.cpp"
#include "src/layer.cpp"
#include "src/network.cpp"

// Utility function to input matricies
    template <class T>
Matrix<T> matrixIn()
{
    T h, w;
    std::cin >> h >> w;
    Matrix<T> ret(h, w);

    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
        {
            T d;
            std::cin >> d;
            ret.set(i, j, d);
        }
    return ret;
}

int main(const int argc, char **argv)
{
    // open files for i/o if passed as arguments
    if (argc >= 1)
        freopen(argv[1], "r", stdin);
    if (argc >= 2)
        freopen(argv[2], "w+", stdout);

    typedef float val_t;                        // use floats for all math
    // input training and testing data
    Matrix<val_t> inp = matrixIn<val_t>();
    Matrix<val_t> expected = matrixIn<val_t>();
    Matrix<val_t> validation = matrixIn<val_t>();

    const int CYCLES = 500;                   // number of epochs
    const int UPDATES = 50;                    // how often to refresh the progress bar

    std::cout << "Creating network..." << std::endl;
    Network net(inp.w());                       // create the network with the same input width as the input data
    net.addLayer(4);                            // add a hidden layer with 3 nodes
    net.addLayer(1);                            // add a hidden layer with 1 node, this is also the output layer

    srand(10);

    std::cout << "Training network..." << std::endl;
    net.train(inp, expected, CYCLES, UPDATES);  // train the network

    net.print();

    std::cout << "Training completed. Testing network..." << std::endl;
    std::cout << "Test data:" << std::endl;
    validation.print();                         // print the test data
    std::cout << "Test result:" << std::endl;
    net.feed(validation).print();               // feed the test data through the network and print the result

    return 0;
}
