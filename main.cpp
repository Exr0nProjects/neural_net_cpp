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
    if (argc >= 1)
        freopen(argv[1], "r", stdin);
    if (argc >= 2)
        freopen(argv[2], "w+", stdout);

    typedef float val_t;
    Matrix<val_t> inp = matrixIn<val_t>();
    Matrix<val_t> expected = matrixIn<val_t>();
	Matrix<val_t> validation = matrixIn<val_t>();
    //printf("expected dims: %dx%d\n", expected.h(), expected.w());

    const int CYCLES = 50000;
    const int UPDATES = 500;

    printf("Creating network...\n\n");
    Network net(inp.w());
    net.addLayer(3);
    net.addLayer(1);

    /*
  Notes:
  With network 19, 5, 3, 1 and an epoch size of 100K,
    the training sometimes works and sometimes the
    error goes to nan or 0.5 after about 30%. Is this
    the vanishing or exploding gradient problem?
  */
    printf("Training network.....\n");
    net.train(inp, expected, CYCLES, UPDATES);

	printf("Training completed. Testing network...\n");
	printf("Test data:\n");
	validation.print();
	printf("Test result:\n");
	net.feed(validation).print();

    return 0;
}
