#pragma once

#include <vector>

#include "matrix.cpp"
#include "layer.cpp"
#include "activation.cpp"
#include "utility.cpp"

/**
 * Network - a general model framework
 */
class Network
{
    typedef float val_t;
    typedef unsigned dim_t;

    std::vector<Layer<val_t>> layers;
    unsigned _epoch_size;
    unsigned _status_log_count;
    dim_t _input_dim;

public:
    /**
   * Default constructor
   */
    Network() {}
    /**
   * "Normal" constructor
   * @param input_dim Dimensionality of the input vector
   */
    Network(dim_t input_dim)
    {
        _input_dim = input_dim;
    }

    /**
   * Adds a layer to the model
   * @param out_dim Dimensionality of the output of this layer
   */
    void addLayer(dim_t out_dim)
    {
        dim_t prev = _input_dim;
        if (layers.size())
            prev = layers[layers.size() - 1].out_size();
        Layer<val_t> nxt(prev, out_dim);
        layers.push_back(nxt);
	}

    /**
   * Feed an input vector through the network to check the result
   */
    Matrix<val_t> feed(Matrix<val_t> inp)
    {
        for (const Layer<val_t> &layer : layers)
        {
            inp = layer.feed(inp);
        }
        return inp;
    }

    /**
   * Train the network
   * @param epoch_size How many iterations to train through
   */
    void train(const Matrix<val_t> &inp, Matrix<val_t> exp, const unsigned epoch = 500000, const unsigned updates = 100)
    {
        printf("\n\n");
        for (unsigned i = 0; i < epoch; ++i)
        {
            std::vector<Matrix<val_t>> snapshots;
            snapshots.push_back(inp);

            // Feed forward
            for (const Layer<val_t> &layer : layers)
            {
                snapshots.push_back(layer.feed(snapshots[snapshots.size() - 1]));
            }

            Matrix<val_t> error = exp;
            error -= snapshots[snapshots.size() - 1];

            if (i % (epoch / 100) == 0)
            {
                // printf("Input:\n");
                // inp.print();
                // printf("Output:\n");
                // snapshots[snapshots.size()-1].print();
                // printf("Exepected:\n");
                // exp.print();

                progressBar(50, (double)i / epoch, 2, "=", " ", 10);

                val_t average_error = 0;
                for (int i = 0; i < error.h(); ++i)
                    average_error += abs(error.get(i, 0));
                average_error /= exp.h() * exp.w();
                printf("error = %.5f\n", average_error);
            }

            //printf("starting backprop...\n\n");

            // Backprop
            for (int i = layers.size() - 1; i >= 0; --i)
            {
                Matrix<val_t> delta = layers[i].backprop(snapshots[i], snapshots[i + 1], error);  //  TODO: don't  copy exp then copy back, just use the same memory
                Matrix<val_t> synT = Matrix<val_t>::transpose(layers[i].syn_raw());
				//printf("    got syn transpose\n");
            }
        }
    }
};
