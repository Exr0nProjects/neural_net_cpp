#pragma once

#include <vector>

#include "matrix.cpp"
#include "layer.cpp"
#include "activation.cpp"

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
    printf("N pre: %d->%d\n", prev, out_dim);
    Layer<val_t> nxt(prev, out_dim);
    printf("N post\n");
    layers.push_back(nxt);
    printf("N post2\n");

    printf("N layers: %d\n", layers.size());
    printf("N nxt: %d\n", &nxt);
  }

  /**
   * Train the network
   * @param epoch_size How many iterations to train through
   */
  void train(const Matrix<val_t> &inp, Matrix<val_t> exp, const unsigned epoch = 500000)
  {
    for (unsigned i = 0; i < epoch; ++i)
    {
      std::vector<Matrix<val_t>> snapshots;
      snapshots.push_back(inp);

      // Feed forward
      for (const Layer<val_t> &layer : layers)
        snapshots.push_back(layer.feed(snapshots[snapshots.size() - 1]));

      Matrix<val_t> error = exp - snapshots[snapshots.size() - 1];

      if (i % (epoch / 100) == 0)
      {
        // printf("Input:\n");
        // inp.print();
        // printf("Output:\n");
        // snapshots[snapshots.size()-2].print();
        // printf("Exepected:\n");
        // exp.print();

        // val_t average_error = 0;
        // for (int i = 0; i < error.h(); ++i)
        //   average_error += abs(error.get(i, 0));
        // average_error /= error.h();
        // printf("\n%d%% progress - error = %.5f\n\n----------\n", i * 100 / epoch, average_error);
      }

      // Backprop
      for (unsigned i = layers.size(); i > 0; --i)
      {
        printf("inp:\n");
        snapshots[i-1].print();
        printf("out:\n");
        snapshots[i].print();
        printf("err:\n");
        error.print();
        error = Matrix<val_t>::dot(
            layers[i].backprop(snapshots[i-1], snapshots[i], error),
            Matrix<val_t>::transpose(layers[i].syn_raw()));
      }
    }
  }
};