#pragma once

#include <vector>

#include "matrix.cpp"
#include "layer.cpp"
#include "activation.cpp"

/**
 * Network - a general model framework
 */ 
class Network {
  typedef float val_t;
  typedef unsigned dim_t;

  std::vector<Layer<val_t> > layers;
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
    if (layers.size()) prev = layers[layers.size()-1].out_size();
    layers.push_back(Layer<val_t>(prev, out_dim));
  }

  /**
   * Train the network
   * @param epoch_size How many iterations to train through
   */
  void train(const Matrix<val_t> &inp, Matrix<val_t> exp)
  {
    std::vector<Matrix<val_t> > snapshots;
    snapshots.push_back(inp);

    for (const Layer<val_t> &layer : layers)
      snapshots.push_back(layer.feed(snapshots[snapshots.size()-1]));
    snapshots.push_back(exp);

    // std::vector<val_t>::reverse_iterator rit = snapshots.rbegin();
    // TODO: BACKPROP AAAAA
  }
};