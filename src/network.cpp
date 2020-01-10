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
public:
  /**
   * Default constructor
   */
  Network () {
    _epoch_size = 500000;
    _status_log_count = 100;
  }
}