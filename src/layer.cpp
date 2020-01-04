#pragma once

#include "matrix.cpp"
#include "activation.cpp"

#include <string>

/**
 * Layer : Abstracted math
 */
template<class val_t> class Layer
{
  typedef unsigned dim_t;
  Matrix<val_t> *_syn;
  std::string _actv;
  dim_t _width, _height;
public:
  /**
   * Default constructor
   */
  Layer()
  {
    _syn = nullptr;
    _actv = "sigmoid";
    _width = 0;
    _height = 0;
  }
  /**
   * "Basic" constructor
   * @param in_size Dimension of input to this layer
   * @param out_size Dimension of expected output (defaults to in_size)
   */
  Layer(const dim_t in_size, const dim_t out_size=0)
  {
    _width = in_size;
    _height = out_size;
    if (_height < 0) _height = _width;
    _syn = new Matrix<val_t>(_width, _height);
  }
  /**
   * "Activation" constructor
   * @param in_size Dimension of input to this layer
   * @param out_size Dimension of expected output
   * @param activation Activation function to use
   */
  Layer(const dim_t in_size, const dim_t out_size, const std::string activation): Layer(in_size, out_size)
  {
    _actv = activation;
  }

  /**
   * Copy constructor
   * @param src The Layer to copy from
   */
  Layer(const Layer &src): Layer(src.in_size(), src.out_size())
  {
    _syn = new Matrix<val_t>(*(src.syn_raw()));
  }

  ~Layer()
  {
    delete _syn;
  }
  
  /* methods */
  // getters
  dim_t in_size() const {return _width;}
  dim_t out_size() const {return _height;}
  const Matrix<val_t> *const syn_raw() const {return _syn;} 
};