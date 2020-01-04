#pragma once

#include "matrix.cpp"
#include "activation.cpp"

#include <string>
#include <cmath>
#include <random>

/**
 * Layer : Abstracted math
 */
template<class val_t> class Layer
{
  typedef unsigned dim_t;
  Matrix<val_t> *_syn;
  Activation<val_t> *_actv;
  dim_t _width, _height;
public:
  /**
   * Default constructor
   */
  Layer()
  {
    _syn = nullptr;
    _actv = new Activation<val_t>("sigmoid");
    _width = 0;
    _height = 0;
  }
  /**
   * "Basic" constructor
   * @param in_size Dimension of input to this layer
   * @param out_size Dimension of expected output (defaults to in_size)
   */
  Layer(const dim_t in_size, const dim_t out_size=0, const unsigned seed=1)
  {
    _width = in_size;
    _height = out_size;
    if (_height < 0) _height = _width;
    _syn = Matrix<val_t>::random(_width, _height, seed);
  }
  /**
   * "Activation" constructor
   * @param in_size Dimension of input to this layer
   * @param out_size Dimension of expected output
   * @param activation Activation function to use
   */
  Layer(const dim_t in_size, const dim_t out_size, const std::string activation, const unsigned seed=1): Layer(in_size, out_size, seed)
  {
    _actv = new Activation<val_t>(activation);
  }

  /**
   * Copy constructor
   * @param src The Layer to copy from
   */
  Layer(const Layer &src): Layer(src.in_size(), src.out_size())
  {
    _syn = new Matrix<val_t>(*(src.syn_raw()));
    _actv = new Activation<val_t>(*(src.actv_raw()));
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
  const Activation<val_t> *const actv_raw() const {return _actv;}

  /**
   * Feed - feed forward through this layer
   * @param in Matrix* of height in_size
   * @return Matrix* of height out_size
   */
  Matrix<val_t> *feed(const Matrix<val_t> *in) const
  {
    
  }
};