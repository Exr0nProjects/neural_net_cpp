#pragma once

#include "matrix.cpp"
#include "activation.cpp"

#include <string>
#include <cmath>
#include <random>
#include <exception>

/**
 * Layer : Abstracted math
 */
template<class val_t> class Layer
{
  typedef unsigned dim_t;
  Matrix<val_t> _syn;
  Activation<val_t> *_actv;
  dim_t _width, _height;
public:
  /**
   * Default constructor
   */
  Layer()
  {
    _syn = Matrix<val_t>(1, 1);
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
    printf("L basic layer constructor called: %d-%d\n", in_size, out_size);
    _height = in_size;
    _width = out_size;
    if (_width == 0) _width = _height;
    Matrix<val_t>::random(_syn, seed); // ! Some weird shenanigans were causing the return value of Matrix<val_t>::random to not get copied... so I had to do this instead
    printf("Syn dimensions: %dx%d\n", _syn.h(), _syn.w());
    _syn.print();
    _actv = new Activation<val_t>("sigmoid");
    printf("L syn: %d\n", &_syn); // DEBUG, TODO: REMOVE
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
    _syn = src.syn_raw();
    _actv = new Activation<val_t>(*(src.actv_raw()));
  }

  ~Layer()
  {
    delete _actv;
  }
  
  /* methods */
  // template<typename... Args> // https://stackoverflow.com/a/16338804
  // void forEach(std::function<void()> const& lambda, Args... args)

  // getters
  dim_t in_size() const {return _height;}
  dim_t out_size() const {return _width;}
  const Matrix<val_t> &syn_raw() const {return _syn;} 
  const Activation<val_t> *const actv_raw() const {return _actv;}

  // setters
  void update_raw(const Matrix<val_t> &mod)
  {
    if (mod.w() != _width || mod.h() != _height)
      throw std::domain_error("Invalid matrix dimensions to update layer!");
    for (dim_t i=0; i<_height; ++i)
      for (dim_t j=0; j<_width; ++j)
      {
        _syn.set(i, j, _syn.get(i, j)+mod.get(i, j));
      }
  }

  /**
   * Feed - feed forward through this layer
   * @param in Matrix* of height in_size
   * @return Matrix* of height out_size
   */
  Matrix<val_t> feed(const Matrix<val_t> *in) const
  {
    if (in->w() != _height)
      throw std::domain_error("Invalid input matrix width!");
    //const Matrix<val_t> *p = Matrix::random(in->h(), _width);
    Matrix<val_t> p = Matrix<val_t>::dot(in, _syn);
    p = (*_actv)(p);
    return p;
    //return nullptr;
  }
  Matrix<val_t> feed(const Matrix<val_t> &in) const
  {
    if (in.w() != _height)
      throw std::domain_error("Invalid input matrix width for network propogation!");
    //const Matrix<val_t> *p = Matrix::random(in->h(), _width);
    Matrix<val_t> p = Matrix<val_t>::dot(in, _syn);
    (*_actv)(p);
    return p;
    //return nullptr;
  }

  /**
   * Calculates and applies the weight changes asserted by backpropogation
   * 
   * @param inp Weights that were input to this layer during training
   * @param out Weights that this layer returned
   * @param err The error in the returned value `out`
   * @return The modifications made
   */
  Matrix<val_t> backprop(const Matrix<val_t> &inp, const Matrix<val_t> &out, const Matrix<val_t> &err)
  {
    if (inp.h() != err.h())
      throw std::domain_error("Invalid 'inp' matrix dimensions for back propogation!");
    if (out.w() != err.w() || out.h() != err.h())
      throw std::domain_error("Invalid 'out' or 'err' matrix dimensions for back propogation!");

    Matrix<val_t> delta = err * (_actv->deriv(out));
    printf("update - dot product:\n"); Matrix<val_t>::dot(Matrix<val_t>::transpose(inp), delta).print();
    //syn_raw().print();
    update_raw(Matrix<val_t>::dot(Matrix<val_t>::transpose(inp), delta));
    return delta;
  }
};