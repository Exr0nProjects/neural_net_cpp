#include "matrix.cpp"

template <class val_t> class Activation
{
  typedef unsigned dim_t;
  std::string _type;
public:
  Activation ()
  {
    _type = "sigmoid";
  }

  Matrix<val_t>* operator() (const Matrix<val_t> *in) const
  {
    printf("Runing through activation fxn...\n");
    Matrix<val_t>* ret = new Matrix<val_t>(in->h(), in->w());
    for (dim_t i=0; i<in->h(); ++i)
      for (dim_t j=0; j<in->w(); ++j)
      {
        val_t t = NULL;
        if (_type == "sigmoid")
        {
          t = 1 / (1 + (std::exp(in->get(i, j))));
        }
        ret->set(i, j, t);
      }
    return ret;
  }

  Matrix<val_t>* deriv(const Matrix<val_t> *in) const
  {
    Matrix<val_t> ret = new Matrix<val_t>(in->h(), in->w());
    for (dim_t i=0; i<in->h(); ++i)
      for (dim_t j=0; j<in->w(); ++j)
      {
        val_t t = NULL;
        if (_type == "sigmoid")
        {
          t = in->get(i, j) * (1-in->get(i, j));
        }
        ret->set(i, j, t);
      }
    return ret;
  }
};