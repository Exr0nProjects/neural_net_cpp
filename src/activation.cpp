#include "matrix.cpp"

template <class val_t> class Activation
{
  string _type;
public:
  Activation ()
  {
    _type = "sigmoid";
  }
  Matrix* operator() ()
  {

  }
}