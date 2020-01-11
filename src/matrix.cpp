#pragma once

#include <vector>
#include <cmath>
#include <iomanip>

/**
 * Matrix: Mathamatical Workhorse
 * Designed to do all of the heavy lifting math wise for the network
 * Will be modified to work with CUDA later
 */
template<class val_t> class Matrix
{
  typedef unsigned dim_t;
  val_t **_data;
  dim_t _height;
  dim_t _width;
  int _id;
public:

  /* static */
  static void random(Matrix &dest, const unsigned seed=1)
  {
    std::srand(seed);
    for (int i=0; i<dest.h(); ++i)
      for (int j=0; j<dest.w(); ++j)
        dest.set(i, j, (val_t)std::rand() / RAND_MAX * 2 - 1); // value between 0 and 1
  }
  static Matrix random(const dim_t height, const dim_t width, const unsigned seed = 1)
  {
    std::srand(seed);
    Matrix ret(height, width);

    printf("creating random matrix %d\n", &ret);
    Matrix<val_t>::random(ret, seed);
    printf("random matrix initialized: %d\n", &ret);
    return ret;
  }

  /**
   * Return a pointer to a new matrix that is the transposition of this matrix
   * @param src The source matrix (unmodified)
   * @return Matrix* The transposed matrix
   */
  static Matrix transpose(const Matrix *const src)
  {
    Matrix ret(src->w(), src->h());
    for (dim_t h = 0; h < src->h(); ++h)
      for (dim_t w = 0; w < src->w(); ++w)
        ret.set(w, h, src->get(h, w));
    return ret;
  }
  static Matrix transpose(const Matrix &src)
  {
    Matrix ret(src.w(), src.h());
    for (dim_t h = 0; h < src.h(); ++h)
      for (dim_t w = 0; w < src.w(); ++w)
        ret.set(w, h, src.get(h, w));
    return ret;
  }

  /**
   * Return a pointer to a new matrix that is this matrix multiplied by another
   * @param lhs Matrix to be multiplied on
   * @param rhs Matrix to be multiplied by
   * @return Matrix* The product matrix
   */
  static Matrix dot(const Matrix *const lhs, const Matrix *const rhs)
  {
    if (lhs->w() != rhs->h())
      throw std::domain_error("Invalid matrix dimensions for static dot multiplication!");
    Matrix ret(lhs->h(), rhs->w());
    for (dim_t r = 0; r < lhs->h(); ++r)
    {
      for (dim_t c = 0; c < rhs->w(); ++c)
      {
        val_t sum = 0;
        for (dim_t i = 0; i < lhs->w(); ++i)
        {
          sum += lhs->get(r, i) * rhs->get(i, c);
        }
        ret.set(r, c, sum);
      }
    }
    return ret;
  }
  static Matrix dot(const Matrix &lhs, const Matrix &rhs)
  {
    if (lhs.w() != rhs.h())
      throw std::domain_error("Invalid matrix dimensions for static dot multiplication!");
    Matrix ret(lhs.h(), rhs.w());
    for (dim_t r = 0; r < lhs.h(); ++r)
    {
      for (dim_t c = 0; c < rhs.w(); ++c)
      {
        val_t sum = 0;
        for (dim_t i = 0; i < lhs.w(); ++i)
        {
          sum += lhs.get(r, i) * rhs.get(i, c);
        }
        ret.set(r, c, sum);
      }
    }
    return ret;
  }

  /* constructors/destructors */
  Matrix(){
    printf("Empty Matrix constructor called! %d\n", this);
    _data = new val_t*;
    *_data = new val_t;
    _height = 0;
    _width = 0;
    _id = rand();
  }

  /**
   * Empty Initializer Constructor - Create a new emtpy Matrix of w by h
   * @param height Height of the new Matrix
   * @param width Width of the new Matrix
   */
  Matrix(const dim_t height, const dim_t width) : _height(height), _width(width)
  {
    printf("Empty matrix init: %d\n", this);
    _data = new val_t*[_height];
    for (dim_t i=0; i<_height; ++i)
      _data[i] = new val_t[_width];
  }

  /**
   * Initalizer Constructor - Create a Matrix from a 2D vector
   * @param src 2D vector to copy from
   */
  Matrix(const std::vector<std::vector<val_t> > &src): Matrix(src.size(), src[0].size())
  {
    // copy data
    for (dim_t i=0; i<_height; ++i)
    {
      // check shape
      if (src[i].size() != _width)
        throw std::domain_error("Non rectangular vector input!");
      // copy row
      for (dim_t j=0; j<_width; ++j)
        set(i, j, src[i][j]);
    }
  }

  /**
   * Copy Constructor - Copy a matrix
   * @param src Matrix to copy from
   */
  Matrix(const Matrix &src): Matrix(src.height(), src.width())
  {
    printf("\n\n\n\n\n\n\n\n\nsuck.\n");
    for (dim_t i=0; i<_height; ++i)
      for (dim_t j=0; j<_width; ++j)
        set(i, j, src.get(i, j));
  }

  ~Matrix()
  {
    printf("~ deleting Matrix %d\n", this);
    if (_data == nullptr) return;
    for (dim_t i=0; i<_height; ++i)
    {
      if (_data[i] != nullptr)
      {
        delete [] _data[i];
      }
    }
    delete [] _data;
    printf("~ finished deleting.\n");
  }

  /* methods */
  // Matrix &operator=(const Matrix &o)
  // { // TODO: rewrite for efficiency? https://docs.microsoft.com/en-us/archive/msdn-magazine/2005/september/c-at-work-copy-constructors-assignment-operators-and-more
  //   this = new Matrix(o);
  //   return *this;
  // }

  /**
   * Prints matrix to stdout
   * @param precision Sets precision of output, number of sigfigs
   */
  void print(const unsigned precision = 3) const
  {
    printf("printing Matrix %d\n", this);
    std::cout << std::setprecision(precision);
    for (int i = 0; i < _height; ++i)
    {
      for (int j = 0; j < _width; ++j)
        std::cout << " " << get(i, j);
      std::cout << std::endl;
    }
    printf("\n");
  }

  /**
   * Returns the width of this matrix
   * @return dim_t The width of the matrix
   */
  inline dim_t width() const { return _width; };
  inline dim_t w() const { return width(); };

  /**
   * Returns the height of this matrix
   * @return dim_t The height of the matrix
   */
  inline dim_t height() const { return _height; };
  inline dim_t h() const { return height(); };

  /**
   * Get the value in the matrix at (x, y)
   * @param y The row of the element to be retrieved
   * @param x The column of the element to be retrieved
   * @return val_t The value of the element at that position
   */
  inline val_t get(const dim_t y, const dim_t x) const { return _data[y][x]; }
  /**
   * Set the value in the matrix at (x, y)
   * @param y The row of the element to be set
   * @param x The column of the element to be set
   * @param dat The data to be copied into that position
   */
  inline void set(const dim_t y, const dim_t x, const val_t &dat)
  {
    _data[y][x] = dat;
  }

  void forEach(std::function<void(dim_t,dim_t,Matrix<val_t>*const,Matrix<val_t>*const)> const& lambda)
  {
    for (dim_t i=0; i<h(); ++i)
      for (dim_t j=0; j<w(); ++j)
      lambda(i, j, this);
  }

  /**
   * Operator overloads
   */
  Matrix<val_t> operator-(const Matrix<val_t> &o) const
  {
    if (w() != o.w() || h() != o.h())
      throw std::domain_error("Invalid matrix dimesions for element-wise subtract!");
    Matrix ret(h(), w());
    for (dim_t i=0; i<h(); ++i)
      for (dim_t j=0; j<w(); ++j)
        ret.set(i, j, get(i, j)-o.get(i, j));
    // printf("Address inside fxn: %d\n", &ret);
    return ret;
    // const auto op = [](const dim_t i, const dim_t j, Matrix<val_t> *l, const Matrix<val_t> *r){l->set(i, j, r->get(i, j));};
    // return Matrix(h(), w()).forEach(op);
  }
  Matrix<val_t> operator*(const Matrix<val_t> &o) const
  {
    // printf("got to operator*\n");
    if (w() != o.w() || h() != o.h())
      throw std::domain_error("Invalid matrix dimesions for element-wise multiply!");
    Matrix ret(h(), w());
    for (dim_t i=0; i<h(); ++i)
      for (dim_t j=0; j<w(); ++j)
        ret.set(i, j, get(i, j)*o.get(i, j));
    // printf("Address inside fxn: %d\n", &ret);
    return ret;
    // const auto op = [](const dim_t i, const dim_t j, Matrix<val_t> *l, const Matrix<val_t> *r){l->set(i, j, r->get(i, j));};
    // return Matrix(h(), w()).forEach(op);
  }

  /**
   * Return a pointer to a new matrix that is the transposition of this matrix
   * @return Matrix* The transposed matrix
   * @deprecated
   */
  Matrix &transpose() const
  {
    for (dim_t h=0; h<_height; ++h)
      for (dim_t w=h+1; w<_width; ++w)
      {
        val_t t = get(h, w);
        set(h, w, get(w, h));
        set(w, h, t);
      }
    return *this;
  }

  /**
   * Take the `e`th power of each value in the matrix
   * @return Matrix* The exponentiated matrix
   * @deprecated
   */
  Matrix<val_t> exp() const
  {
    Matrix ret(_height, _width);
    for (dim_t i=0; i<_height; ++i)
      for (dim_t j=0; j<_width; ++j)
      {
        ret.set(i, j, (val_t) std::exp(get(i, j)));
      }
    return ret;
  }
};