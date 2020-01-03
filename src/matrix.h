#pragma once

#include <vector>
#include <cmath>

/**
 * Matrix: Mathamatical Workhorse
 * Designed to do all of the heavy lifting math wise for the network
 * Will be modified to work with CUDA later
 */
template<class val_t> class Matrix
{
  typedef unsigned int dim_t;
  val_t **_data = nullptr;
  dim_t _height = 0;
  dim_t _width = 0;
  int _id = rand();
public:

  /* constructors/destructors */
  Matrix() {}

  /**
   * Empty Initializer Constructor - Create a new emtpy Matrix of w by h
   * @param height Height of the new Matrix
   * @param width Width of the new Matrix
   */
  Matrix(const dim_t height, const dim_t width): _height(height), _width(width)
  {
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
        throw "Non rectangular vector input!";
      // copy row
      for (dim_t j=0; j<_width; ++j)
        set(i, j, src[i][j]);
    }
  }

  /**
   * Copy Constructor - Copy a matrix
   * @param src Matrix to copy from
   */
  Matrix(const Matrix &src): Matrix(src.width(), src.height())
  {
    for (dim_t i=0; i<_height; ++i)
      for (dim_t j=0; j<_width; ++j)
        set(i, j, src.get(i, j));
  }

  ~Matrix()
  {
    for (dim_t i=0; i<_height; ++i)
    {
      delete [] _data[i];
    }
    delete [] _data;
  }

  /* methods */
  /**
   * Prints matrix to stdout
   */
  void print()
  {
    for (int i = 0; i < _height; ++i)
    {
      for (int j = 0; j < _width; ++j)
        std::cout << " " << get(i, j);
      std::cout << std::endl;
    }
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

  /**
   * Return a pointer to a new matrix that is the transposition of this matrix
   * @return Matrix* The transposed matrix
   */
  Matrix* transpose() const
  {
    Matrix *ret = new Matrix(_width, _height);
    for (dim_t h=0; h<_height; ++h)
      for (dim_t w=0; w<_width; ++w)
        ret->set(w, h, this->get(h, w));
    return ret;
  }

  /**
   * Return a pointer to a new matrix that is this matrix multiplied by another
   * @param matrix Matrix to be multiplied by
   * @return Matrix* The product matrix
   */
  Matrix *dot(const Matrix &o)
  {
    if (_width != o.h())
      throw "Invalid matrix dimensions!";
    Matrix *ret = new Matrix(_height, o.w());
    for (dim_t r = 0; r < _height; ++r)
    {
      for (dim_t c = 0; c < o.w(); ++c)
      {
        val_t sum = 0;
        for (dim_t i = 0; i < _width; ++i)
        {
          sum += get(r, i) * o.get(i, c);
        }
        ret->set(r, c, sum);
      }
    }
    return ret;
  }
};