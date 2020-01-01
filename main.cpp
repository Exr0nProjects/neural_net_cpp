/**
 * A toy neural net in C++
 * @author Exr0n
 * @version 0.0.1
 */

#include <iostream>
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
  dim_t _width = 0;
  dim_t _height = 0;
public:
  /* pre */
  val_t get (const dim_t x, const dim_t y) const;
  void set(const dim_t x, const dim_t y, const val_t &dat);

  /* constructors/destructors */
  Matrix() {}

  /**
   * Empty aInitializer Constructor - Create a new emtpy Matrix of w by h
   * @param width Width of the new Matrix
   * @param height Height of the new Matrix
   */
  Matrix(const dim_t width, const dim_t height): _width(width), _height(height)
  {
    _data = new val_t*[height];
    for (dim_t i=0; i<height; ++i)
      _data[i] = new val_t[width];
  }

  /**
   * Initalizer Constructor - Create a Matrix from a 2D vector
   * @param src 2D vector to copy from
   */
  Matrix(const vector<vector<val_t> > &src): Matrix(src.size(), src[0].size())
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

  ~Matrix()
  {
    for (dim_t i=0; i<_height; ++i)
      delete [] _data[i];
    delete [] _data;
  }

  /* methods */
  /**
   * Get the value in the matrix at (x, y)
   * @param x The column of the element to be retrieved
   * @param y The row of the element to be retrieved
   * @return val_t The value of the element at that position
   */
  val_t get(const dim_t x, const dim_t y) const
  {
    return _data[y][x];
  }
  /**
   * Set the value in the matrix at (x, y)
   * @param x The column of the element to be set
   * @param y The row of the element to be set
   * @param dat The data to be copied into that position
   */
  void set(const dim_t x, const dim_t y, const val_t &dat)
  {
    _data[y][x] = dat;
  }
}