#pragma once

#include <vector>

/**
 * Matrix: Mathamatical Workhorse
 * Designed to do all of the heavy lifting math wise for the network
 * Will be modified to work with CUDA later
 */
template<class val_t> class Matrix
{
  typedef unsigned int dim_t;
  val_t **_data;
  dim_t _width;
  dim_t _height;
public:

  /* constructors/destructors */
  Matrix() {}

  /**
   * Empty Initializer Constructor - Create a new emtpy Matrix of w by h
   * @param width Width of the new Matrix
   * @param height Height of the new Matrix
   */
  Matrix(const dim_t width, const dim_t height);

  /**
   * Initalizer Constructor - Create a Matrix from a 2D vector
   * @param src 2D vector to copy from
   */
  Matrix(const std::vector<std::vector<val_t> > &src);

  /**
   * Copy Constructor - Copy a matrix
   * @param src Matrix to copy from
   */
  Matrix(const Matrix &src);

  ~Matrix();

  /* methods */
  /**
   * Returns the width of this matrix
   * @return dim_t The width of the matrix
   */
  dim_t width() const;

  /**
   * Returns the height of this matrix
   * @return dim_t The height of the matrix
   */
  dim_t height() const;

  /**
   * Get the value in the matrix at (x, y)
   * @param x The column of the element to be retrieved
   * @param y The row of the element to be retrieved
   * @return val_t The value of the element at that position
   */
  val_t get(const dim_t x, const dim_t y) const;
  /**
   * Set the value in the matrix at (x, y)
   * @param x The column of the element to be set
   * @param y The row of the element to be set
   * @param dat The data to be copied into that position
   */
  void set(const dim_t x, const dim_t y, const val_t &dat);

  /**
   * Return a copy of this matrix, but transposed
   * @return matrix A Matrix that is this matrix but transposed
   */
  Matrix transpose() const;
};