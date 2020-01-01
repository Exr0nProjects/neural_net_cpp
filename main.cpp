/**
 * A toy neural net in C++
 * @author Exr0n
 * @version 0.0.1
 */

#include <iostream>
#include <vector>
#include <cmath>

#define ld long double

/**
 * Matrix: Mathamatical Workhorse
 * Designed to do all of the heavy lifting math wise for the network
 * Will be modified to work with CUDA later
 */
class Matrix
{
  typedef unsigned int dim_t;
  ld **_data = nullptr;
  dim_t width = 0;
  dim_t height = 0;
public:
  Matrix() {}
  
  /**
   * Initalizer Constructor - Create a Matrix from a 2D vector
   * @param src 2D vector to copy from
   */
  Matrix(const vector<vector<ld> > &src)
  {
    // init variables
    height = src.size();
    width = src[0].size();
    _data = new int*[height];

    // copy data
    for (int i=0; i<height; ++i)
    {
      // check shape
      if (src[i].size() != width)
        throw "Non rectangular vector input!";
      // copy row
      _data[i] = new int[width];
      for (int j=0; j<width; ++j)
        _data[i][j] = src[i][j];
    }
  }
}