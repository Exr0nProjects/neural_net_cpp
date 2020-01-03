/**
 * A toy neural net in C++
 * @author Exr0n
 * @version 0.0.1
 */

#include <iostream>
#include <vector>

#pragma once
#include "src/matrix.h"
#include "src/layer.h"
#include "src/network.h"

template <class T>
Matrix<T>* matrixIn()
{
  std::cout << "Please enter a matrix with dimensions at the top:" << std::endl;
  T h, w;
  std::cin >> h >> w;
  Matrix<T>* ret = new Matrix<T>(h, w);

  for (int i=0; i<h; ++i)
    for (int j=0; j<w; ++j)
    {
      T d;
      std::cin >> d;
      ret->set(i, j, d);
    }
  return ret;
}

int main()
{
  Matrix<int> *a = matrixIn<int>();
  Matrix<int> *b = matrixIn<int>();

  printf("Multiplying matricies...\n");

  Matrix<int> *ret = a->dot(*b);

  printf("Done! (%dx%d) Here it is:\n", ret->h(), ret->w());
  ret->print();

  /*
2 3
1 2 3
4 5 6
3 3
1 0 0
0 1 0
0 0 1
*/

  return 0;
}