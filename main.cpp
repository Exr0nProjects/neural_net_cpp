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

int main()
{
  std::cout << "Please enter the dimensions for the matrix to transpose:" << std::endl;
  int w, h;
  scanf("%d%d", &w, &h);

  Matrix<int> mat(w, h);

  for (int i=0; i<h; ++i)
  {
    for (int j=0; j<w; ++j)
    {
      int d;
      scanf("%d", &d);
      mat.set(j, i, d);
    }
  }

  printf("Okay, transposing your matrix...\n");

  Matrix<int>* ret = mat.transpose();

  printf("Done! (%d, %d) Here it is:\n", ret->w(), ret->h());

  ret->print();

/*
3 2
1 2 3
4 5 6
*/

  return 0;
}