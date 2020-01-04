/**
 * A toy neural net in C++
 * @author Exr0n
 * @version 0.0.1
 */

#include <iostream>
#include <vector>

#pragma once
#include "src/matrix.cpp"
#include "src/activation.cpp"
#include "src/layer.cpp"
#include "src/network.cpp"

template <class T>
Matrix<T>* matrixIn()
{
  //std::cout << "Please enter a matrix with dimensions at the top:" << std::endl;
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

int main(const int argc, char ** argv)
{
  if (argc >= 1)
    freopen(argv[1], "r", stdin);
  if (argc >= 2)
    freopen(argv[2], "w+", stdout);
  // freopen("test.in", "r", stdin);
  // freopen("test.out", "w+", stdout);
  // std::ios_base::sync_with_stdio(false);
  // std::cin.tie(NULL);
  Matrix<double> *a = matrixIn<double>();

  a->print();

  printf("Activating\n");

  Activation<double> *activator = new Activation<double>();
  Matrix<double> *ret = (*activator)(a);

  printf("\nanswer\n");

  //printf("Done! (%dx%d) Here it is:\n", ret->h(), ret->w());
  ret->print(15);

  /*
2 3
1 2 3
4 5 6
3 3
1 0 0
0 1 0
0 0 1

2 3
1 2 3
4 5 6
3 3
1 0 0
0 2 0
0 0 3
*/

  return 0;
}