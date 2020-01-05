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
  // std::ios_base::sync_with_stdio(false);
  // std::cin.tie(NULL);

  std::cout << "Please enter a matrix with dimensions at the top:" << std::endl;
  typedef double val_t;
  Matrix<val_t> *inp = matrixIn<val_t>();

  inp->print();

  printf("Creating layer\n");

  Layer<val_t> *layer = new Layer<val_t>(inp->w(), 1);

  printf("Created a %dx%d layer\n", layer->in_size(), layer->out_size());
  
  layer->syn_raw()->print();

  auto nxt = *inp - *inp;
  printf("Addr   outside fxn: %d\n", &nxt);
  nxt.print();

  printf("\nTraining...\n");

  const int CYCLES = 60000;
  const int UPDATES = 10;

  // Matrix<val_t> * l1 = layer->feed(inp);

  for (int i=1; i<CYCLES; ++i)
  {
    auto l1 = layer->feed(inp);


    if (i % (CYCLES/UPDATES) == 0)
    {
      printf("%d%% progress: \n", i*100/CYCLES);
    }
  }

  //printf("Done! (%dx%d) Here it is:\n", ret->h(), ret->w());
  //l1->print(15);

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