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
Matrix<T> matrixIn()
{
  T h, w;
  std::cin >> h >> w;
  Matrix<T> ret(h, w);

  for (int i=0; i<h; ++i)
    for (int j=0; j<w; ++j)
    {
      T d;
      std::cin >> d;
      ret.set(i, j, d);
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
  typedef float val_t;
  Matrix<val_t> inp = matrixIn<val_t>();
  Matrix<val_t> expected = matrixIn<val_t>();

  inp.print();

  Layer<val_t> *layer1 = new Layer<val_t>(inp.w(), 3);
  Layer<val_t> *layer2 = new Layer<val_t>(layer1->out_size(), 1);

  printf("Created a %dx%d layer\n", layer1->in_size(), layer1->out_size());

  // auto nxt = *inp * *inp;
  // printf("Addr   outside fxn: %d\n", &nxt);
  // nxt.print();

  printf("\nTraining...\n");

  const int CYCLES = 500000;
  const int UPDATES = 50; 

  for (int i=1; i<CYCLES; ++i)
  {
    Matrix<val_t> l1 = layer1->feed(inp);
    Matrix<val_t> l2 = layer2->feed(l1);

    // expected.print();
    // l1.print();

    Matrix<val_t> error = expected - l2;
    // Matrix<val_t> l2_delta = l2_error * (layer2->actv_raw()->deriv(l2));

    // Matrix<val_t> l1_error = Matrix<val_t>::dot(l2_delta, Matrix<val_t>::transpose(layer2->syn_raw()));
    // Matrix<val_t> l1_delta = l1_error * (layer1->actv_raw()->deriv(l1));

    // layer2->update_raw(Matrix<val_t>::dot(Matrix<val_t>::transpose(l1), l2_delta));
    Matrix<val_t> l2_delta = layer2->backprop(l1, l2, expected - l2);
    // layer1->update_raw(Matrix<val_t>::dot(Matrix<val_t>::transpose(inp), l1_delta));
    Matrix<val_t> l1_delta = layer1->backprop(inp, l1, Matrix<val_t>::dot(l2_delta, Matrix<val_t>::transpose(layer2->syn_raw())));

    if (i % (CYCLES/UPDATES) == 0)
    {
      printf("Input:\n");
      inp.print();
      printf("Output:\n");
      l2.print();
      printf("Exepected:\n");
      expected.print();
      // l1_error.print();
      val_t average_error = 0;
      for (int i=0; i<error.h(); ++i)
        average_error += abs(error.get(i, 0));
      average_error /= error.h();
      printf("\n%d%% progress - error = %.5f\n\n----------\n", i*100/CYCLES, average_error);
      //layer->syn_raw()->print();
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