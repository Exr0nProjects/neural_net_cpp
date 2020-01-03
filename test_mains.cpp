// matrix transpose
int main()
{
  Matrix<int> *mat = matrixIn<int>();

  printf("Okay, transposing your matrix...\n");

  Matrix<int> *ret = mat->transpose();

  printf("Done! (%dx%d) Here it is:\n", ret->h(), ret->w());

  ret->print();

  /*
2 3
1 2 3
4 5 6
*/

  return 0;
}