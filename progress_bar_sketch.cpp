#include "src/utility.cpp"

using namespace std;

int main()
{
  long long epoch = 800000000;
  printf("\n\nGET READY FOR PROGRESS BAR MADNESS\n\n");
  for (long long i=0; i<epoch; ++i)
  {
    if ((i % (epoch/250)) == 0)
    {
      progressBar(50, (double)i/epoch, true, 12);
      // printf("%lf\n", (double)i/epoch);
    }
  }
  printf("\e[3mepic.\n\n");
}