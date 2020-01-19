#include <cstdio>
#include <chrono>
#include <stdexcept>

void progressBar(int width, double progress, bool overwrite=true, int max_dots=4)
{
  if (progress > 1 || progress < 0)
    throw std::domain_error("Invalid progress!");

  if (overwrite) printf("\033[F");
  
  int pos = width * progress;
  static int dot_offset = 0;
  dot_offset = (dot_offset + 1) % max_dots;

  printf("[");
  if (progress*100+1 < 100)
  {
    for (int i=0; i<width; ++i)
    {
      if (i < pos)
      {
        if (i%max_dots == dot_offset)
          printf(" ");
        else
          printf("=");
      }
      else if (i == pos) printf(">");
      else if (i > pos) printf(" ");
    }
  }
  else
  {
    for (int i=0; i<width; ++i) printf("=");
  }
  printf("] %2d%%\n", (int)(progress*100+1));
}