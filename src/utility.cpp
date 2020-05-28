#include <cstdio>
#include <chrono>
#include <stdexcept>

void progressBar(int width, double progress, int overwrite = true, const std::string &fill = "=", const std::string &empty = " ", int segment_size = 4)
{
    if (progress > 1 || progress < 0)
        throw std::domain_error("Invalid progress!");

    for (int i = 0; i < overwrite; ++i)
        printf("\033[F");

    int pos = width * progress;
    if (segment_size > pos)
        segment_size = pos + 1; // make gaps at the beginning
    static int dot_offset = 0;
    dot_offset = (dot_offset + 1) % segment_size;

    printf("[");
    if (progress * 100 + 1 < 100)
    {
        for (int i = 0; i < width; ++i)
        {
            if (i < pos)
            {
                if (i % segment_size == dot_offset)
                    printf("%s", empty.c_str());
                else
                    printf("%s", fill.c_str());
            }
            else if (i == pos)
                printf(">");
            else if (i > pos)
                printf(" ");
        }
    }
    else
    {
        for (int i = 0; i < width; ++i)
            printf("%s", fill.c_str());
    }
    printf("] %3d%%\n", (int)(progress * 100 + 1));
}

