CC=g++
CFLAGS=-std=gnu++17 -I

debug: main.cpp
	$(CC) main.cpp -o main $(CFLAGS) && clear && ./main test.in