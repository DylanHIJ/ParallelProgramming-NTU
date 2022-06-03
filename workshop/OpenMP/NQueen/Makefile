all: n_queen

n_queen: main.c
	gcc -std=c99 -O2 -fopenmp main.c -o n_queen

n_queen_bit: main-bit.c
	gcc -std=c99 -O2 -fopenmp main-bit.c -o n_queen_bit

clean:
	rm -f n_queen n_queen_bit