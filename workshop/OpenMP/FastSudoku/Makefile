all: sudoku

sudoku: main.c
	gcc -std=c99 -O2 -fopenmp main.c -o sudoku

bit: main-bit.c
	gcc -std=c99 -O2 -fopenmp main-bit.c -o sudoku-bit

clean:
	rm -f sudoku