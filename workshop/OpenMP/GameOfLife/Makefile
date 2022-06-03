all: game_of_life

game_of_life: main.c
	gcc -std=c99 -O2 -fopenmp main.c -o game_of_life

clean:
	rm -f game_of_life