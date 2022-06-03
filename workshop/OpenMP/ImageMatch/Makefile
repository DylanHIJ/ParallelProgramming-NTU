all: image_match

image_match: main.c
	gcc -std=c99 -O2 -fopenmp main.c -o image_match

clean:
	rm -f image_match