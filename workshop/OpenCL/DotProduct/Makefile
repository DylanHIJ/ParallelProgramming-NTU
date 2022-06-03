CC = gcc
CFLAGS = -std=c99
CL_TARGET = dot-product-cl
LIBS = -lOpenCL

all: $(CL_TARGET)
%-cl: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
clean:
	rm -f $(CL_TARGET)
