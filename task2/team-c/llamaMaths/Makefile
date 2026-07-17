CC = gcc
CFLAGS = -Wall -Wextra -fPIC -Iinc
LDFLAGS = -shared -lm

SRC = src/kernels/silu.c  src/kernels/mul.c
TARGET = build/libkernels.so

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p build
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf build

.PHONY: all clean