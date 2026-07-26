# Makefile for building linear_nobias kernel and running its test

CC = gcc
CFLAGS = -O2 -fPIC
INCLUDES = -I/usr/local/include/blis
LDFLAGS = -L/usr/local/lib -lblis -lpthread -lm

SRC = \
src/linear_nobias.c \
src/rope.c \
src/silu.c \
src/mul.c \
src/swiglu.c \
src/rmsnorm.c


BUILD_DIR = build
TARGET = $(BUILD_DIR)/libkernels.so

.PHONY: all test clean

# Default target: build the shared library
all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

# Build (if needed) and run the Python test
test: $(TARGET)
	pip install -r requirements.txt
	python3 test/test_linear_nobias.py
	python3 test/test_rope.py
	python3 test/test_swiglu.py
	python3 test/rmsnorm.py

# Remove build artifacts
clean:
	rm -rf $(BUILD_DIR)
