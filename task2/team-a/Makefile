CC = gcc
CFLAGS = -O3 -Wall -Wextra -fPIC -I./inc
LDFLAGS = -shared -lm

SRC_DIR = src/kernels
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

# Using .so for Linux shared object
TARGET = $(BUILD_DIR)/libkernels.so

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

test: all
	python3 tests/layernorm.py
	python3 tests/add.py

clean:
	rm -rf $(BUILD_DIR)