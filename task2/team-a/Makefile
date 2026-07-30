CC = gcc
CFLAGS = -O3 -Wall -Wextra -fPIC -I./include -I/usr/include/blis
LDFLAGS = -shared -lm -lblis

SRC_DIR = source/kernels
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

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
	python3 tests/linear.py

clean:
	rm -rf $(BUILD_DIR)
