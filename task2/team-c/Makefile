CC = gcc

CFLAGS = -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lblis -lm

TARGET = ./build/libkernels.so
SRC = ./src/rmsnorm.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -shared -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)