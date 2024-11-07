# Makefile for compiling compute_IV.c with gcc

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -Wall

# Output executable name
TARGET = compute_IV

# Source file
SRC = compute_IV.c

# Default target
all: $(TARGET)

# Compile the source file
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean up
clean:
	rm -f $(TARGET)
