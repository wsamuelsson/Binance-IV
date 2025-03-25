# Makefile for compiling compute_IV.c with gcc

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O3 -Wall -Wall -ftree-vectorize

#Link flags
LFLAGS = -lm

# Output executable name
TARGET = compute_IV

# Source file
SRC = compute_IV.c

# Default target
all: $(TARGET)

# Compile the source file
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LFLAGS)

# Clean up
clean:
	rm -f $(TARGET)

