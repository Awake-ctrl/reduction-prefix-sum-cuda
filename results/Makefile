CC = gcc
NVCC = nvcc
CFLAGS = -O3 -std=c99
NVCCFLAGS = -O3 -std=c++11
SRCDIR = src
TESTDIR = test_cases
RESULTDIR = results
BINDIR = bin

SOURCES_CPU = $(SRCDIR)/cpu_operations.c
SOURCES_GPU = $(SRCDIR)/gpu_operations.cu
SOURCES_MAIN = $(SRCDIR)/main.cu

TARGET = $(BINDIR)/main
TEST_GENERATOR = $(BINDIR)/generate_tests

all: dirs $(TARGET) $(TEST_GENERATOR)

dirs:
	mkdir -p $(BINDIR) $(RESULTDIR) $(TESTDIR)

$(TARGET): $(SOURCES_MAIN) $(SOURCES_CPU) $(SOURCES_GPU)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(TEST_GENERATOR): $(TESTDIR)/generate_test_cases.c
	$(CC) $(CFLAGS) -o $@ $^

run: all
	./$(TARGET)

tests: $(TEST_GENERATOR)
	./$(TEST_GENERATOR)

clean:
	rm -rf $(BINDIR) $(RESULTDIR)/*.csv

.PHONY: all dirs run tests clean