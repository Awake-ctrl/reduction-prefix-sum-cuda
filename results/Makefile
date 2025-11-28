# =================== COMPILER CONFIGURATION ===================
CC = g++
NVCC = nvcc
CFLAGS = -O3 -std=c++11
NVCCFLAGS = -O3 -std=c++11

# =================== DIRECTORY STRUCTURE ===================
SRCDIR = src
TESTDIR = test_cases
RESULTDIR = results
BINDIR = bin

# =================== SOURCE FILES ===================
SOURCES_CPU = $(SRCDIR)/cpu_operations.cpp
SOURCES_GPU = $(SRCDIR)/gpu_operations.cu
SOURCES_MAIN = $(SRCDIR)/main.cu

# =================== TARGETS ===================
TARGET = $(BINDIR)/main
TEST_GENERATOR = $(BINDIR)/generate_tests

# =================== MAIN TARGET ===================
all: dirs $(TARGET) $(TEST_GENERATOR)

dirs:
      mkdir -p $(BINDIR) $(RESULTDIR) $(TESTDIR)

# =================== COMPILATION RULES ===================
$(TARGET): $(SOURCES_MAIN) $(SOURCES_CPU) $(SOURCES_GPU)
        $(NVCC) $(NVCCFLAGS) -o $@ $^ -lm

$(TEST_GENERATOR): $(TESTDIR)/generate_test_cases.cpp
        $(CC) $(CFLAGS) -o $@ $^

# =================== UTILITY TARGETS ===================
run: all
        ./$(TARGET)

tests: $(TEST_GENERATOR)
        ./$(TEST_GENERATOR)

clean:
        rm -rf $(BINDIR) $(RESULTDIR)/*.csv

.PHONY: all dirs run tests clean