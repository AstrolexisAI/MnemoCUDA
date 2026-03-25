NVCC ?= nvcc
CC ?= gcc
CUDA_HOME ?= /usr/local/cuda
CFLAGS = -O2 -Wall -Isrc -I$(CUDA_HOME)/include
NVCCFLAGS = -O2 --use_fast_math
LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lpthread -lm

# Auto-detect GPU architecture (default sm_75 for broad compat)
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/' || echo "sm_75")

BUILD_DIR = build

all: $(BUILD_DIR)/mnemo_server $(BUILD_DIR)/libmnemo_cuda.so

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# CUDA kernels
$(BUILD_DIR)/kernels.o: src/kernels.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -arch=$(GPU_ARCH) -Xcompiler -fPIC -c $< -o $@

# Engine
$(BUILD_DIR)/engine.o: src/engine.c src/engine.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

# Shared library
$(BUILD_DIR)/libmnemo_cuda.so: $(BUILD_DIR)/kernels.o $(BUILD_DIR)/engine.o
	$(NVCC) -shared -o $@ $^ $(LDFLAGS)

# Server binary
$(BUILD_DIR)/mnemo_server: src/mnemo_server.c $(BUILD_DIR)/kernels.o $(BUILD_DIR)/engine.o
	$(CC) $(CFLAGS) -o $@ $< $(BUILD_DIR)/kernels.o $(BUILD_DIR)/engine.o $(LDFLAGS)

# Tests
$(BUILD_DIR)/test_engine: tests/test_engine.c $(BUILD_DIR)/kernels.o $(BUILD_DIR)/engine.o
	$(CC) $(CFLAGS) -o $@ $< $(BUILD_DIR)/kernels.o $(BUILD_DIR)/engine.o $(LDFLAGS)

$(BUILD_DIR)/test_heat: tests/test_heat.c
	$(CC) -O2 -Wall -o $@ $< -lm

test: $(BUILD_DIR)/test_heat
	$(BUILD_DIR)/test_heat

install: $(BUILD_DIR)/mnemo_server
	install -m 755 $(BUILD_DIR)/mnemo_server /usr/local/bin/mnemo_server

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean test install
