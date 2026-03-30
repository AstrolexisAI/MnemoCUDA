NVCC ?= nvcc
CC ?= gcc
CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
CFLAGS = -O2 -Wall -Wextra -Isrc -I$(CUDA_HOME)/include -fstack-protector-strong -D_FORTIFY_SOURCE=2
NVCCFLAGS = -O2 --use_fast_math
LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lpthread -lm -Wl,-z,relro -Wl,-z,now
NVCC_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lpthread -lm -Xlinker -z,relro -Xlinker -z,now

# Use g++-14 as nvcc host compiler (GCC 15+ unsupported by CUDA 12.x)
NVCC_HOST_CXX := $(shell which g++-14 2>/dev/null || which $(CXX) 2>/dev/null)
ifneq ($(NVCC_HOST_CXX),)
  NVCCFLAGS += -ccbin=$(NVCC_HOST_CXX)
endif

# Auto-detect GPU architecture (default sm_75 for broad compat)
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/' || echo "sm_75")

BUILD_DIR = build

# Source modules
ENGINE_OBJS = $(BUILD_DIR)/engine.o $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/io_pool.o $(BUILD_DIR)/heat.o $(BUILD_DIR)/forward.o $(BUILD_DIR)/json_helpers.o

all: $(BUILD_DIR)/mnemo_server $(BUILD_DIR)/libmnemo_cuda.so

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# CUDA kernels
$(BUILD_DIR)/kernels.o: src/kernels.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -arch=$(GPU_ARCH) -Xcompiler -fPIC -c $< -o $@

# Engine modules
$(BUILD_DIR)/engine.o: src/engine.c src/engine.h src/engine_internal.h src/log.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(BUILD_DIR)/tokenizer.o: src/tokenizer.c src/tokenizer.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(BUILD_DIR)/io_pool.o: src/io_pool.c src/io_pool.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(BUILD_DIR)/heat.o: src/heat.c src/heat.h src/engine_internal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(BUILD_DIR)/forward.o: src/forward.c src/forward.h src/engine_internal.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(BUILD_DIR)/json_helpers.o: src/json_helpers.c src/json_helpers.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

# Shared library
$(BUILD_DIR)/libmnemo_cuda.so: $(BUILD_DIR)/kernels.o $(ENGINE_OBJS)
	$(NVCC) -shared -o $@ $^ $(NVCC_LDFLAGS)

# Server binary
$(BUILD_DIR)/mnemo_server: src/mnemo_server.c src/log.h $(BUILD_DIR)/kernels.o $(ENGINE_OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(BUILD_DIR)/kernels.o $(ENGINE_OBJS) $(LDFLAGS)

# Tests
$(BUILD_DIR)/test_engine: tests/test_engine.c $(BUILD_DIR)/kernels.o $(ENGINE_OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(BUILD_DIR)/kernels.o $(ENGINE_OBJS) $(LDFLAGS)

$(BUILD_DIR)/test_heat: tests/test_heat.c
	$(CC) -O2 -Wall -o $@ $< -lm

$(BUILD_DIR)/test_server: tests/test_server.c $(BUILD_DIR)/json_helpers.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(BUILD_DIR)/json_helpers.o -lm

test: $(BUILD_DIR)/test_heat $(BUILD_DIR)/test_engine $(BUILD_DIR)/test_server
	$(BUILD_DIR)/test_heat
	$(BUILD_DIR)/test_engine
	$(BUILD_DIR)/test_server

install: $(BUILD_DIR)/mnemo_server
	install -m 755 $(BUILD_DIR)/mnemo_server /usr/local/bin/mnemo_server

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean test install
