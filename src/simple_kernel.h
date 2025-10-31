#pragma once
#include <cuda_runtime.h>

// Runs a simple GPU compute simulation (multiplies data by 2)

void run_matmul_layer(void* dA, void* dB, void* dC, int m, int n, int k);