#include <cstdlib> 
#include <iostream>

#include "herring_bridge.h"

#include "cuda_runtime.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static cudaStream_t cudaStream;

HerringBridge::HerringBridge() {
    // Get local rank
    const char* value = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int localRank;
    if(value == NULL) {
        std::cerr << "OMPI_COMM_WORLD_LOCAL_RANK not available" << std::endl;
        localRank = 0;
    } else {
        localRank = std::atoi(value);
    }

    // CUDA init
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaStreamCreate(&cudaStream));
}

HerringBridge::~HerringBridge() {
}

HerringBridge& HerringBridge::getInstance() {
    static HerringBridge instance;
    return instance;
}

void HerringBridge::queue_allreduce(const uint32_t* var_id_gpu, int len, const void* data, void* buffer) {
    uint32_t var_id_cpu;
    cudaMemcpyAsync(&var_id_cpu, var_id_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream);
}

void* HerringBridge::get_result(uint32_t var_id) {
    return NULL;
}

