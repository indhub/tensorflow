#include <cstdlib>
#include <iostream>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi/include/mpi.h"

#include "herring_bridge.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

HerringBridge::HerringBridge() {
}

HerringBridge::~HerringBridge() {
}

HerringBridge& HerringBridge::getInstance() {
    static HerringBridge instance;
    return instance;
}

void HerringBridge::queue_allreduce(const uint32_t* var_id_gpu, int len, const void* data, void* buffer, void* output) {
}

void HerringBridge::copy_allreduced_data(const uint32_t* var_id,
        const void* data_in, void* buffer, void* dest, std::function<void()> asyncDoneCallback) {
}

