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

static cudaStream_t cudaStream;

HerringBridge::HerringBridge() {
  // Spin the background thread that does CUDA operations
  std::thread cuda_thread(&HerringBridge::bg_thread, this);
  cuda_thread.detach();

}

void HerringBridge::bg_thread() {
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking));

  while(true) {
    sem_bg_thread.wait();
    auto task = bg_thread_queue.front(); bg_thread_queue.pop();

    switch(task->task_type) {
      case PartialAllReduceTask::TYPE_START_AR:
        cudaMemcpyAsync(task->data_out, task->data_in, task->data_size, cudaMemcpyDeviceToDevice, cudaStream);
        break;
    }
  }

}

HerringBridge::~HerringBridge() {
}

HerringBridge& HerringBridge::getInstance() {
    static HerringBridge instance;
    return instance;
}

void HerringBridge::queue_allreduce(const uint32_t* var_id_gpu, int len, const void* data, void* buffer, void* output) {
  auto task = start_allreduce(var_id_gpu, len, data, buffer, output);
}

void HerringBridge::copy_allreduced_data(const uint32_t* var_id, const void* orig_grad, const void* data_in,
                                         void* buffer, void* dest, std::function<void()> asyncDoneCallback) {
  // Just copy from orig_grad to dest

}

std::shared_ptr<PartialAllReduceTask> HerringBridge::start_allreduce(const uint32_t* var_id_gpu, int data_len,
        const void* data_in, void* data_buffer, void* data_out) {
    auto task = std::make_shared<PartialAllReduceTask>(PartialAllReduceTask::TYPE_START_AR,
            var_id_gpu, data_len * sizeof(float), data_in, data_buffer, data_out);
    {
        std::lock_guard<std::mutex> guard(mtx_bg_thread);
        bg_thread_queue.push(task);
    }
    sem_bg_thread.notify();
    //task->done.wait();
    return task;
}

