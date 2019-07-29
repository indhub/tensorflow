#include <cstdlib> 
#include <iostream>
#include <thread>
#include <chrono>
#include <iostream>

#include "herring_bridge.h"

#include "cuda_runtime.h"
#include "semaphore.h"

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
    
    // Spin the background thread that does CUDA operations
    std::thread thread(&HerringBridge::bg_thread, this);
    thread.detach();
}

HerringBridge::~HerringBridge() {
}

HerringBridge& HerringBridge::getInstance() {
    static HerringBridge instance;
    return instance;
}

void HerringBridge::queue_allreduce(const uint32_t* var_id_gpu, int len, const void* data, void* buffer) {

    unsigned long milliseconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count();

    auto task = start_allreduce(var_id_gpu, data, buffer, len);

    std::cout << "var_id: " << task->var_id_cpu << " len: " << len << " Time: " << milliseconds_since_epoch << std::endl;
    // First iteration - temporarily store in CPU memory

    // Beginning of second iteration - figure out where this data goes

    // After we know where this data goes
}

void* HerringBridge::get_result(uint32_t var_id) {
    return NULL;
}

void HerringBridge::bg_thread() {
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaStreamCreate(&cudaStream));
    while(true) {
        sem_bg_thread.wait();

        auto task = bg_thread_queue.front(); bg_thread_queue.pop();
        cudaMemcpyAsync(&(task->var_id_cpu), task->var_id_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream);
        cudaMemcpyAsync(task->data_dest, task->data_in, task->data_size, cudaMemcpyDeviceToDevice, cudaStream);
        //CUDACHECK(cudaStreamSynchronize(cudaStream));

        task->done.notify();
    }
}

std::shared_ptr<BeginAllReduceTask> HerringBridge::start_allreduce(const uint32_t* var_id_gpu, const void* data_in, void* data_dst, int data_size) {
    auto task = std::make_shared<BeginAllReduceTask>(var_id_gpu, data_in, data_dst, data_size);
    bg_thread_queue.push(task);
    sem_bg_thread.notify();
    task->done.wait();
    return task;
}
