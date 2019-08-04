#include <cstdlib> 
#include <iostream>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

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
    char* value = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int localRank;
    if(value == NULL) {
        localRank = 0;
    } else {
        localRank = std::atoi(value);
    }
    
    // Spin the background thread that does CUDA operations
    std::thread thread(&HerringBridge::bg_thread, this);
    thread.detach();
    
    // AllReduce Segments
    value = std::getenv("AR_SEGMENTS");
    if(value == NULL) {
        value = "arsegments.csv";
    }
    std::ifstream arfile(value);

    std::string line;
    int cur_seg_index = 0, cur_offset = 0;
    while(std::getline(arfile, line, ',')) {
        // Get var_id
        int var_id = std::stoi(line);
        // Get length
        std::getline(arfile, line);
        int len = std::stoi(line);
        
        if(var_id == -1) {
            // End of segment
            cur_seg_index++;
        } else {
            // record offset
            offsets[var_id] = cur_offset;
            cur_offset += len * sizeof(float);
            var_length[var_id] = len;
            
            // record segment index of this gradient
            segment_index[var_id] = cur_seg_index;
            
            // How many variables are there in this segment
            segment_var_count[cur_seg_index]++;
        }
    }
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

    //std::cout << "var_id: " << task->var_id_cpu << " len: " << len << " Time: " << milliseconds_since_epoch << std::endl;
    // First iteration - temporarily store in CPU memory

    // Beginning of second iteration - figure out where this data goes

    // After we know where this data goes
}


void HerringBridge::copy_allreduced_data(const uint32_t* var_id, const void* buffer, void* dest) {
    auto task = std::make_shared<BeginAllReduceTask>(BeginAllReduceTask::TYPE_COPY_RESULT, var_id, buffer, dest);
    bg_thread_queue.push(task);
    sem_bg_thread.notify();
    task->done.wait();
}

void HerringBridge::bg_thread() {
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaStreamCreate(&cudaStream));
    while(true) {
        sem_bg_thread.wait();

        auto task = bg_thread_queue.front(); bg_thread_queue.pop();
        switch(task->task_type) {
        case BeginAllReduceTask::TYPE_START_AR:
            cudaMemcpyAsync(&(task->var_id_cpu), task->var_id_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream);
            CUDACHECK(cudaStreamSynchronize(cudaStream));

            cudaMemcpyAsync((char*)task->data_dest + offsets[task->var_id_cpu], task->data_in, task->data_size, cudaMemcpyDeviceToDevice, cudaStream);
            break;
        case BeginAllReduceTask::TYPE_COPY_RESULT:
            cudaMemcpyAsync(&(task->var_id_cpu), task->var_id_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream);
            cudaMemcpyAsync(task->data_dest, (char*)task->buffer + offsets[task->var_id_cpu], 
                    var_length[task->var_id_cpu] * sizeof(float), cudaMemcpyDeviceToDevice, cudaStream);
            CUDACHECK(cudaStreamSynchronize(cudaStream));
            break;
        }
        task->done.notify();
    }
}

std::shared_ptr<BeginAllReduceTask> HerringBridge::start_allreduce(const uint32_t* var_id_gpu, const void* data_in, void* data_dst, int data_size) {
    auto task = std::make_shared<BeginAllReduceTask>(BeginAllReduceTask::TYPE_START_AR, var_id_gpu, data_in, data_dst, data_size);
    bg_thread_queue.push(task);
    sem_bg_thread.notify();
    task->done.wait();
    return task;
}
