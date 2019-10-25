#ifndef HERRING_BRIDGE_H
#define HERRING_BRIDGE_H

#include <stdint.h>
#include <cstddef>
#include <memory>
#include <queue>
#include <map>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "semaphore.h"
#include "herring_bridge_internal.h"

class PartialAllReduceTask {
public:
    int task_type;
    const uint32_t* var_id_gpu;
    uint32_t var_id_cpu;

    const void* data_in;
    void* buffer;
    void* data_out;
    int data_size;

    Semaphore done;
    std::function<void()> asyncDoneCallback;

    PartialAllReduceTask(int task_type, const uint32_t* var_id_gpu, int data_size, 
                       const void* data_in, void* buffer, void* data_out, std::function<void()> asyncDoneCallback)
        : task_type(task_type),
          var_id_gpu(var_id_gpu),
          data_size(data_size),
          data_in(data_in),
          buffer(buffer),
          data_out(data_out),
          asyncDoneCallback(asyncDoneCallback)
    {
    }

    PartialAllReduceTask(int task_type, const uint32_t* var_id_gpu, int data_size,
                       const void* data_in, void* buffer, void* data_out)
        : task_type(task_type),
          var_id_gpu(var_id_gpu),
          data_size(data_size),
          data_in(data_in),
          buffer(buffer),
          data_out(data_out)
    {
    }

    static const int TYPE_START_AR = 0;
    static const int TYPE_COPY_RESULT = 1;
};

class HerringBridge {
public:
    static HerringBridge& getInstance();
    void queue_allreduce(const uint32_t* var_id, int len, 
                         const void* data_in, void* buffer, void* data_out);
    void copy_allreduced_data(const uint32_t* var_id, 
                         const void* data_in, void* buffer, void* data_out, std::function<void()> done);
private:
    HerringBridge();
    ~HerringBridge();
    HerringBridge(HerringBridge const&);
    void operator=(HerringBridge const&);

    AllReduceHelper& helper;

    void bg_thread();

    std::mutex mtx_bg_thread;
    Semaphore sem_bg_thread;
    std::queue<std::shared_ptr<PartialAllReduceTask> > bg_thread_queue;

    std::shared_ptr<PartialAllReduceTask> start_allreduce(const uint32_t* var_id_gpu, int data_size, 
            const void* data_in, void* data_buffer, void* data_output);

    // AR segments
    std::map<int, int> offsets; // Offset of a given variable in buffer
    std::map<int, int> var_length; // Length of a given variable
    std::map<int, int> segment_index; // Segment index of a given variable
    std::map<int, int> segment_var_count; // Number of variables in a given segment
    std::map<int, int> segment_recv_count; // Number of variables received in this iteration for a given segment
    std::map<int, int> segment_offset; // Offset of a given segment in buffer
    std::map<int, int> segment_length; // Length of a given segment in buffer
    std::mutex mtx_ar_segments;
    
    // Synchronization for finish_allreduce
    std::mutex mtx_finish_allreduce;
    // Queue for when finisher comes first. Finisher will queue the var id, dest address
    // and done callback. allreduce handler can copy data and call the callbacks.
    std::unordered_map<int, std::queue<std::shared_ptr<PartialAllReduceTask>>> gradsAwaitingAllreduce;
    // How many grads are immediately available (allready allreduced) for this segment
    std::unordered_map<int, int> num_grads_available_for_segment;
    Semaphore sem_allreduce_event_available;
    std::queue<std::shared_ptr<CudaEvent>> queueAllReduceEvents;
    std::queue<int> queueAllReduceSegmentIds;
    void allreduce_event_handler();


    int myRank, nRanks, localRank;
};

HerringBridge& getHerringBridge() {
    return HerringBridge::getInstance();
}

#endif
