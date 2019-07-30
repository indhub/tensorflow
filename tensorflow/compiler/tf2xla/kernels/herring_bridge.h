#ifndef HERRING_BRIDGE_H
#define HERRING_BRIDGE_H

#include <stdint.h>
#include <cstddef>
#include <memory>
#include <queue>
#include <map>
#include <mutex>

#include "semaphore.h"

class BeginAllReduceTask {
public:
    const uint32_t* var_id_gpu;
    uint32_t var_id_cpu;

    const void* data_in;
    void* data_dest;
    int data_size;

    Semaphore done;

    BeginAllReduceTask(const uint32_t* var_id_gpu, const void* data_in, void* data_dest, int data_size)
        : var_id_gpu(var_id_gpu),
          data_in(data_in),
          data_dest(data_dest),
          data_size(data_size)
    {
    }
};

class HerringBridge {
public:
    static HerringBridge& getInstance();
    void queue_allreduce(const uint32_t* var_id, int len, const void* data, void* buffer);
    void copy_allreduced_data(const uint32_t* var_id, void* dest);
private:
    HerringBridge();
    ~HerringBridge();
    HerringBridge(HerringBridge const&);
    void operator=(HerringBridge const&);

    void bg_thread();

    Semaphore sem_bg_thread;
    std::queue<std::shared_ptr<BeginAllReduceTask> > bg_thread_queue;

    std::shared_ptr<BeginAllReduceTask> start_allreduce(const uint32_t* var_id_gpu, const void* data_in, void* data_dst, int data_size);

    // AR segments
    std::map<int, int> offsets;
    std::map<int, int> segment_index;
    std::map<int, int> segment_var_count;
    std::map<int, int> segment_recv_count;
    std::mutex mtx_ar_segments;
};


#endif
