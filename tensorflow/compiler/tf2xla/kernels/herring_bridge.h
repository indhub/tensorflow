#ifndef HERRING_BRIDGE_H
#define HERRING_BRIDGE_H

#include <stdint.h>
#include <cstddef>
#include <memory>
#include <queue>

#include "semaphore.h"

class BeginAllReduceTask {
public:
    const uint32_t* var_id_gpu;
    uint32_t var_id_cpu;

    const void* data_in;
    void* data_dest;
    int data_size;

    Semaphore done;

    BeginAllReduceTask(const uint32_t* var_id_gpu, const void* data_in, void* data_dest, int data_size) {
    }
};

class HerringBridge {
public:
    static HerringBridge& getInstance();
    void queue_allreduce(const uint32_t* var_id, int len, const void* data, void* buffer);
    void* get_result(uint32_t var_id);
private:
    HerringBridge();
    ~HerringBridge();
    HerringBridge(HerringBridge const&);
    void operator=(HerringBridge const&);

    void bg_thread();

    Semaphore sem_bg_thread;
    std::queue<std::shared_ptr<BeginAllReduceTask> > bg_thread_queue;

    std::shared_ptr<BeginAllReduceTask> start_allreduce(const uint32_t* var_id_gpu, const void* data_in, void* data_dst, int data_size);
};


#endif
