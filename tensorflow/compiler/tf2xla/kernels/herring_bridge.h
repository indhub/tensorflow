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

    // Singleton. We don't want two instance of this class causing confusion.
    static HerringBridge& getInstance();
    
    // When gradient is received by the XLA op, it is sent here.
    // This method is synchronous and needs to return as quick as possible.
    void queue_allreduce(const uint32_t* var_id, int len,
                         const void* data_in, void* buffer, void* data_out);

    // The op that collects allreduced gradients canns this function. This function is asynchronous.
    // The done callback will be called when gradients are ready.
    void copy_allreduced_data(const uint32_t* var_id, const void* orig_grad, const void* data_in,
                              void* buffer, void* data_out, std::function<void()> done);

private:
    // Singleton related stuff
    HerringBridge();
    ~HerringBridge();
    HerringBridge(HerringBridge const&);
    void operator=(HerringBridge const&);

    void bg_thread();
    std::mutex mtx_bg_thread;
    Semaphore sem_bg_thread;
    std::queue<std::shared_ptr<PartialAllReduceTask> > bg_thread_queue;

    std::shared_ptr<PartialAllReduceTask> start_allreduce(const uint32_t* var_id_gpu, int data_size,
            const void* data_in, void* data_buffer, void* data_output);
    
};

HerringBridge& getHerringBridge() {
    return HerringBridge::getInstance();
}

#endif
