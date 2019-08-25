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
    void copy_allreduced_data(const uint32_t* var_id,
                         const void* data_in, void* buffer, void* data_out, std::function<void()> done);

private:
    // Singleton related stuff
    HerringBridge();
    ~HerringBridge();
    HerringBridge(HerringBridge const&);
    void operator=(HerringBridge const&);

    
};

HerringBridge& getHerringBridge() {
    return HerringBridge::getInstance();
}

#endif
