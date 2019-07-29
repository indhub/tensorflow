#ifndef HERRING_BRIDGE_H
#define HERRING_BRIDGE_H

#include <stdint.h>
#include <cstddef>

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
};

#endif
