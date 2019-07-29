#include "herring_bridge.h"

HerringBridge& HerringBridge::getInstance() {
    static HerringBridge instance;
    return instance;
}

void HerringBridge::queue_allreduce(const uint32_t* var_id, int len, const void* data) {
}

void* HerringBridge::get_result(uint32_t var_id) {
    return NULL;
}

