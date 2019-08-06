#ifndef HERRING_BRIDGE_INTERNAL_
#define HERRING_BRIDGE_INTERNAL_

#include "cuda_runtime.h"

class CudaEvent {
public:
    CudaEvent() {
        cudaEventCreate(&_event);
    }

    ~CudaEvent() {
        cudaEventDestroy(_event);
    }

    operator cudaEvent_t() {
        return _event;
    }

    CudaEvent(const CudaEvent&) = delete; // don't allow copy
    CudaEvent& operator=(const CudaEvent&) = delete; // don't allow copy
private:
    cudaEvent_t _event;
};

#endif

