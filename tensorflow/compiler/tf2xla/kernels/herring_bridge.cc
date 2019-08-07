#include <cstdlib> 
#include <iostream>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "efa/include/mpi.h"

#include "herring_bridge_internal.h"
#include "herring_bridge.h"
#include "semaphore.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static cudaStream_t cudaStream;
static ncclComm_t ncclComm;

HerringBridge::HerringBridge() {
    // Get local rank
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    char* value = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(value == NULL) {
        localRank = 0;
    } else {
        localRank = std::atoi(value);
    }

    // Spin the background thread that does CUDA operations
    std::thread cuda_thread(&HerringBridge::bg_thread, this);
    cuda_thread.detach();
    
    // Spin the allreduce event handler
    std::thread ar_events_thread(&HerringBridge::allreduce_event_handler, this);
    ar_events_thread.detach();

    // AllReduce Segments
    value = std::getenv("AR_SEGMENTS");
    if(value == NULL) {
        value = "arsegments.csv";
    }
    std::ifstream arfile(value);

    std::string line;
    int cur_seg_index = 0, cur_offset = 0;
    int cur_segment_length = 0;

    segment_offset[cur_seg_index] = cur_offset;
    
    while(std::getline(arfile, line, ',')) {
        // Get var_id
        int var_id = std::stoi(line);
        // Get length
        std::getline(arfile, line);
        int len = std::stoi(line);
        
        if(var_id == -1) {
            // End of segment; set segment length
            segment_length[cur_seg_index] = cur_segment_length;
            cur_segment_length = 0;
            // Start tracking next segment
            cur_seg_index++;
            segment_offset[cur_seg_index] = cur_offset;
        } else {
            // record offset
            offsets[var_id] = cur_offset;
            cur_offset += len * sizeof(float);
            var_length[var_id] = len;
            
            // record segment index of this gradient
            segment_index[var_id] = cur_seg_index;
            
            // How many variables are there in this segment
            segment_var_count[cur_seg_index]++;

            // Increment segment length
            cur_segment_length += len;
        }
    }
}

HerringBridge::~HerringBridge() {
}

HerringBridge& HerringBridge::getInstance() {
    static HerringBridge instance;
    return instance;
}

void HerringBridge::queue_allreduce(const uint32_t* var_id_gpu, int len, const void* data, void* buffer, void* output) {

    unsigned long milliseconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count();

    auto task = start_allreduce(var_id_gpu, len, data, buffer, output);
}


void HerringBridge::copy_allreduced_data(const uint32_t* var_id,
        const void* data_in, void* buffer, void* dest, std::function<void()> asyncDoneCallback) {
    // Do CUDA operations from a seperate thread. Just wrap everything and send it to 
    auto task = std::make_shared<PartialAllReduceTask>(PartialAllReduceTask::TYPE_COPY_RESULT, var_id, 0, 
                                                       data_in, buffer, dest, asyncDoneCallback);
    bg_thread_queue.push(task);
    sem_bg_thread.notify();
}

void HerringBridge::allreduce_event_handler() {
    CUDACHECK(cudaSetDevice(localRank));
    while(true) {
        sem_allreduce_event_available.wait(); // Wait for the next ncclAllReduce call to be issued
        auto event = queueAllReduceEvents.front(); queueAllReduceEvents.pop(); // Grab the cuda event
        int segmentId = queueAllReduceSegmentIds.front(); queueAllReduceSegmentIds.pop(); // Grab the segment ID
        cudaEventSynchronize(*event); // Wait for the AllReduce call to complete

        {
            // pick the lock so that we don't race with finisher
            std::lock_guard<std::mutex> guard(mtx_finish_allreduce);
            // How many gradient tensors did we just AllReduce?
            int num_grads_available = segment_var_count[segmentId];
            // Grab the waiting tasks from finisher corresponding to this segment ID
            auto queue_tasks = gradsAwaitingAllreduce[segmentId];
            // Iterate through the tasks
            while(!queue_tasks.empty()) {
                auto task = queue_tasks.front(); queue_tasks.pop();
                // Copy gradient
                cudaMemcpy(task->data_out, (char*)task->buffer + offsets[task->var_id_cpu],
                        var_length[task->var_id_cpu] * sizeof(float), cudaMemcpyDeviceToDevice);
                // Fire the callback to notify we are done
                task->asyncDoneCallback();
                num_grads_available--; // Reduce the available gradients
            }
            // How many more gradients do we have available? 
            // Set it so that when finisher checks in, we can hand over immediately
            num_grads_available_for_segment[segmentId] = num_grads_available;
        }
    }
}

void print_some_floats(const void* data, int count) {
    std::vector<float> vect;
    vect.resize(count);
    float* ptr = &vect[0];

    cudaMemcpy(ptr, data, count * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "[";
    for(int i=0; i<count; i++) {
        std::cout << ptr[i] << " ";
    }
    std::cout << "]" << std::endl;
}

void HerringBridge::bg_thread() {

    ncclUniqueId ncclId;
    if (myRank == 0) ncclGetUniqueId(&ncclId);
    MPICHECK(MPI_Bcast((void *)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));

    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaStreamCreate(&cudaStream));

    NCCLCHECK(ncclCommInitRank(&ncclComm, nRanks, ncclId, myRank));

    int segment_id;
    while(true) {
        sem_bg_thread.wait();

        auto task = bg_thread_queue.front(); bg_thread_queue.pop();
        switch(task->task_type) {
        case PartialAllReduceTask::TYPE_START_AR:
            // Copy the gradients to a temporary buffer and return immediately
            cudaMemcpy(&(task->var_id_cpu), task->var_id_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy((char*)task->buffer + offsets[task->var_id_cpu], task->data_in, 
                    task->data_size, cudaMemcpyDeviceToDevice);

            // If we have received all data for this segment, start a ncclAllReduce asynchronously
            segment_id = segment_index[task->var_id_cpu]; // Which segment ID?
            segment_recv_count[segment_id]++; // We received one more for this segment
            if(segment_recv_count[segment_id] == segment_var_count[segment_id]) { // If we received all of them...

                // Get a pointer to the corresponding data in the buffer
                void* allreduce_ptr = (char*)task->buffer + segment_offset[segment_id];

                // Get the length of the subsequence to AllReduce
                int allreduce_length = segment_length[segment_id];

                // Call NCCL AllReduce!
                ncclAllReduce(allreduce_ptr, allreduce_ptr, segment_length[segment_id],
                              ncclFloat32, ncclSum, ncclComm, cudaStream);

                // Record an event. allreduce_event_handler will track this event and complete it.
                auto cudaEvent = std::make_shared<CudaEvent>();
                {
                    // ToDo: This lock isn't really required. Test and remove.
                    std::lock_guard<std::mutex> guard(mtx_finish_allreduce);
                    queueAllReduceEvents.push(cudaEvent);
                    queueAllReduceSegmentIds.push(segment_id);
                }
                sem_allreduce_event_available.notify();

                // Reset number of grads received for this segment
                segment_recv_count[segment_id] = 0;
            }
            task->done.notify();
            break;
        case PartialAllReduceTask::TYPE_COPY_RESULT:
            // Get the var id
            cudaMemcpy(&(task->var_id_cpu), task->var_id_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            // Get the segment id
            int seg_indx = segment_index[task->var_id_cpu];
            // Assume for now that the grad is not already available
            bool gradAvailableNow = false;
            {
                // Grab the lock. Don't race with allreduce event handler
                std::lock_guard<std::mutex> guard(mtx_finish_allreduce);
                // Is the grad already available?
                if(num_grads_available_for_segment[seg_indx] > 0) {
                    num_grads_available_for_segment[seg_indx]--; // One less available now
                    gradAvailableNow = true; // Remember we have the grad
                } else {
                    // Grad is not already available. Queue this one. 
                    // allreduce event handler will complete this when gradient is available
                    gradsAwaitingAllreduce[seg_indx].push(task);
                }
            }
            if(gradAvailableNow) {
                // If grad is available, copy it and call the done callback
                cudaMemcpy(task->data_out, (char*)task->buffer + offsets[task->var_id_cpu],
                        var_length[task->var_id_cpu] * sizeof(float), cudaMemcpyDeviceToDevice);
                task->asyncDoneCallback();
            }
            break;
        }
    }
}

std::shared_ptr<PartialAllReduceTask> HerringBridge::start_allreduce(const uint32_t* var_id_gpu, int data_len, 
        const void* data_in, void* data_buffer, void* data_out) {
    auto task = std::make_shared<PartialAllReduceTask>(PartialAllReduceTask::TYPE_START_AR, 
            var_id_gpu, data_len * sizeof(float), data_in, data_buffer, data_out);
    bg_thread_queue.push(task);
    sem_bg_thread.notify();
    task->done.wait();
    return task;
}
