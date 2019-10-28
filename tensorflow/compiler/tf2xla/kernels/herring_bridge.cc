#include <cstdlib> 
#include <iostream>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "nccl.h"
#include "efa/include/mpi.h"

#include "herring_bridge_internal.h"
#include "herring_bridge.h"
#include "semaphore.h"
#include "mpi_helper.h"
#include "mpi_initializer.hpp"


#include <memory>
#include <mutex>
#include <queue>
#include <sstream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/stream.h"

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

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

//
// MemoryManager
//
class MemoryManager {
public:
  static MemoryManager& getInstance(int block_size) {
    static MemoryManager instance(block_size);
    return instance;
  }
  void* get_block(int block_index, bool is_input_buffer) {
    void* ret;
    std::lock_guard<std::mutex> guard(mtx_available_blocks);
    if(available_blocks.size() > 0) {
      auto itr = available_blocks.begin();
      ret = *itr;
      available_blocks.erase(itr);
      return ret;
    } else {
      //ret = shared_mem_alloc(msg_factory, block_index, block_size, is_input_buffer);
      //CUDACHECK(cudaHostRegister(ret, block_size, cudaHostRegisterDefault));
      void* ret;
      CUDACHECK(cudaMallocHost(&ret, block_size));
      return ret;
    }
  }

  void* return_block(void* addr) {
    std::lock_guard<std::mutex> guard(mtx_available_blocks);
    available_blocks.insert(addr);
  }

private:
  MemoryManager(int block_size)
    :block_size(block_size)
  {
  }
public:
  MemoryManager(MemoryManager const&) = delete;
  void operator=(MemoryManager const&) = delete;


private:
  std::unordered_set<void*> available_blocks;
  std::mutex mtx_available_blocks;
  size_t block_size;
};

using namespace tensorflow;

class DataReadyEvent {
public:
  DataReadyEvent(DeviceContext* device_context) {
    auto executor = device_context->stream()->parent();
    auto ready_event = new perftools::gputools::Event(executor);
    ready_event->Init();
    stream_executor::Stream& stream = device_context->stream()->ThenRecordEvent(ready_event);
    //stream_executor::cuda::AsCUDAStream(&stream).GpuStreamMemberHack();
    //stream.implementation()->GpuStreamMemberHack();
    this->event = std::shared_ptr<perftools::gputools::Event>(ready_event);
    this->device_context = device_context;
  }

  void wait() {
    device_context->stream()->ThenWaitFor(this->event.get());
  }

private:
  std::shared_ptr<perftools::gputools::Event> event;
  DeviceContext* device_context;
};


//
// Task
//

enum TaskType{BEGIN_XLA_ALL_REDUCE, D2H_COPY, H2D_COPY, D2H_COPY_COMPLETE, H2D_COPY_COMPLETE,
  REDUCE_SCATTER, REDUCE_SCATTER_COMPLETE, ALL_GATHER, ALL_GATHER_COMPLETE,
  ACC_SEND, ACC_RECV, NCCL_ALL_REDUCE, DUMMY_COPY};

class Task {
public:
  Task(TaskType type, int segment_index, const void* src, void* ram_dest, void* eventual_dest,
       int length, std::function<void()> fn, MPIHelper mpi_helper)
    : type(type), segment_index(segment_index), src(src), ram_dest(ram_dest), eventual_dest(eventual_dest),
      length(length), done_callback(fn)
  {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaEventCreate(&cudaEvent));
  }

public:
  const void* src;
  void *ram_dest, *eventual_dest;
  int segment_index, length;
  int reduce_scatter_offset;
  int buffer_offset;
  std::function<void()> done_callback;
  TaskType type;
  cudaEvent_t cudaEvent;
  bool isLastBlock = false;
  std::vector<std::unique_ptr<Task> > subtasks;
  std::shared_ptr<DataReadyEvent> data_ready_event;
  const uint32_t* var_id_gpu;
};

//
// AllReduceHelper
//

class AllReduceHelper {
private:
  std::string prefix() {
    std::stringstream ss;
    ss << "[client-" << mpi_helper.get_world_rank() << "] ";
    return ss.str();
  }
public:
  static AllReduceHelper& getInstance() {
    static AllReduceHelper instance;
    return instance;
  }

  std::chrono::system_clock::duration time_begin, time_end;

private:
  bool first_iteration = true;
  std::vector<int> param_order;
  std::mutex mtx_param_order;
  std::unordered_map<int, std::unique_ptr<Task> > individual_tasks;
  std::mutex mtx_individual_tasks;
  std::vector<Semaphore* > sem_individual_tasks;
  std::unordered_set<int> seen_in_first_itr;

  int buffer_length = 0;
  unsigned short *buffer_;

public:

  //
  // XLA BEGIN_ALLREDUCE
  //
  void begin_xla_all_reduce(const uint32_t* var_id_gpu, const void* data, void* output, int length, 
          std::function<void()> fn) {
    auto task = make_unique<Task>(BEGIN_XLA_ALL_REDUCE, -1, data, (void*)NULL, output,
        length, fn, mpi_helper);
    task->var_id_gpu = var_id_gpu;

    {
      std::lock_guard<std::mutex> guard(mtx_begin_xla_all_reduce_queue);
      begin_xla_all_reduce_queue.push(std::move(task));
    }
  }

  void begin_xla_all_reduce_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&begin_xla_all_reduce_stream, cudaStreamNonBlocking));

    uint32_t var_id_cpu;

    while(true) {
      sem_xla_all_reduce_thread.wait();

      std::unique_ptr<Task> task;
      {
        std::lock_guard<std::mutex> guard(mtx_begin_xla_all_reduce_queue);
        task = std::move(begin_xla_all_reduce_queue.front()); begin_xla_all_reduce_queue.pop();
      }

      // Get the var ID. TODO: Remove the expensive cudaMemcpy
      cudaMemcpy(&var_id_cpu, task->var_id_gpu, sizeof(uint32_t), cudaMemcpyDeviceToHost);

      begin_all_reduce(var_id_cpu, task->src, task->eventual_dest, task->length, NULL, []{});
    }
  }

  //
  // NON XLA BEGIN_ALLREDUCE
  //
  void begin_all_reduce(int param_index, const void* input_ptr, void* output_ptr,
      int length, std::shared_ptr<DataReadyEvent> ready_event, std::function<void()> fn) {

    auto task = make_unique<Task>(DUMMY_COPY, 0, input_ptr, (void*)NULL, output_ptr,
        length, fn, mpi_helper);
    task->data_ready_event = ready_event;
    { 
      std::lock_guard<std::mutex> guard(mtx_dummy_copy_queue);
      dummy_copy_queue.push(std::move(task));
    }
    sem_dummy_copy_thread.notify();
  }

  std::mutex mtx_begin_all_reduce;
  void begin_all_reduce_orig(int param_index, const void* input_ptr, void* output_ptr,
      int length, std::shared_ptr<DataReadyEvent> ready_event, std::function<void()> fn) {

    std::lock_guard<std::mutex> out_guard(mtx_begin_all_reduce);
    //std::this_thread::sleep_for(std::chrono::microseconds(750));
    //std::cout << "starting allreduce for param " << param_index << std::endl;
    // Insert task into individual_tasks
    auto task = make_unique<Task>(NCCL_ALL_REDUCE, param_index, input_ptr, (void*)NULL, output_ptr,
        length, fn, mpi_helper);
    task->data_ready_event = ready_event;
    CUDACHECK(cudaEventRecord(task->cudaEvent, 0));
    {
    std::lock_guard<std::mutex> guard(mtx_individual_tasks);
    individual_tasks.erase(param_index);
    individual_tasks.insert(std::make_pair(param_index, std::move(task)));
    }

    //std::this_thread::sleep_for(std::chrono::microseconds(750));

    // If we have seen this param already, we are no more in first iteration
    if(first_iteration && seen_in_first_itr.count(param_index) > 0) {
      first_iteration = false;

      // Create the buffer
      buffer_length += (buffer_length%8==0)? 0 : 8 - (buffer_length%8);
      CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
      std::cout << "allocating buffer of length " << buffer_length << std::endl;
      CUDACHECK(cudaMalloc(&buffer_, buffer_length * sizeof(unsigned short)));

      // Start the after_first_iteration_thread
      std::thread after_first_itr_thread(&AllReduceHelper::after_first_iteration_thread_fn, this);
      after_first_itr_thread.detach();
    }

    // If we are still in first iteration..
    if(first_iteration) {
      seen_in_first_itr.insert(param_index);
      // We need little more buffer space
      buffer_length += length;
      // Expand sem_individual_tasks if necessary
      {
        std::lock_guard<std::mutex> guard(mtx_individual_tasks);
        while(sem_individual_tasks.size() <= param_index) {
          Semaphore* sem_ptr = new Semaphore();
          sem_individual_tasks.push_back(sem_ptr);
        }
      }

      // If I'm rank 0, I need to signal the first iteration thread
      if(mpi_helper.get_world_rank()==0) {
        {
        // add param_index to param_order
        std::lock_guard<std::mutex> guard(mtx_param_order);
        param_order.push_back(param_index);
        }
        // signal the first iteration thread
        sem_first_iteration_thread.notify();
      }
    } else {
      sem_individual_tasks[param_index]->notify();
    }
  }

  //
  // Dummy Copy Thread
  //
  void dummy_copy_thread_fn() {

    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&dummy_copy_stream, cudaStreamNonBlocking));

    while(true) {
      sem_dummy_copy_thread.wait();

      std::unique_ptr<Task> task;
      {
        std::lock_guard<std::mutex> guard(mtx_dummy_copy_queue);
        task = std::move(dummy_copy_queue.front()); dummy_copy_queue.pop();
      }

      //cudaMemcpy(task->eventual_dest, task->src, task->length * sizeof(unsigned short),
      //           cudaMemcpyDeviceToDevice);
      //CUDACHECK(cudaStreamSynchronize(0));
      task->data_ready_event->wait();
      //std::this_thread::sleep_for(std::chrono::microseconds(10));
      //CUDACHECK(cudaDeviceSynchronize());
      CUDACHECK(cudaMemcpyAsync(task->eventual_dest, task->src, task->length * sizeof(unsigned short),
          cudaMemcpyDeviceToDevice, dummy_copy_stream));
      CUDACHECK(cudaStreamSynchronize(dummy_copy_stream));
      //NCCLCHECK(ncclAllReduce(task->src, task->eventual_dest, task->length, ncclFloat16, ncclSum,
      //  nccl_comm_workers, 0));
      //CUDACHECK(cudaStreamSynchronize(0));

      task->done_callback();
    }

  }

  // 
  // FIRST ITERATION THREAD
  //
  void first_iteration_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&begin_stream, cudaStreamNonBlocking));

    int rank = mpi_helper.get_world_rank();
    int param_index;
    int i = 0;
    while(true) {
      //std::this_thread::sleep_for(std::chrono::microseconds(750));
      // If rank 0, wait for the next param to be added 
      // to param_order list. Every other rank will wait
      // for rank 0 to broadcast this index.
      if(rank == 0) {
        sem_first_iteration_thread.wait();
        param_index = param_order[i++];
      }
      // After this bcast, everybody will have the same param_index
      // to allreduce next.
      MPICHECK(MPI_Bcast((void*)&param_index, sizeof(param_index), MPI_BYTE, 0,
          mpi_initializer.getSubCluster()));

      // If I'm not rank 0, make note of the order of parameters
      if(rank != 0) {
        param_order.push_back(param_index);
      }

      // Grab the task corresponding to the param_index
      while(individual_tasks.count(param_index)==0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      std::unique_ptr<Task> task;
      {
        std::lock_guard<std::mutex> guard(mtx_individual_tasks);
        task = std::move(individual_tasks[param_index]);
      }

      // Call ncclAllReduce
      //CUDACHECK(cudaMemcpyAsync(task->eventual_dest, task->src, task->length*2, cudaMemcpyDeviceToDevice, begin_stream));
      //cudaStreamSynchronize(0);
      CUDACHECK(cudaEventSynchronize(task->cudaEvent));
      task->data_ready_event->wait(); // This doesn't work
      NCCLCHECK(ncclAllReduce(task->src, task->eventual_dest, task->length, ncclFloat16, ncclSum,
        nccl_comm_workers, begin_stream));
      CUDACHECK(cudaStreamSynchronize(begin_stream));
      //cudaStreamSynchronize(0);

      // Mark it done
      task->done_callback();
    }
  }

  std::unordered_set<int> param_printed;

  // 
  // AFTER FIRST ITERATION THREAD
  //
  void after_first_iteration_thread_fn() {
    int stepIndex = -1;
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    while(true) {
      int offset = 0;
      int send_begin = 0;
      int block_index = 0;
      std::queue<std::unique_ptr<Task> > subtasks;

      int first_param = param_order[0];
      int last_param = param_order[param_order.size()-1];
      for(int param_index: param_order) {
        bool is_last = (param_index == last_param)? true : false;

        if(param_index == first_param) {

          //MPI_Barrier(worker_comm);

          stepIndex++;
          if(stepIndex==50) {
            cudaProfilerStart();
          } else if(stepIndex==60) {
            cudaProfilerStop();
          }
        }

        // Wait for the next parameter to be ready
        sem_individual_tasks[param_index]->wait();

        //std::this_thread::sleep_for(std::chrono::microseconds(750));

        // Grab the task corresponding to the next parameter
        std::unique_ptr<Task> task;
        {
        std::lock_guard<std::mutex> guard(mtx_individual_tasks);
        task = std::move(individual_tasks[param_index]);
        }

        // Copy to buffer
        void* dest = (void*)(buffer_+offset);
        const void* src = task->src;
        int len = task->length;
        //CUDACHECK(cudaStreamSynchronize(0));
        //CUDACHECK(cudaEventSynchronize(task->cudaEvent));
        //CUDACHECK(cudaDeviceSynchronize());
        task->data_ready_event->wait();
        CUDACHECK(cudaMemcpyAsync(dest, src, len * sizeof(unsigned short),
            cudaMemcpyDeviceToDevice, begin_stream));
        //CUDACHECK(cudaStreamSynchronize(0));
        task->buffer_offset = offset;

        if(param_printed.count(param_index) == 0) {
          std::cout << "param " << param_index << " of length " << len << " at offset " << offset << std::endl;
          param_printed.insert(param_index);
        }

        offset += task->length;

        // Add to subtasks if it must be added
        // bool is_last = false;
        // bool all_subtasks_added = false;
        //if(offset - send_begin <= COPY_BLOCK_SIZE/sizeof(unsigned short) || task->isLastBlock) {
        //  is_last = task->isLastBlock;
        //  subtasks.push(std::move(task));
        //  all_subtasks_added = true;
        //}

        bool this_subtask_sent = false;

        int copy_block_len = COPY_BLOCK_SIZE/sizeof(unsigned short);

        // Queue reduce-scatter if we have enough data
        while((offset - send_begin >= copy_block_len) || is_last) {
          //std::cout << "about to reduce scatter" << std::endl;

          // CPU addr
          void* dest = mem_manager.get_block(block_index, true);

          int copy_block_len = COPY_BLOCK_SIZE/sizeof(unsigned short);
          int len = (offset-send_begin > copy_block_len)? copy_block_len : offset-send_begin;
          bool break_loop = (offset-send_begin <= copy_block_len)? true : false;

          len += (len%8==0)? 0 : (8-(len%8)); // must be divisible by 8

          // Create new task to push to next stage
          auto new_task = make_unique<Task>(REDUCE_SCATTER, block_index, buffer_+send_begin, dest,
              buffer_+send_begin, len, nullptr, mpi_helper);
          CUDACHECK(cudaEventRecord(task->cudaEvent, begin_stream));

          if(break_loop) {
            if((offset - send_begin == COPY_BLOCK_SIZE) || is_last) {
              subtasks.push(std::move(task));
              this_subtask_sent = true;
            }
          }

          // copy over subtasks
          while(subtasks.size()>0) {
            auto task = std::move(subtasks.front());
            //std::cout << prefix() << "Adding param " << task->segment_index 
            //  << " to segment " << new_task->segment_index << std::endl;
            new_task->subtasks.push_back(std::move(task));
            subtasks.pop();
          }

          // any subtask we didn't copy over? Send it next time.
          //if(!all_subtasks_added) {
          //  subtasks.push(std::move(task));
          //}

          // push task into reduce_scatter_queue
          {
          std::lock_guard<std::mutex> guard(mtx_reduce_scatter_queue);
          reduce_scatter_queue.push(std::move(new_task));
          }
          // wait for copy to be over
          // cudaStreamSynchronize(begin_stream);
          // signal reduce_scatte thread
          sem_reduce_scatter_queue.notify();
          //std::cout << "Signalled reduce scatter thread for block " << block_index << std::endl;
          // Some bookkeeping
          block_index++;
          send_begin += len;

          if(break_loop) break;
        }

        if(!this_subtask_sent) {
          //std::cout << prefix() << "Queuing param " << task->segment_index << std::endl;
          subtasks.push(std::move(task));
        }
      }
    }
  }

  //
  // REDUCE SCATTER THREAD
  //
  void reduce_scatter_thread_fn() {
    //if(mpi_helper.get_local_rank() == 0) NCCLCHECK(ncclGetUniqueId(&nccl_id));
    //MPICHECK(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_initializer.getLocalComm()));

    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&reduce_scatter_stream, cudaStreamNonBlocking));

    //NCCLCHECK(ncclCommInitRank(&nccl_comm_rs, mpi_helper.get_local_size(), nccl_id, mpi_helper.get_local_rank()));

    std::unique_ptr<Task> task;
    while(true) {
      sem_reduce_scatter_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_reduce_scatter_queue);
        task = std::move(reduce_scatter_queue.front()); reduce_scatter_queue.pop();
      }

      assert(task->type == REDUCE_SCATTER);
      CUDACHECK(cudaEventSynchronize(task->cudaEvent));
      //cudaStreamWaitEvent(reduce_scatter_stream, task->cudaEvent, 0);

      int len = task->length/mpi_helper.get_local_size();
      int local_rank = mpi_helper.get_local_rank();
      //MPI_Barrier(worker_comm);
      //CUDACHECK(cudaDeviceSynchronize());
      //CUDACHECK(cudaStreamSynchronize(0));
      //CUDACHECK(cudaStreamSynchronize(reduce_scatter_stream));
      NCCLCHECK(ncclReduceScatter(task->src, (unsigned short*)task->eventual_dest + local_rank*len,
            len, ncclFloat16, ncclSum, nccl_comm_rs, reduce_scatter_stream));
      //NCCLCHECK(ncclAllReduce(task->src, task->eventual_dest, task->length, ncclFloat16, ncclSum,
      //      nccl_comm_workers, reduce_scatter_stream));
      CUDACHECK(cudaEventRecord(task->cudaEvent, reduce_scatter_stream));

      task->type = REDUCE_SCATTER_COMPLETE;
      {
        std::lock_guard<std::mutex> guard(mtx_reduce_scatter_completion_queue);
        reduce_scatter_completion_queue.push(std::move(task));
      }
      sem_reduce_scatter_completion_queue.notify();
    }
  }

  //
  // REDUCE SCATTER COMPLETION THREAD
  //
  void reduce_scatter_completion_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&copy_to_var_stream, cudaStreamNonBlocking));
    std::unique_ptr<Task> task;
    while(true) {
      sem_reduce_scatter_completion_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_reduce_scatter_completion_queue);
        task = std::move(reduce_scatter_completion_queue.front());
        reduce_scatter_completion_queue.pop();
      }

      assert(task->type == REDUCE_SCATTER_COMPLETE);
      CUDACHECK(cudaEventSynchronize(task->cudaEvent));

      /*
      for(int i=0; i<task->subtasks.size(); i++) {
        unsigned short* src = (unsigned short*)buffer_ + task->subtasks[i]->buffer_offset;
        unsigned short* dest = (unsigned short*) task->subtasks[i]->eventual_dest;
        CUDACHECK(cudaMemcpyAsync(dest, src, task->subtasks[i]->length * sizeof(unsigned short),
           cudaMemcpyDeviceToDevice, copy_to_var_stream));
      }
      */

      //CUDACHECK(cudaEventRecord(task->cudaEvent, copy_to_var_stream));
      //CUDACHECK(cudaEventSynchronize(task->cudaEvent));

      //CUDACHECK(cudaStreamSynchronize(copy_to_var_stream));

      //for(int i=0; i<task->subtasks.size(); i++) {
      //  //std::cout << "done with param " << task->subtasks[i]->segment_index
      //  //  << " in segment " << task->segment_index << std::endl;
      //  task->subtasks[i]->done_callback();
      //}


      task->type = D2H_COPY;
      {
        std::lock_guard<std::mutex> guard(mtx_d2h_copy_queue);
        d2h_copy_queue.push(std::move(task));
      }
      sem_d2h_copy_queue.notify();


      /*
      task->type = ALL_GATHER;
      {
        std::lock_guard<std::mutex> guard(mtx_all_gather_queue);
        all_gather_queue.push(std::move(task));
      }
      sem_all_gather_queue.notify();
      */
    }
  }

private:
  ncclComm_t nccl_comm_rs, nccl_comm_ag, nccl_comm_workers;
  ncclUniqueId nccl_id_rs, nccl_id_ag, nccl_id_workers;

public:
  //
  // D2H COPY THREAD
  //

  void d2h_copy_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking));
    std::unique_ptr<Task> task;
    while(true) {
      sem_d2h_copy_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_d2h_copy_queue);
        task = std::move(d2h_copy_queue.front()); d2h_copy_queue.pop();
      }

      assert(task->type == D2H_COPY);
      int len = task->length/mpi_helper.get_local_size();
      int local_rank = mpi_helper.get_local_rank();
      //MPI_Barrier(worker_comm);
      CUDACHECK(cudaMemcpyAsync(task->ram_dest, (unsigned short*)task->eventual_dest + local_rank*len,
            len * sizeof(unsigned short), cudaMemcpyDeviceToHost, d2h_stream));
      CUDACHECK(cudaEventRecord(task->cudaEvent, d2h_stream));

      task->type = D2H_COPY_COMPLETE;
      {
        std::lock_guard<std::mutex> guard(mtx_d2h_completion_queue);
        d2h_completion_queue.push(std::move(task));
      }
      sem_d2h_completion_queue.notify();
    }
  }

  //
  // D2H COMPLETION THREAD
  //

  void d2h_completion_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    std::unique_ptr<Task> task;
    while(true) {
      sem_d2h_completion_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_d2h_completion_queue);
        task = std::move(d2h_completion_queue.front()); d2h_completion_queue.pop();
      }

      assert(task->type == D2H_COPY_COMPLETE);
      CUDACHECK(cudaEventSynchronize(task->cudaEvent));


      task->type = ACC_SEND;
      {
        std::lock_guard<std::mutex> guard(mtx_acc_send_queue);
        acc_send_queue.push(std::move(task));
      }
      sem_acc_send_queue.notify();


      /*
      task->type = H2D_COPY;
      {
        std::lock_guard<std::mutex> guard(mtx_h2d_copy_queue);
        h2d_copy_queue.push(std::move(task));
      }
      sem_h2d_copy_queue.notify();
      */
    }
  }

  //
  // ACC SEND THREAD
  //
  void acc_send_thread_fn() {
    MPI_Comm worker_comm = mpi_initializer.getSubCluster();
    std::unique_ptr<Task> task;
    while(true) {
      sem_acc_send_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_acc_send_queue);
        task = std::move(acc_send_queue.front()); acc_send_queue.pop();
      }

      //std::this_thread::sleep_for(std::chrono::microseconds(750));

      assert(task->type == ACC_SEND);

      /*
      int len = task->length / mpi_helper.get_local_size();
      int num_acc = mpi_info.acc_ranks.size();
      int per_acc_len = len/num_acc;

      std::vector<MPI_Request> requests(num_acc);
      int req_idx = 0;

      for(int i=0; i<num_acc; i++) {
        unsigned short* send_data_head = (unsigned short*) task->ram_dest;
        unsigned short* send_ptr = send_data_head + i*per_acc_len; 
        int send_len = (i==num_acc-1)? len-(num_acc-1)*per_acc_len : per_acc_len;
        //std::cout << prefix() << "Initiating send for segment " << task->segment_index
        //  << " to rank " << mpi_info.acc_ranks[i] << std::endl;
        //std::cout << prefix() << ", " << task->segment_index << ", " << send_len << ", " << (void*)send_ptr << std::endl;
        MPICHECK( MPI_Isend(send_ptr, send_len, MPI_SHORT, mpi_info.acc_ranks[i], task->segment_index,
            MPI_COMM_WORLD, &requests[req_idx++]) );
        //std::cout << prefix() << "Initiated send for segment " << task->segment_index 
        //  << " to rank " << mpi_info.acc_ranks[i] << std::endl;
      }
      
      //MPI_Waitall(num_acc, &requests[0], MPI_STATUSES_IGNORE);
      //int num_completed = 0;
      //while(num_completed < num_acc) {
      //  int idx;
      //  MPICHECK( MPI_Waitany(requests.size(), &requests[0], &idx, MPI_STATUS_IGNORE) );
      //  //std::cout << prefix() << "Completed send of seg " << task->segment_index
      //  //  << " to " << mpi_info.acc_ranks[idx] << std::endl;
      //  num_completed++;
      //}
      //std::cout << prefix() << "All send complete" << std::endl;

      //MPI_Barrier(worker_comm);
      //std::this_thread::sleep_for(std::chrono::microseconds(10));
      */

      task->type = ACC_RECV;
      {
        std::lock_guard<std::mutex> guard(mtx_acc_recv_queue);
        acc_recv_queue.push(std::move(task));
      }
      sem_acc_recv_queue.notify();

    }
  }

  //
  // ACC RECV THREAD
  //
  void acc_recv_thread_fn() {
    std::unique_ptr<Task> task;
    while(true) {
      sem_acc_recv_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_acc_recv_queue);
        task = std::move(acc_recv_queue.front()); acc_recv_queue.pop();
      }

      assert(task->type == ACC_RECV);

      /*
      int len = task->length / mpi_helper.get_local_size();
      int num_acc = mpi_info.acc_ranks.size();
      int per_acc_len = len/num_acc;

      std::vector<MPI_Request> requests(num_acc);
      int req_idx = 0;

      for(int i=0; i<num_acc; i++) {
        unsigned short* recv_data_head = (unsigned short*) task->ram_dest;
        unsigned short* recv_ptr = recv_data_head + i*per_acc_len;
        int recv_len = (i==num_acc-1)? len-(num_acc-1)*per_acc_len : per_acc_len;
        MPICHECK( MPI_Irecv(recv_ptr, recv_len, MPI_SHORT, mpi_info.acc_ranks[i], task->segment_index,
            MPI_COMM_WORLD, &requests[req_idx++]) );

        //std::cout << prefix() << "Initiated recv for segment " << task->segment_index 
        //  << " from rank " << mpi_info.acc_ranks[i] << std::endl;
      }

      MPI_Waitall(num_acc, &requests[0], MPI_STATUSES_IGNORE);
      //std::cout << prefix() << "All recv complete" << std::endl;
      */

      task->type = H2D_COPY;
      {
        std::lock_guard<std::mutex> guard(mtx_h2d_copy_queue);
        h2d_copy_queue.push(std::move(task));
      }
      sem_h2d_copy_queue.notify();
    }
  }

  //
  // H2D COPY THREAD
  //

  void h2d_copy_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking));
    std::unique_ptr<Task> task;
    while(true) {
      sem_h2d_copy_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_h2d_copy_queue);
        task = std::move(h2d_copy_queue.front()); h2d_copy_queue.pop();
      }

      assert(task->type == H2D_COPY);

      int len = task->length/mpi_helper.get_local_size();
      int local_rank = mpi_helper.get_local_rank();
      CUDACHECK(cudaMemcpyAsync((unsigned short*)task->eventual_dest + local_rank*len,
            task->ram_dest, len * sizeof(unsigned short),
            cudaMemcpyHostToDevice, h2d_stream));
      CUDACHECK(cudaEventRecord(task->cudaEvent, h2d_stream));

      task->type = H2D_COPY_COMPLETE;
      {
        std::lock_guard<std::mutex> guard(mtx_h2d_completion_queue);
        h2d_completion_queue.push(std::move(task));
      }
      sem_h2d_completion_queue.notify();
    }
  }

  //
  // H2D COMPLETION THREAD
  //

  void h2d_completion_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    std::unique_ptr<Task> task;
    while(true) {
      sem_h2d_completion_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_h2d_completion_queue);
        task = std::move(h2d_completion_queue.front()); h2d_completion_queue.pop();
      }

      assert(task->type == H2D_COPY_COMPLETE);
      CUDACHECK(cudaEventSynchronize(task->cudaEvent));
      mem_manager.return_block(task->ram_dest);

      task->type = ALL_GATHER;
      {
        std::lock_guard<std::mutex> guard(mtx_all_gather_queue);
        all_gather_queue.push(std::move(task));
      }
      sem_all_gather_queue.notify();
    }
  }

  //
  // ALL GATHER THREAD
  //
  void all_gather_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&all_gather_stream, cudaStreamNonBlocking));

    std::unique_ptr<Task> task;
    while(true) {
      sem_all_gather_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_all_gather_queue);
        task = std::move(all_gather_queue.front()); all_gather_queue.pop();
      }

      assert(task->type == ALL_GATHER);

      int len = task->length/mpi_helper.get_local_size();
      int local_rank = mpi_helper.get_local_rank();

      //std::cout << prefix() << "ncclAllGather " << (void*)((unsigned short*)task->eventual_dest + local_rank*len)
      //  << ", " << task->eventual_dest << ", " << local_rank << ", " << len  << ", " << (void*)buffer_<< std::endl;
      NCCLCHECK(ncclAllGather((unsigned short*)task->eventual_dest + local_rank*len,
            task->eventual_dest, len, ncclFloat16, nccl_comm_ag, all_gather_stream));

      CUDACHECK(cudaEventRecord(task->cudaEvent, all_gather_stream));

      task->type = ALL_GATHER_COMPLETE;
      {
        std::lock_guard<std::mutex> guard(mtx_all_gather_completion_queue);
        all_gather_completion_queue.push(std::move(task));
      }
      sem_all_gather_completion_queue.notify();
    }
  }

  //
  // ALL GATHER COMPLETION THREAD
  //
  void all_gather_completion_thread_fn() {
    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));
    CUDACHECK(cudaStreamCreateWithFlags(&copy_to_var_stream, cudaStreamNonBlocking));
    std::unique_ptr<Task> task;
    while(true) {
      sem_all_gather_completion_queue.wait();
      {
        std::lock_guard<std::mutex> guard(mtx_all_gather_completion_queue);
        task = std::move(all_gather_completion_queue.front());
        all_gather_completion_queue.pop();
      }
      //std::cout << "all_gather_completion_thread woke up" << std::endl;
      assert(task->type == ALL_GATHER_COMPLETE);
      CUDACHECK(cudaEventSynchronize(task->cudaEvent));
      //std::cout << "cudaEventSynchronize done" << std::endl;

      for(int i=0; i<task->subtasks.size(); i++) {
        unsigned short* src = (unsigned short*)buffer_ + task->subtasks[i]->buffer_offset;
        unsigned short* dest = (unsigned short*) task->subtasks[i]->eventual_dest;
        CUDACHECK(cudaMemcpyAsync(dest, src, task->subtasks[i]->length * sizeof(unsigned short),
           cudaMemcpyDeviceToDevice, copy_to_var_stream));
      }

      //std::cout << "copies queued" << std::endl;

      CUDACHECK(cudaStreamSynchronize(copy_to_var_stream));

      //std::cout << "copies done" << std::endl;

      for(int i=0; i<task->subtasks.size(); i++) {
        //std::cout << "done with param " << task->subtasks[i]->segment_index 
        //  << " in segment " << task->segment_index << std::endl;
        task->subtasks[i]->done_callback();
      }

      //if(task->isLastBlock) {
      //  task->done_callback();
      //  MPI_Barrier(worker_comm);
      //  //time_end = std::chrono::system_clock::now().time_since_epoch();
      //  //auto ar_time = (time_end.count() - time_begin.count())/1000000.0;
      //  //if(mpi_helper.get_world_rank()==0) {
      //  //  std::cout << "allreduce time: " << (int)ar_time << std::endl;
      //  //}
      //}
    }
  }

private:
  const int COPY_BLOCK_SIZE = 16 * 1024 * 1024; //100 MB
  MemoryManager& mem_manager;
  std::queue<std::unique_ptr<Task> > d2h_copy_queue, h2d_copy_queue, d2h_completion_queue, h2d_completion_queue,
    reduce_scatter_queue, reduce_scatter_completion_queue, all_gather_queue, all_gather_completion_queue,
    acc_send_queue, acc_recv_queue, dummy_copy_queue, begin_xla_all_reduce_queue;
  std::mutex mtx_d2h_copy_queue, mtx_h2d_copy_queue, mtx_d2h_completion_queue, mtx_h2d_completion_queue,
    mtx_reduce_scatter_queue, mtx_reduce_scatter_completion_queue,
    mtx_all_gather_queue, mtx_all_gather_completion_queue,
    mtx_acc_send_queue, mtx_acc_recv_queue, mtx_dummy_copy_queue, mtx_begin_xla_all_reduce_queue;
  Semaphore sem_d2h_copy_queue, sem_h2d_copy_queue, sem_d2h_completion_queue, sem_h2d_completion_queue,
            sem_reduce_scatter_queue, sem_reduce_scatter_completion_queue,
            sem_all_gather_queue, sem_all_gather_completion_queue,
            sem_acc_send_queue, sem_acc_recv_queue,
            sem_first_iteration_thread, sem_dummy_copy_thread, sem_xla_all_reduce_thread;
  cudaStream_t d2h_stream, h2d_stream, reduce_scatter_stream, all_gather_stream, begin_stream,
               copy_to_var_stream, dummy_copy_stream, begin_xla_all_reduce_stream;
public:
  AllReduceHelper(AllReduceHelper const&) = delete;
  void operator=(AllReduceHelper const&) = delete;
private:
  MPIHelper mpi_helper;
  MPIInfo mpi_info;
  MPIInitializer& mpi_initializer;
  MPI_Comm worker_comm, eightway_comm;
  AllReduceHelper()
    : mpi_initializer(MPIInitializer::getInstance()),
      mem_manager(MemoryManager::getInstance(COPY_BLOCK_SIZE))
  {
    worker_comm = mpi_initializer.getSubCluster();
    eightway_comm = mpi_initializer.getEightWayComm();

    CUDACHECK(cudaSetDevice(mpi_helper.get_local_rank()));

    if(mpi_helper.get_local_rank() == 0) NCCLCHECK(ncclGetUniqueId(&nccl_id_rs));
    MPICHECK(MPI_Bcast((void *)&nccl_id_rs, sizeof(nccl_id_rs), MPI_BYTE, 0, mpi_initializer.getLocalComm()));
    NCCLCHECK(ncclCommInitRank(&nccl_comm_rs, mpi_helper.get_local_size(), nccl_id_rs, mpi_helper.get_local_rank()));

    if(mpi_helper.get_local_rank() == 0) NCCLCHECK(ncclGetUniqueId(&nccl_id_ag));
    MPICHECK(MPI_Bcast((void *)&nccl_id_ag, sizeof(nccl_id_ag), MPI_BYTE, 0, mpi_initializer.getLocalComm()));
    NCCLCHECK(ncclCommInitRank(&nccl_comm_ag, mpi_helper.get_local_size(), nccl_id_ag, mpi_helper.get_local_rank()));

    if(mpi_helper.get_world_rank() == 0) NCCLCHECK(ncclGetUniqueId(&nccl_id_workers));
    MPICHECK(MPI_Bcast((void *)&nccl_id_workers, sizeof(nccl_id_workers), MPI_BYTE, 0,
          mpi_initializer.getSubCluster()));
    NCCLCHECK(ncclCommInitRank(&nccl_comm_workers, mpi_helper.get_world_size()/2, nccl_id_workers,
          mpi_helper.get_world_rank()));

    int nccl_comm_workers_size, nccl_comm_workers_rank;
    NCCLCHECK(ncclCommCount(nccl_comm_workers, &nccl_comm_workers_size));
    NCCLCHECK(ncclCommUserRank(nccl_comm_workers, &nccl_comm_workers_rank));
    std::cout << "nccl_comm_workers_size: " << nccl_comm_workers_size << std::endl;
    std::cout << "nccl_comm_workers_rank: " << nccl_comm_workers_rank << std::endl;

    std::thread d2h_copy_thread(&AllReduceHelper::d2h_copy_thread_fn, this);
    d2h_copy_thread.detach();
    std::thread h2d_copy_thread(&AllReduceHelper::h2d_copy_thread_fn, this);
    h2d_copy_thread.detach();
    std::thread d2h_completion_thread(&AllReduceHelper::d2h_completion_thread_fn, this);
    d2h_completion_thread.detach();
    std::thread h2d_completion_thread(&AllReduceHelper::h2d_completion_thread_fn, this);
    h2d_completion_thread.detach();
    std::thread reduce_scatter_thread(&AllReduceHelper::reduce_scatter_thread_fn, this);
    reduce_scatter_thread.detach();
    std::thread reduce_scatter_completion_thread(&AllReduceHelper::reduce_scatter_completion_thread_fn, this);
    reduce_scatter_completion_thread.detach();
    std::thread all_gather_thread(&AllReduceHelper::all_gather_thread_fn, this);
    all_gather_thread.detach();
    std::thread all_gather_completion_thread(&AllReduceHelper::all_gather_completion_thread_fn, this);
    all_gather_completion_thread.detach();
    std::thread acc_send_thread(&AllReduceHelper::acc_send_thread_fn, this);
    acc_send_thread.detach();
    std::thread acc_recv_thread(&AllReduceHelper::acc_recv_thread_fn, this);
    acc_recv_thread.detach();
    std::thread first_iteration_thread(&AllReduceHelper::first_iteration_thread_fn, this);
    first_iteration_thread.detach();
    std::thread dummy_copy_thread(&AllReduceHelper::dummy_copy_thread_fn, this);
    dummy_copy_thread.detach();
    std::thread begin_xla_all_reduce_thread(&AllReduceHelper::begin_xla_all_reduce_thread_fn, this);
    begin_xla_all_reduce_thread.detach();
  }
  ~AllReduceHelper() {}
};


static cudaStream_t cudaStream;
static ncclComm_t ncclComm;

HerringBridge::HerringBridge() 
{
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
}

HerringBridge::~HerringBridge() {
}

HerringBridge& HerringBridge::getInstance() {
    static HerringBridge instance;
    return instance;
}

//
// ENTRY POINT
//
void HerringBridge::queue_allreduce(const uint32_t* var_id_gpu, int len, const void* data, void* buffer, void* output) {
    AllReduceHelper& helper = AllReduceHelper::getInstance();
    //helper.begin_xla_all_reduce(var_id_gpu, data, output, len, [this, var_id_gpu]{ this->handle_ar_completion(var_id_gpu) });
    helper.begin_xla_all_reduce(var_id_gpu, data, output, len, [this, var_id_gpu]{});
}

//
// ENTRY POINT
//
void HerringBridge::copy_allreduced_data(const uint32_t* var_id,
        const void* data_in, void* buffer, void* dest, std::function<void()> asyncDoneCallback) {
    // Do CUDA operations from a seperate thread. Just wrap everything and send it to 
    auto task = std::make_shared<PartialAllReduceTask>(PartialAllReduceTask::TYPE_COPY_RESULT, var_id, 0, 
                                                       data_in, buffer, dest, asyncDoneCallback);
    {
        std::lock_guard<std::mutex> mtx(mtx_bg_thread);
        bg_thread_queue.push(task);
    }
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
            // Iterate through the tasks waiting from finisher corresponding to this segment ID
            while(!gradsAwaitingAllreduce[segmentId].empty()) {
                auto task = gradsAwaitingAllreduce[segmentId].front(); gradsAwaitingAllreduce[segmentId].pop();
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
        //unsigned long milliseconds_since_epoch;
        switch(task->task_type) {
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

