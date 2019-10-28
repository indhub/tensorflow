#ifndef MPI_HELPER_HPP_INCLUDED
#define MPI_HELPER_HPP_INCLUDED

#include <cstdlib>
#include <vector>
#include <iostream>
#include <sstream>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


class MPIHelper {

private:

  int get_env(const char* key) {
    if(const char* env_p = std::getenv(key))
      return atoi(env_p);
    return 0;
  }

public:

  int get_world_size() {
    return get_env("OMPI_COMM_WORLD_SIZE");
  }

  int get_world_rank() {
    return get_env("OMPI_COMM_WORLD_RANK");
  }

  int get_local_size() {
    return get_env("OMPI_COMM_WORLD_LOCAL_SIZE");
  }

  int get_local_rank() {
    return get_env("OMPI_COMM_WORLD_LOCAL_RANK");
  }

};

struct MPIInfo {
  int world_size, world_rank, local_size, local_rank, num_nodes, node_index;
  int num_worker_nodes, num_acc_nodes;
  std::vector<int> worker_ranks;
  std::vector<int> acc_ranks;
  MPIHelper mpi_helper;

  std::string vec_to_str(std::vector<int> vec) {
    std::stringstream ss;
    ss << "[";
    for(int i: vec) ss << i << ", ";
    ss << "]";
    return ss.str();
  }

  MPIInfo()
    : world_size(mpi_helper.get_world_size()),
      world_rank(mpi_helper.get_world_rank()),
      local_size(mpi_helper.get_local_size()),
      local_rank(mpi_helper.get_local_rank())
  {
    bool do_log = (world_rank==0 || world_rank==world_size/2);

    num_nodes = world_size / local_size;
    node_index = world_rank / local_size;

    num_worker_nodes = num_nodes / 2;
    num_acc_nodes = num_nodes / 2;

    if(num_nodes==1) {
      return;
    }

    bool this_is_worker = world_rank < world_size/2;

    if(this_is_worker) {
      int num_acc_nodes_per_rank = num_acc_nodes / local_size;
      if(do_log) std::cout << "num_acc_nodes_per_rank: " << num_acc_nodes_per_rank << std::endl;
      int first_acc_node = num_worker_nodes + local_rank * num_acc_nodes_per_rank;
      if(do_log) std::cout << "first_acc_node: " << first_acc_node << std::endl;

      int worker_node_index = node_index;
      int num_worker_nodes_per_acc_rank = num_worker_nodes / local_size;
      if(do_log) std::cout << "num_worker_nodes_per_acc_rank: " <<
        num_worker_nodes_per_acc_rank << std::endl;
      int peer_acc_local_rank = worker_node_index / num_worker_nodes_per_acc_rank;
      if(do_log) std::cout << "peer_acc_local_rank: " << peer_acc_local_rank << std::endl;

      for(int i=0; i<num_acc_nodes_per_rank; i++) {
        acc_ranks.push_back(first_acc_node*local_size + i*local_size + peer_acc_local_rank);
      }
      if(do_log) std::cout << "acc_ranks: " << vec_to_str(acc_ranks) << std::endl;
    } else { //this is an accelerator rank
      int num_worker_nodes_per_rank = num_worker_nodes / local_size;
      int first_worker_node = local_rank * num_worker_nodes_per_rank;

      int acc_node_index = node_index - num_worker_nodes;
      int num_acc_nodes_per_worker_rank = num_acc_nodes / local_size;
      int peer_worker_local_rank = acc_node_index / num_acc_nodes_per_worker_rank;

      for(int i=0; i<num_worker_nodes_per_rank; i++) {
        worker_ranks.push_back(first_worker_node*local_size + i*local_size + peer_worker_local_rank);
      }
      if(do_log) std::cout << "worker ranks: " << vec_to_str(worker_ranks) << std::endl;
    }
  }
};

#endif

