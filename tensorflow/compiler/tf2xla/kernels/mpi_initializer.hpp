#ifndef MPI_INITIALIZER_HPP_INCLUDED
#define MPI_INITIALIZER_HPP_INCLUDED

static bool mpi_initialized = false;
static std::mutex mutex_mpi_init;

#include "mpi_helper.h"

class MPIInitializer
{

public:
  static MPIInitializer& getInstance() {
    static MPIInitializer    instance;
    return instance;
  }

  MPI_Comm getSubCluster() {
    return comm_subcluster;
  }

  MPI_Comm getLocalComm() {
    return comm_local;
  }

  MPI_Comm getEightWayComm() {
    return comm_eightway;
  }

private:
  MPIInitializer() {
    MPIInfo mpi_info;
    int provided;
    MPICHECK(MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided));
    assert(provided==3);

    int color = (mpi_info.world_rank < (mpi_info.world_size/2))? 0 : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, mpi_info.world_rank, &comm_subcluster);

    color = mpi_info.world_rank / mpi_info.local_size;
    MPI_Comm_split(MPI_COMM_WORLD, color, mpi_info.world_rank, &comm_local);

    color = mpi_info.local_rank;
    MPI_Comm_split(comm_subcluster, color, mpi_info.world_rank, &comm_eightway);
  }

  ~MPIInitializer() {
    MPICHECK(MPI_Finalize());
  }

  MPI_Comm comm_subcluster, comm_local, comm_eightway;
public:
  MPIInitializer(MPIInitializer const&) = delete;
  void operator=(MPIInitializer const&) = delete;
};

#endif

