#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include "iostream"

void init_mpi_util(int rank, int num_nodes, int total_samples);

int mpi_rank();

int mpi_world_size();

bool is_root();

int mpi_my_samples();

int mpi_start_index(int rank);

int mpi_end_index(int rank);

int mpi_num_samples(int rank);

template <class T>
void mpi_print(std::string prefix, T val) {
    std::cout << "[node " << mpi_rank() << "] " << prefix << " " << val << std::endl;
}

template <class T>
void mpi_print(T val) {
    std::cout << "[node " << mpi_rank() << "] " << val << std::endl;
}

#endif
