#include "mpi_util.h"

static int rank;
static int num_nodes;
static int total_samples;

void init_mpi_util(int rank_, int num_nodes_, int total_samples_) {
    rank = rank_;
    num_nodes = num_nodes_;
    total_samples = total_samples_;
}

int mpi_my_samples() {
    return mpi_num_samples(rank);
}

int mpi_start_index(int rank) {
    int per_node = total_samples / num_nodes;
    return per_node * rank;
}

int mpi_end_index(int rank) {
    if (rank == num_nodes - 1) {
        return total_samples;
    } else {
        int per_node = total_samples / num_nodes;
        return per_node * (rank + 1);
    }
}

int mpi_num_samples(int rank) {
    return mpi_end_index(rank) - mpi_start_index(rank);
}

int mpi_rank() {
    return rank;
}

int mpi_world_size() {
    return num_nodes;
}

bool is_root() {
    return rank == 0;
}


