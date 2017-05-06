#include "mpi_util.h"

static int rank;
static int num_nodes;

void init_mpi_util(int rank_, int num_nodes_) {
    rank = rank_;
    num_nodes = num_nodes_;
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


