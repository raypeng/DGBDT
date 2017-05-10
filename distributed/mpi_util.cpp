#include "mpi_util.h"
#include "tree.h"

static int my_rank;
static int num_nodes;
static int total_samples;
static MPI_Datatype type;

static void setupType(MPI_Datatype *type) {
    MPI_Datatype temp_type;
    int blocks[5] = {1,1,1,1,1};
    MPI_Datatype types[5] = {MPI_INT, MPI_FLOAT, MPI_FLOAT,
		MPI_FLOAT, MPI_FLOAT};
    MPI_Aint offsets[5];
    offsets[0] = offsetof(SplitInfo, split_feature_id);
    offsets[1] = offsetof(SplitInfo, min_entropy);
    offsets[2] = offsetof(SplitInfo, split_threshold);
    offsets[3] = offsetof(SplitInfo, left_entropy);
    offsets[4] = offsetof(SplitInfo, right_entropy);

    MPI_Type_create_struct(5, blocks, offsets, types, type);

    /*
    // Resize type to account for struct alignment.
    MPI_Aint lb, extent;
    MPI_Type_get_extent(temp_type, &lb, &extent);
    MPI_Type_create_resized(temp_type, lb, extent, type);
    */

    MPI_Type_commit(type);
}

void init_mpi_util(int rank_, int num_nodes_, int total_samples_) {
    my_rank = rank_;
    num_nodes = num_nodes_;
    total_samples = total_samples_;
	setupType(&type);
}

MPI_Datatype split_info_type() {
	return type;
}

int mpi_my_samples() {
    return mpi_num_samples(my_rank);
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
    return my_rank;
}

int mpi_world_size() {
    return num_nodes;
}

bool is_root() {
    return my_rank == 0;
}


