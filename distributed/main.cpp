//
// Created by Rui Peng on 4/15/17.
//

#include "dataset.h"
#include "decision_tree.h"
#include <mpi.h>
#include <omp.h>
#include "mypprint.hpp"
#include "CycleTimer.h"
#include "mpi_util.h"

using namespace std;

int main(int argc, char** argv) {

    int rank;
    int nproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    init_mpi_util(rank, nproc);

    /*
    DatasetParser dp;

    Dataset d = dp.parse_tsv("../dataset/mslr30k.s123.tab.txt", 136, 5);

    // Dataset d = dp.parse_tsv("../dataset/iris.data.tab.txt", 4, 3);
    // Dataset d = dp.parse_tsv("../dataset/wiki.txt", 3, 2);

    cout << "d.num_samples:" << d.num_samples << endl;
    cout << "d.num_features:" << d.num_features << endl;
    cout << "d.num_classs:" << d.num_classes << endl;
    // cout << "d.y " << d.y << endl;
    // cout << "d.x[0]" << d.x[0] << endl;
    // */

    /*
    cout << "bins: " << d.bins << endl;
    cout << "num bins: " << d.num_bins << endl;
    cout << "bin edges: " << d.bin_edges << endl;
    cout << "bin dists: " << d.bin_dists << endl;
    */

    //DecisionTree dt = DecisionTree(255);
    /*
    cout << "training started" << endl;
    double _t = CycleTimer::currentSeconds();
    dt.train(d);
    cout << "training done" << endl;
    cout << "training has taken " << (CycleTimer::currentSeconds() - _t) << "s" << " rank: " << rank <<  endl;
    //cout << "test on sample 0, predicted label: " << dt.test_single_sample(d, 0) << endl;
    cout << "test on training set, accuracy: " << dt.test(d) << endl;

    Dataset t = dp.parse_tsv("../dataset/mslr30k.s5.tab.txt", 136, 5);
    cout << "test on test set, accuracy: " << dt.test(t) << endl;
    */

    mpi_print("hello world!");

    int number;

    if (mpi_rank() == 0) {
        number = -1;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (mpi_rank() == 1) {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        mpi_print("Process 1 received number:", number);

    }

    MPI_Finalize();
}
