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

    // Slight hack, hardcode the total samples of the dataset.
    // TODO: Actually chunk the file. For now, each node reads the whole dataset
    // operates on an assigned interval of that dataset
    int total_samples = 2270296;
    //int total_samples = 723411;
    init_mpi_util(rank, nproc, total_samples);

    DatasetParser dp;

    //Dataset d = dp.parse_tsv("../dataset/123.tab.txt", 136, 5);
    Dataset d = dp.distributed_parse_tsv("../dataset/123.tab.txt", 136, 5);
    //Dataset d = dp.distributed_parse_tsv("../dataset/mslr30k.s123.tab.txt", 136, 5);
    //Dataset d = dp.parse_tsv("../dataset/mslr30k.s123.tab.txt", 136, 5);

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

    DecisionTree dt = DecisionTree(255);
    mpi_print("training started");
    double _t = CycleTimer::currentSeconds();
    dt.train(d);
    mpi_print("training done");
    mpi_print("training has taken: ", CycleTimer::currentSeconds() - _t);
    //cout << "test on sample 0, predicted label: " << dt.test_single_sample(d, 0) << endl;
    mpi_print("test on training set, accuracy: ", dt.test(d));

    //Dataset t = dp.parse_tsv("../dataset/mslr30k.s5.tab.txt", 136, 5);
    Dataset t = dp.parse_tsv("../dataset/5.tab.txt", 136, 5);
    mpi_print("test on test set, accuracy: ", dt.test(t));

    MPI_Finalize();
}
