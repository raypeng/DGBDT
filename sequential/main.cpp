//
// Created by Rui Peng on 4/15/17.
//

#include "dataset.h"
#include "decision_tree.h"
#include "mypprint.hpp"
#include "CycleTimer.h"

using namespace std;

int main() {

    DatasetParser dp;

    Dataset d = dp.parse_tsv("../dataset/big/123.tab.txt", 136, 5);

    // Dataset d = dp.parse_tsv("../dataset/iris.data.tab.txt", 4, 3);
    // Dataset d = dp.parse_tsv("../dataset/wiki.txt", 3, 2);

    cout << "d.num_samples:" << d.num_samples << endl;
    cout << "d.num_features:" << d.num_features << endl;
    cout << "d.num_classs:" << d.num_classes << endl;
    // cout << "d.y " << d.y << endl;
    // cout << "d.x[0]" << d.x[0] << endl;

    cout << "starting building bins" << endl;
    float bin_start = CycleTimer::currentSeconds();

    d.build_bins(255);

    float bin_end = CycleTimer::currentSeconds();
    float bin_time = bin_end - bin_start;
    cout << "building bins took: " << bin_time << " seconds" << endl;

    /*
    cout << "bins: " << d.bins << endl;
    cout << "num bins: " << d.num_bins << endl;
    cout << "bin edges: " << d.bin_edges << endl;
    cout << "bin dists: " << d.bin_dists << endl;
    */

    DecisionTree dt = DecisionTree(255);
    cout << "training started" << endl;
    double _t = CycleTimer::currentSeconds();
    dt.train(d);
    cout << "training done" << endl;
    cout << "training has taken " << (CycleTimer::currentSeconds() - _t) + bin_time << "s" << endl;
    cout << "test on sample 0, predicted label: " << dt.test_single_sample(d, 0) << endl;
    cout << "test on training set, accuracy: " << dt.test(d) << endl;

    Dataset t = dp.parse_tsv("../dataset/big/5.tab.txt", 136, 5);
    cout << "test on test set, accuracy: " << dt.test(t) << endl;
    cout << "building bins took: " << bin_time << " seconds" << endl;
}
