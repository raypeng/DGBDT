//
// Created by Rui Peng on 4/15/17.
//

#include "dataset.h"
#include "decision_tree.h"
#include "mypprint.hpp"

using namespace std;

int main() {
    Dataset d = parse_tsv("../dataset/big/mslr10k.train.tab.txt", 136, 5);

//     string file_path = "../dataset/iris.data.tab.txt";
//     Dataset d = parse_tsv(file_path, 4, 3);
    // string file_path = "../dataset/wiki.txt";
    // Dataset d = parse_tsv(file_path, 3, 2);

    cout << "d.num_samples:" << d.num_samples << endl;
    cout << "d.num_features:" << d.num_features << endl;
    cout << "d.num_classs:" << d.num_classes << endl;
    cout << "d.y " << d.y << endl;
    cout << "d.x[0]" << d.x[0] << endl;

    DecisionTree dt = DecisionTree(5);
    cout << "training started" << endl;
    dt.train(d);
    cout << "training done" << endl;
    cout << "test on sample 0, predicted label:" << dt.test_single_sample(d, 0) << endl;
    cout << "test on training set, accuracy:" << dt.test(d) << endl;

    Dataset t = parse_tsv("../dataset/big/mslr10k.test.tab.txt", 136, 5);
    cout << "test on test set, accuracy:" << dt.test(t) << endl;

}
