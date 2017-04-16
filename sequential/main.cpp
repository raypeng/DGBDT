//
// Created by Rui Peng on 4/15/17.
//

#include "dataset.h"
#include "decision_tree.h"
#include "mypprint.hpp"

using namespace std;

int main() {
    string file_path = "../../dataset/iris.data.tab.txt";
    Dataset d = parse_tsv(file_path, 4, 3);
//    string file_path = "../../dataset/wiki.txt";
//    Dataset d = parse_tsv(file_path, 3, 2);

    print(d.num_samples, "d.num_samples:");
    print(d.num_features, "d.num_features");
    print(d.num_classes, "d.num_classes");
    print(d.y, "d.y");
    print(d.x[0], "d.x[0]");

    DecisionTree dt = DecisionTree(1000);
    dt.train(d);
    print(dt.test_single_sample(d, 0), "test on sample 0, predicted label:");
    print(dt.test(d), "test on training set, accuracy:");
}
