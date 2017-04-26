//
// Created by Rui Peng on 4/15/17.
//

#include <iostream>

#include "dataset.h"
#include "mypprint.hpp"

using namespace std;


Dataset DatasetParser::parse_tsv(string file_path, int num_features, int num_classes) {
    cout << "parsing started\n";
    fstream infile(file_path);
    Dataset d;
    d.num_features = num_features;
    d.num_classes = num_classes;
    d.x.resize(d.num_features);
    string label, curr;
    int num_labels = 0;
    while (!infile.eof()) {
        infile >> label;
        if (label_map.find(label) == label_map.end()) {
            label_map[label] = num_labels++;
        }
        d.y.push_back(label_map[label]);
        for (int i = 0; i < d.num_features; i++) {
            infile >> curr;
            d.x[i].push_back(stod(curr));
        }
    }
    d.num_samples = d.y.size();
    cout << "parsing done\n";
    return d;
}
