//
// Created by Rui Peng on 4/15/17.
//

#include <iostream>

#include "dataset.h"
#include "mypprint.hpp"
#include "mpi_util.h"

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

Dataset DatasetParser::distributed_parse_tsv(string file_path, int num_features, int num_classes) {
    mpi_print("distributed parsing started");
    fstream infile(file_path);
    Dataset d;
    d.distributed = true;
    d.num_features = num_features;
    d.num_classes = num_classes;
    d.x.resize(d.num_features);
    string label, curr;
    int num_labels = 0;
    int counter = 0;
    int start = mpi_start_index(mpi_rank());
    int end = mpi_end_index(mpi_rank());

    while (!infile.eof()) {
        infile >> label;
        if (label_map.find(label) == label_map.end()) {
            label_map[label] = num_labels++;
        }
        if (counter >= start && counter < end) {
            d.y.push_back(label_map[label]);
            for (int i = 0; i < d.num_features; i++) {
                infile >> curr;
                d.x[i].push_back(stod(curr));
            }
        } else {
            for (int i = 0; i < d.num_features; i++) {
                infile >> curr;
            }
        }

        counter++;
    }
    d.num_samples = d.y.size();
    mpi_print("parsing done");
    return d;
}

ostream& operator<<(ostream& os, const DistributedBin& b) {
    os << "(" << b.bin << ", " << b.bin_start << ", "
        << b.rank << ", " << b.v << ")";
}

bool cmp_bin(const DistributedBin& a, const DistributedBin& b) {
    if (float_equal(a.v, b.v)) {
        return a.bin_start;
    }
    return a.v < b.v;
}

