//
// Created by Rui Peng on 4/15/17.
//

#ifndef SEQUENTIAL_DATASET_H
#define SEQUENTIAL_DATASET_H

#include <vector>
#include <map>
#include <sstream>
#include <fstream>

using namespace std;


struct Dataset {
    int num_features;
    int num_samples;
    int num_classes;
    // x: num_features by num_samples
    vector<vector<float>> x;
    // y: num_samples
    vector<int> y;
};

Dataset parse_tsv(string file_path, int num_features, int num_classes);

#endif //SEQUENTIAL_DATASET_H
