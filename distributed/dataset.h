//
// Created by Rui Peng on 4/15/17.
//

#ifndef SEQUENTIAL_DATASET_H
#define SEQUENTIAL_DATASET_H

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "tree.h"
#include "bindist.h"
#include "mpi_util.h"

using namespace std;

#define START_BIN_SIZE 1e-6

#define BIN_SIZE_MULTIPIER 2

struct DataInfo {
    int label;
    int index;
    float v;
};

struct Dataset {
    int num_features;
    int num_samples;
    int num_classes;
    // x: num_features by num_samples
    vector<vector<float>> x;

    // y: num_samples
    vector<int> y;

    // num_features by num_samples
    vector<vector<int>> bins;

    // number of bins for each feature
    vector<int> num_bins;

    // bin edges to act as split indices for each feature
    vector<vector<float>> bin_edges;

    int max_bins;

    // WARNING: calling this will render x to be unusable.
    //
    // Use bins as a discretized dataset after calling this.
    void build_bins(int max_bins_, TreeNode* root) {
        max_bins = max_bins_;

        BinDist& bin_dist = root->setup_bin_dist(num_features, max_bins_, num_classes);

        bins.resize(num_features);
        bin_edges.resize(num_features);
        num_bins.resize(num_features);

#pragma omp parallel
        {

// (original index, feature value) pairs
        vector<DataInfo> data_infos(num_samples);

#pragma omp for schedule(static)
        for (int f = 0; f < num_features; f++) {
            vector<float>& feature_row  = x[f];

            for (int i = 0; i < num_samples; i++) {
                data_infos[i].index = i;
                data_infos[i].label = y[i];
                data_infos[i].v = feature_row[i];
            }

            sort(data_infos.begin(), data_infos.end(), [](const DataInfo& a,
                        const DataInfo& b) {
                return a.v < b.v;
            });

            //sort(feature_row.begin(), feature_row.end());

            if (data_infos[0].v == data_infos[num_samples - 1].v) {
                cerr << "Constant feature, should somehow handle this" << endl;
                exit(1);
            }

            vector<int>& f_bins = bins[f];
            vector<float>& f_bin_edges = bin_edges[f];

            f_bins.resize(num_samples);
            f_bin_edges.resize(max_bins);

            float bin_size = START_BIN_SIZE;

            while (true) {

                int first_index = data_infos[0].index;
                f_bins[first_index] = 0;
                int curr_bin = 0;

                bin_dist.inc(f, curr_bin, y[first_index]);

                float prev_v = data_infos[0].v;
                float bin_left = prev_v;

                bool failed = false;

                for (int i = 1; i < feature_row.size(); i++) {
                    DataInfo& data_info = data_infos[i];

                    float v = data_info.v;

                    // Check if we need to create a new bin.
                    if (v > bin_left + bin_size) {
                        // Can't fit all values into current sized bins.
                        if (curr_bin == max_bins - 1) {
                            failed = true;
                            break;
                        }

                        f_bin_edges[curr_bin] = prev_v;
                        bin_left = v;
                        curr_bin++;
                    }

                    f_bins[data_info.index] = curr_bin;
                    bin_dist.inc(f, curr_bin, data_info.label);
                    prev_v = v;
                }

                if (failed) {
                    bin_dist.reset(f);
                    bin_size *= BIN_SIZE_MULTIPIER;
                } else {
                    f_bin_edges[curr_bin] = prev_v;

                    int n = curr_bin + 1;
                    num_bins[f] = n;
                    f_bin_edges.resize(n);

                    break;
                }
            }
        }

    }
    }
};

class DatasetParser {
private:
    map<string, int> label_map;
public:
    Dataset parse_tsv(string file_path, int num_features, int num_classes);
};

#endif //SEQUENTIAL_DATASET_H
