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

using namespace std;

#define START_BIN_SIZE 1e-6

#define BIN_SIZE_MULTIPIER 2


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

    // WARNING: calling this will render x to be unusable.
    //
    // Use bins as a discretized dataset after calling this.
    void build_bins(int max_bins) {

        bins.resize(num_features);
        bin_edges.resize(num_features);
        num_bins.resize(num_features);


#pragma omp parallel
        {

// (original index, feature value) pairs
        vector<pair<int,float>> ft_pairs(num_samples);

#pragma omp for schedule(static)
        for (int f = 0; f < num_features; f++) {
            vector<float>& feature_row  = x[f];

            for (int i = 0; i < num_samples; i++) {
                ft_pairs[i].first = i;
                ft_pairs[i].second = feature_row[i];
            }

            sort(ft_pairs.begin(), ft_pairs.end(), [](const pair<int, float>& a,
                        const pair<int, float>& b) {
                return a.second < b.second;
            });

            //sort(feature_row.begin(), feature_row.end());

            if (ft_pairs[0].second == ft_pairs[num_samples - 1].second) {
                cerr << "Constant feature, should somehow handle this" << endl;
                exit(1);
            }

            vector<int>& f_bins = bins[f];
            vector<float>& f_bin_edges = bin_edges[f];

            f_bins.resize(num_samples);
            f_bin_edges.resize(max_bins);

            float bin_size = START_BIN_SIZE;

            while (true) {

                f_bins[ft_pairs[0].first] = 0;
                int curr_bin = 0;

                float prev_v = ft_pairs[0].second;
                float bin_left = prev_v;

                bool failed = false;

                for (int i = 1; i < feature_row.size(); i++) {
                    float v = ft_pairs[i].second;

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

                    f_bins[ft_pairs[i].first] = curr_bin;
                    prev_v = v;
                }

                if (failed) {
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
