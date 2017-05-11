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
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>
#include "tree.h"
#include "bindist.h"
#include "mpi_util.h"
#include "util.h"

using namespace std;

#define START_BIN_SIZE 1e-6

#define BIN_SIZE_MULTIPIER 2

struct DataInfo {
    int label;
    int index;
    float v;
};

struct DistributedBin {
    int bin;
    int rank;
    float v;

    DistributedBin(int bin_, int rank_, float v_) {
        bin = bin_;
        rank = rank_;
        v = v_;
    }
};

bool cmp_bin(const DistributedBin& a, const DistributedBin& b);

ostream& operator<<(ostream& os, const DistributedBin& b);

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
    vector<vector<float>> bin_ends;

    // Possible split points, used for distributed setting when merging histograms.
    // distributed_bins[feature][i] to access ith bin for given feature
    vector<vector<DistributedBin>> distributed_bins;

    // num bins for each feature on each node: distributed_num_bins[rank][feature]
    vector<vector<int>> distributed_num_bins;

    vector<BinDist*> distributed_bin_dist;

    bool distributed;

    int max_bins;

    // WARNING: calling this will render x to be unusable.
    //
    // Use bins as a discretized dataset after calling this.
    void build_bins(int max_bins_, TreeNode* root) {
        max_bins = max_bins_;

        BinDist& bin_dist = root->setup_bin_dist(num_features, max_bins_, num_classes);

        bins.resize(num_features);
        bin_ends.resize(num_features);
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
            vector<float>& f_bin_ends = bin_ends[f];

            f_bins.resize(num_samples);
            f_bin_ends.resize(max_bins);

            float bin_size = START_BIN_SIZE;

            while (true) {

                int curr_bin = 0;
                float bin_left = data_infos[0].v;
                float prev_v = bin_left;

                int first_index = data_infos[0].index;
                f_bins[first_index] = 0;
                bin_dist.inc(f, curr_bin, y[first_index]);

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

                        f_bin_ends[curr_bin] = prev_v;
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
                    f_bin_ends[curr_bin] = prev_v;

                    int n = curr_bin + 1;
                    num_bins[f] = n;
                    f_bin_ends.resize(n);

                    break;
                }
            }
        }
        }

        mpi_print("entering distributed part");
        if (distributed) {

            int num_bin_tag = 0;
            int bin_end_tag = 2;


            distributed_bin_dist.resize(mpi_world_size());
            if (mpi_rank() == 0) {
                // Set root's distributed bin dist locally from the node.
                for (int i = 1; i < mpi_world_size(); i++) {
                    distributed_bin_dist[i] = new BinDist(num_features, max_bins, num_classes);
                }

                distributed_num_bins.resize(mpi_world_size(), vector<int>(num_features));
                distributed_num_bins[0] = num_bins;

                for (int r = 1; r < mpi_world_size(); r++) {
                    MPI_Recv(distributed_num_bins[r].data(), num_features, MPI_INT, r, num_bin_tag,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                distributed_bins.resize(num_features);

                for (int f = 0; f < num_features; f++) {
                    vector<DistributedBin>& bins = distributed_bins[f];

                    int nbins = distributed_num_bins[0][f];

                    for (int i = 0; i < nbins; i++) {
                        bins.push_back(DistributedBin(i, 0, bin_ends[f][i]));
                    }

                    vector<float> remote_bin_ends(max_bins);

                    for (int r = 1; r < mpi_world_size(); r++) {
                        nbins = distributed_num_bins[r][f];

                        // TODO: change these tags if we use asyncrhonous receives.
                        //mpi_print("receiving starts for feature: ", f);
                        //mpi_print("receiving ends for feature: ", f);
                        MPI_Recv(remote_bin_ends.data(), nbins, MPI_FLOAT, r, MPI_ANY_TAG,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        for (int i = 0; i < nbins; i++) {
                            bins.push_back(DistributedBin(i, r, remote_bin_ends[i]));
                        }
                    }
                    sort(bins.begin(), bins.end(), cmp_bin);
                }
            } else {
                MPI_Send(num_bins.data(), num_features, MPI_INT, 0, num_bin_tag, MPI_COMM_WORLD);

                for (int f = 0; f < num_features; f++) {
                    MPI_Send(bin_ends[f].data(), num_bins[f], MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
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
    Dataset distributed_parse_tsv(string file_path, int num_features, int num_classes);
};

#endif //SEQUENTIAL_DATASET_H
