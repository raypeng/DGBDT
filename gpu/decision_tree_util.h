#ifndef DECISION_TREE_UTIL_H
#define DECISION_TREE_UTIL_H

#include <iostream>
#include <cuda.h>
#include <vector>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>

#include "CycleTimer.h"
#include "decision_tree.h"

using namespace std;


void update_smaller_bin_dist(vector<vector<int>>& bins,
			     vector<vector<vector<int>>>& smaller_bin_dist,
			     vector<int>& indices,
			     vector<int>& labels,
			     vector<int>& y,
			     vector<int>& num_bins,
			     int start_index,
			     int end_index,
			     int*& d_bins,
			     int*& d_bin_dist,
			     int*& d_indices,
			     int*& d_y,
			     int*& h_bin_dist,
			     int num_features_gpu,
			     int max_bins,
			     int num_classes);

#endif
