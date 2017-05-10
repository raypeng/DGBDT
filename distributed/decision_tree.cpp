//
// Created by Rui Peng on 4/15/17.
//

#include <list>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iterator>
#include <map>
#include <mpi.h>

#include "decision_tree.h"
#include "util.h"
#include "mypprint.hpp"
#include "CycleTimer.h"
#include "mpi_util.h"

#define INFO_GAIN_THRESHOLD 1e-3

using namespace std;

SplitInfo DecisionTree::find_new_entropy_by_split_on_feature(Dataset& d, int feature_id, TreeNode* curr_node) {
    // entropy before split does not affect comparing info gain values across different features to split
    // so only pick smallest total entropy after split
    // equivalent to taking entropy before split as zero
    _t -= CycleTimer::currentSeconds();

    BinDist& bin_dist = curr_node->get_bin_dist();

    vector<BinDist*> distributed_bin_dist = d.distributed_bin_dist;

    // Accumulate total sample counts, individual bin sample counts,
    // and class distribution across all nodes.
    int N = 0;
    vector<vector<int>> bin_counts(mpi_world_size(), vector<int>(d.max_bins));
    vector<int> class_dist(d.num_classes);

    for (int r = 0; r < mpi_world_size(); r++) {
        int num_bins = d.distributed_num_bins[r][feature_id];

        for (int i = 0; i < num_bins; i++) {
            int count = 0;

            for (int j = 0; j < d.num_classes; j++) {
                int val = distributed_bin_dist[r]->get(feature_id,i,j);
                count += val;
                class_dist[j] += val;
            }

            bin_counts[r][i] = count;
            N += count;
        }
    }

    _t += CycleTimer::currentSeconds();
    _t2 -= CycleTimer::currentSeconds();

    vector<int> left_dist(d.num_classes,0);
    const vector<float>& bin_ends = d.bin_ends[feature_id];

    float min_entropy = numeric_limits<float>::max();
    int total_samples_left = 0;
    float best_left_entropy = -1, best_right_entropy = -1, best_split_thres = -1;

    vector<DistributedBin> dbins = d.distributed_bins[feature_id];

    // Map (rank,bin) to the bin to store active bins while scanning.
    //
    // Since there are only O(num_bins * num_nodes) many bins, we use map here
    // to avoid the overhead of an unordered_map on repeated insertions,
    // deletes, and iterations across the map. unordered_map may or may not be
    // faster, but it's probably about the same.
    map<pair<int,int>, DistributedBin> active_bins;

    for (int i = 0; i < dbins.size(); i++) {

        DistributedBin dbin = dbins[i];

        // Find first non-empty distributed bin.
        while (i < dbins.size() && bin_counts[dbin.rank][dbin.bin] == 0) {
            i++;
            dbin = dbins[i];
        }

        // No point splitting on the end.
        if (i == dbins.size()) {
            break;
        }

        if (dbin.bin_start) {
            // add to active bins
            active_bins.insert({{dbin.rank,dbin.bin},dbin});

        } else {

            BinDist* bin_dist = distributed_bin_dist[dbin.rank];

            vector<int> accumulate_dist(d.num_classes,0);

            // on bin ends, first find the first non-repeating bin end, then perform the
            // entropy calculation of a potential threshold split point.
            while (i < dbins.size() - 1
                    && !(dbins[i+1].bin_start)
                    && float_equal(dbin.v, dbins[i+1].v)) {
                add_vector(accumulate_dist, accumulate_dist, bin_dist->head(feature_id, dbin.bin));
                total_samples_left += bin_counts[dbin.rank][dbin.bin];

                active_bins.erase({dbin.rank,dbin.bin});
                i++;
                dbin = dbins[i];
				bin_dist = distributed_bin_dist[dbin.rank];
            }

            add_vector(left_dist, left_dist, accumulate_dist);
            add_vector(left_dist, left_dist, bin_dist->head(feature_id,dbin.bin));

            // TODO: interpolation of active bins

            active_bins.erase({dbin.rank,dbin.bin});
            total_samples_left += bin_counts[dbin.rank][dbin.bin];
            int total_samples_right = N - total_samples_left;

            float left_entropy = 0, right_entropy = 0;
            for (int c = 0; c < d.num_classes; c++) {
                int left_samples_per_class = left_dist[c];
                if (left_samples_per_class != 0) {
                    float left_frac_yes = 1. * left_samples_per_class / total_samples_left;
                    left_entropy -= left_frac_yes * log2(left_frac_yes);
                }
                int right_samples_per_class = class_dist[c] - left_samples_per_class;
                if (right_samples_per_class != 0) {
                    float right_frac_yes = 1. * right_samples_per_class / total_samples_right;
                    right_entropy -= right_frac_yes * log2(right_frac_yes);
                }
            }
            float curr_entropy = 0;
            if (total_samples_left != 0) {
                curr_entropy += (1. * total_samples_left / N) * left_entropy;
           }
            if (total_samples_right != 0) {
                curr_entropy += (1. * total_samples_right / N) * right_entropy;
            }
            if (curr_entropy < min_entropy) {
                min_entropy = curr_entropy;
                best_split_thres = dbin.v;
                best_left_entropy = left_entropy;
                best_right_entropy = right_entropy;
            }

            }

        }

    _t2 += CycleTimer::currentSeconds();
    // cerr << "find_split inner main loops entropy \t taking " << CycleTimer::currentSeconds() - _tt << "s" << endl;
    return {feature_id, min_entropy, best_split_thres, best_left_entropy, best_right_entropy};
}

static SplitInfo bcast_split_info(SplitInfo info) {
	MPI_Bcast(&info, 1, split_info_type(), 0, MPI_COMM_WORLD);
	return info;
}

SplitInfo DecisionTree::find_split(Dataset& d, vector<int>& indices, TreeNode* curr_node) {
    if (none_of(indices.begin(), indices.end(), [](bool x) {
        return x;
    })) {
        // indices all false - no data to split
        // should not happen
        cerr << "find_split ending up no data, still deciding what to do with this case";
        abort();
        return {MinSize, -1};
    }

	// Stop if all nodes say stop.
	pair<bool, NodeStatus> stop_result = should_stop(curr_node);
	bool to_stop;
	MPI_Allreduce(&stop_result.first, &to_stop, 1, MPI_BYTE, MPI_LAND, MPI_COMM_WORLD);
	if (to_stop) {
		mpi_print("detected should stop");
		return {stop_result.second, -1, -1, -1, -1};
	}

	mpi_print("searching for feature to split");
    if (mpi_rank() == 0) {
        int best_feature = -1;
        float min_entropy = numeric_limits<float>::max();
        float best_left_entropy = -1, best_right_entropy = -1, best_split_thres = -1;

        _t = 0, _t2 = 0;
        double _tt = CycleTimer::currentSeconds();

		// avoid copying our own bin distribution
		BinDist& bin_dist = curr_node->get_bin_dist();
		d.distributed_bin_dist[0] = &bin_dist;
		for (int i = 1 ; i < d.distributed_bin_dist.size(); i++) {
			// Use feature_id as tag to differentiate between different features
			// if we parallelize this across different features.
			MPI_Recv(d.distributed_bin_dist[i]->head(), bin_dist.size(), MPI_INT, i,
					0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

        mpi_print("receiving bin dists taking: ", CycleTimer::currentSeconds() - _tt);

        for (int f = 0; f < d.num_features; f++) {
            auto curr_split_info = find_new_entropy_by_split_on_feature(d, f, curr_node);
            if (curr_split_info.min_entropy < min_entropy) {
                min_entropy = curr_split_info.min_entropy;
                best_split_thres = curr_split_info.split_threshold;
                best_left_entropy = curr_split_info.left_entropy;
                best_right_entropy = curr_split_info.right_entropy;
                best_feature = f;
            }
        }
        mpi_print("find_split outer main loop \t taking ", CycleTimer::currentSeconds() - _tt);
        /*
        cerr << "find_split outer initializing bin counts \t taking " << _t << "s" << endl;
        cerr << "find_split outer finding split index \t taking " << _t2 << "s" << endl;
        */
        // some extra stuff to check if the SplitInfo meets our requirement
        //
        float info_gain = curr_node->get_entropy() - min_entropy;
        if (info_gain < INFO_GAIN_THRESHOLD) {
            return bcast_split_info({NoGain, -1, -1, -1, -1});
        }
        return bcast_split_info({best_feature, min_entropy, best_split_thres, best_left_entropy, best_right_entropy});
    } else {
		BinDist& bin_dist = curr_node->get_bin_dist();

		MPI_Send(bin_dist.head(), bin_dist.size(), MPI_INT, 0,
				0, MPI_COMM_WORLD);

        /*
        MPI_Request request;
		MPI_Isend(bin_dist.head(), bin_dist.size(), MPI_INT, 0,
				0, MPI_COMM_WORLD, &request);
                */

		SplitInfo info;
		MPI_Bcast(&info, 1, split_info_type(), 0, MPI_COMM_WORLD);
		return info;
    }
}

// Partitions indices according to the node.
//
// Returns the index of the split.
int DecisionTree::split_data(vector<int>& indices, const Dataset& d, TreeNode* curr_node) {

    int feature_id = curr_node->split_info.split_feature_id;
    float split_point = curr_node->split_info.split_threshold;

    double _t = CycleTimer::currentSeconds();

    const vector<int>& bins = d.bins[feature_id];
    const vector<float>& feature_row = d.x[feature_id];
    auto begin = indices.begin() + curr_node->get_left();
    auto end = indices.begin() + curr_node->get_right();
    auto bound = stable_partition(begin, end,
            [&feature_row, &split_point](const int index) {
                return feature_row[index] <= split_point;
            });

    //cerr << "split_data\t taking " << CycleTimer::currentSeconds() - _t << "s" << endl;

    return distance(indices.begin(), bound);
}

pair<bool, NodeStatus> DecisionTree::should_stop(TreeNode* curr) {

    bool perfect_split = true;
    bool found_class = false;
    vector<int>& class_dist = curr->get_class_dist();
    for (int i = 0; i < class_dist.size(); i++) {
        if (class_dist[i] > 0) {
            if (found_class) {
                perfect_split = false;
                break;
            } else {
                found_class = true;
            }
        }
    }
    if (perfect_split) { // all samples are from same label
         return {true, PerfectSplit};
    } else if (curr->get_depth() >= max_depth) {
        return {true, MaxDepth};
    } else if (curr->get_size() <= min_node_size) {
        return {true, MinSize};
    }

    return {false, Ok};
}

DecisionTree::DecisionTree(int max_num_leaves_, int max_depth_,
      int min_node_size_) {
    num_leaves = 0;
    max_num_leaves = max_num_leaves_;
    max_depth = max_depth_;
    min_node_size = min_node_size_;
    root = NULL;
}

void DecisionTree::train(Dataset &d) {

    // Setup root node.

    int curr_node_id = 0;

    float dummy_large_entropy = 1e3; // hack! coz we always split when we start anyways
    root = new TreeNode(curr_node_id++, 0, d.num_samples, dummy_large_entropy, 0, d.num_samples);

    mpi_print("starting building bins");
    float bin_start = CycleTimer::currentSeconds();

    d.build_bins(255, root);

    float bin_end = CycleTimer::currentSeconds();
    float bin_time = bin_end - bin_start;
    mpi_print("building bins took: ", bin_time);

    vector<int>& class_dist = root->get_class_dist();
    class_dist.resize(d.num_classes,0);

    double _ttt = CycleTimer::currentSeconds();

    vector<int> indices(d.num_samples);

#pragma omp parallel
    {
      vector<int> local_class_dist(d.num_classes);
#pragma omp for schedule(static) nowait
      for (int i = 0; i < d.num_samples; i++) {
        local_class_dist[d.y[i]]++;
        indices[i] = i;
      }
      for (int label = 0; label < d.num_classes; label++) {
#pragma omp atomic
	class_dist[label] += local_class_dist[label];
      }
    }

    mpi_print("class_dist root ", CycleTimer::currentSeconds() - _ttt);

    list<TreeNode*> work_queue;
    work_queue.push_back(root);

    // Pop leaves from work queue to split in a BFS order.
    double _t; // DEBUG
    while (num_leaves + work_queue.size() < max_num_leaves) {
        if (work_queue.empty()) {
            print("done", "work queue empty, exit");
            return;
        }
        // remove a node from work queue to perform split on
        TreeNode* curr = work_queue.front();
        work_queue.pop_front();

        // find split according to the data in curr
        mpi_print("working on splitting node id: ", curr->node_id);
        _t = CycleTimer::currentSeconds();
        curr->split_info = find_split(d, indices, curr);
        /*
        cerr << "find_split for node " << curr->node_id << "\t taking " << CycleTimer::currentSeconds() - _t << "s" << endl;
        */

        if (curr->split_info.split_feature_id < 0) { // no need to split
            num_leaves++;
            switch (curr->split_info.split_feature_id) {
                case PerfectSplit:
					cout << "perfect split already for node " << curr->node_id << endl;
                    break;
                case MaxDepth:
                    cout << "node at max depth with id " << curr->node_id << endl;
                    break;
                case MinSize:
                    cout << "node at min size with id " << curr->node_id << endl;
                    break;
                case NoGain:
                    cout << "node had no gain " << curr->node_id << endl;
                    break;
                default:
                    cout << "node became leaf for unknown reason " << curr->node_id << endl;
            }
            curr->update_majority_label();
            continue;
        }
        mpi_print("split on feature: ", curr->split_info.split_feature_id);
        mpi_print("split on threshold: ", curr->split_info.split_threshold);

        /*
        cout << "split on feature: " << curr->split_info.split_feature_id << endl;
        cout << "split on threshold: " << curr->split_info.split_threshold << endl;
        */

        mpi_print("splitting data");
        // split data into two halves
        int split_index = split_data(indices, d, curr);
        mpi_print("making children");
        // create two child nodes and add to work queue
        // TODO: make the size of the node actually real after partitioning is implemented
        curr->left_child = new TreeNode(curr_node_id++, curr->get_depth()+1, d.num_samples, curr->split_info.left_entropy, curr->left, split_index);
        curr->right_child = new TreeNode(curr_node_id++, curr->get_depth()+1, d.num_samples, curr->split_info.right_entropy, split_index, curr->right);

        double dist_start_time = CycleTimer::currentSeconds();

        // Update class and bin distributions
        vector<int>& left_dist = curr->left_child->get_class_dist();
        vector<int>& right_dist = curr->right_child->get_class_dist();
        vector<int>& curr_dist = curr->get_class_dist();
        left_dist.resize(d.num_classes, 0);
        right_dist.resize(d.num_classes, 0);

        BinDist& left_bin_dist = curr->left_child->setup_bin_dist(d.num_features, d.max_bins, d.num_classes);
        BinDist& right_bin_dist = curr->right_child->setup_bin_dist(d.num_features, d.max_bins, d.num_classes);
        BinDist& curr_bin_dist = curr->get_bin_dist();

        int left_size = split_index - curr->left;
        int right_size = curr->right - split_index;

        BinDist& smaller_bin_dist = right_size > left_size ? left_bin_dist : right_bin_dist;
        BinDist& larger_bin_dist = right_size > left_size ? right_bin_dist : left_bin_dist;
        vector<int>& smaller_dist = right_size > left_size ? left_dist : right_dist;
        vector<int>& larger_dist = right_size > left_size ? right_dist : left_dist;
        int start_index;
        int end_index;

        if (right_size > left_size) {
            start_index = curr->left;
            end_index = split_index;
        } else {
            start_index = split_index;
            end_index = curr->right;
        }

        vector<int> labels(end_index - start_index);
        for (int i = start_index; i < end_index; i++) {
            int index = indices[i];
            int label = d.y[index];
            labels[i - start_index] = label;
            smaller_dist[label]++;
        }

	/*
#pragma omp parallel
	    {
	      vector<int> local_smaller_dist(d.num_classes);
#pragma omp for schedule(static) nowait
	      for (int i = start_index; i < end_index; i++) {
		int index = indices[i];
		int label = d.y[index];
		labels[i - start_index] = label;
		local_smaller_dist[label]++;
	      }
	      for (int label = 0; label < d.num_classes; label++) {
#pragma omp atomic
		smaller_dist[label] += local_smaller_dist[label];
	      }
	    }
	*/

#pragma omp parallel for schedule(static)
        for (int f = 0; f < d.num_features; f++) {

            vector<int>& bins = d.bins[f];

            for (int i = start_index; i < end_index; i++) {
                int index = indices[i];
                int label = labels[i - start_index];

                int bin = bins[index];
                smaller_bin_dist.inc(f,bin,label);
            }
        }


        // Calculate right_dist by using left_dist
        subtract_vector(larger_dist, curr_dist, smaller_dist);

        // Similarly for bin dist
        larger_bin_dist.diff(curr_bin_dist, smaller_bin_dist);

        double dist_end_time = CycleTimer::currentSeconds();

        mpi_print("calculating children dist took ", dist_end_time - dist_start_time);

		mpi_print("left size: ", left_size);
		mpi_print("right size: ", right_size);
        //cout<< "split index: " << split_index << endl;
        // cout<< "left size: " << curr->left_child->right - curr->left_child->left << endl;
        // cout<< "right size: " << curr->right_child->right - curr->right_child->left << endl;
        work_queue.push_back(curr->left_child);
        work_queue.push_back(curr->right_child);
    }
    for (TreeNode* left_over_node : work_queue) {
        left_over_node->update_majority_label();
    }

    mpi_print("building bins took: ", bin_time);
}

int DecisionTree::test_single_sample(const Dataset& d, int sample_id) {
    TreeNode* curr = root;
    while (curr) {
        int curr_feature = curr->split_info.split_feature_id;
        float curr_thres = curr->split_info.split_threshold;
        if (curr->is_leaf()) {
            return curr->majority_label;
        }
        if (d.x[curr_feature][sample_id] <= curr_thres) {
            if (curr->left_child) {
                curr = curr->left_child;
            } else {
                return curr->majority_label;
            }
        } else {
            if (curr->right_child) {
                curr = curr->right_child;
            } else {
                return curr->majority_label;
            }
        }
    }
    // should not reach here
    cerr << "test_single_sample impossible case";
    abort();
}

float DecisionTree::test(const Dataset& d) {
    int num_correct = 0;
    for (int sample_id = 0; sample_id < d.num_samples; sample_id++) {
        int predicted_label = test_single_sample(d, sample_id);
        if (predicted_label == d.y[sample_id]) {
            num_correct++;
        }
    }
    return 1. * num_correct / d.num_samples;
}
