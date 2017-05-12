//
// Created by Rui Peng on 4/15/17.
//

#include <list>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iterator>
#include <omp.h>

#include "decision_tree.h"
#include "mypprint.hpp"
#include "util.h"
#include "CycleTimer.h"

#include "decision_tree_util.h"

#define INFO_GAIN_THRESHOLD 1e-3
#define HYBRID_CUTOFF 10000


using namespace std;

void update_smaller_bin_dist_cpu(vector<vector<int>>& bins,
				 vector<vector<vector<int>>>& smaller_bin_dist,
				 vector<int>& indices,
				 vector<int>& labels,
				 int start_index,
				 int end_index,
				 int start_feature,
				 int end_feature) {
    double cpu_t = CycleTimer::currentSeconds();
#pragma omp parallel for schedule(static)
  for (int f = start_feature; f < end_feature; f++) {
    vector<int>& feature_bins = bins[f];
    vector<vector<int>>& bin_dists = smaller_bin_dist[f];
    for (int i = start_index; i < end_index; i++) {
      int index = indices[i];
      int label = labels[i - start_index];
      int bin = feature_bins[index];
      bin_dists[bin][label]++;
    }
  }
  cout << "exiting cpu: " << end_feature - start_feature << " features\t" << CycleTimer::currentSeconds() - cpu_t << endl;

}

SplitInfo DecisionTree::find_new_entropy_by_split_on_feature(const Dataset& d, vector<int>& indices, int feature_id, TreeNode* curr_node) {
    // entropy before split does not affect comparing info gain values across different features to split
    // so only pick smallest total entropy after split
    // equivalent to taking entropy before split as zero
    print(feature_id, "find_split_feature feature_id");
    _t -= CycleTimer::currentSeconds();

    vector<vector<int>>& bin_dists = curr_node->bin_dists_for_feature(feature_id);

    int N = curr_node->right - curr_node->left;
    int num_bins = d.num_bins[feature_id];
    const vector<int>& bins = d.bins[feature_id];
    vector<int> bin_counts(num_bins, 0);
    for (int i = 0; i < num_bins; i++) {
        int count = 0;
        vector<int>& dist = bin_dists[i];

        for (int j = 0; j < d.num_classes; j++) {
            count += dist[j];

        }
	bin_counts[i] = count;
    }

    _t += CycleTimer::currentSeconds();
    // cerr << "find_split inner construct ft_pairs \t taking " << CycleTimer::currentSeconds() - _tt << "s" << endl;

    // _t += CycleTimer::currentSeconds();
    // cerr << "find_split inner sorting ft_pairs \t taking " << CycleTimer::currentSeconds() - _tt << "s" << endl;
    // print(ft_pairs, "find_split_feature ft_pairs");
    // get statistics of how many samples are from each class that says yes to feature < thres

    _t2 -= CycleTimer::currentSeconds();

    // print(num_yes, "find_split_feature num_yes");

    /*
    if (ft_pairs.front().first == ft_pairs.back().first) {
        // all values same, no proper way to split without ending up with an empty left child
        // in this the new entropy will be the same as the parent
        return {feature_id, curr_node->get_entropy(), -1, -1, -1}; // rest is dummy
    }
    */

    vector<int> left_dist(d.num_classes,0);
    vector<int>& class_dist = curr_node->get_class_dist();
    const vector<float>& bin_edges = d.bin_edges[feature_id];

    float min_entropy = numeric_limits<float>::max();
    int total_samples_left = 0;
    float best_left_entropy = -1, best_right_entropy = -1, best_split_thres = -1;
    int best_split_bin = -1;

    for (int split_index = 0; split_index < num_bins - 1; split_index++) {
        while (split_index < num_bins - 1 && bin_counts[split_index] == 0) {
            split_index++;
        }

        if (split_index == num_bins - 1) {
            break;
        }

        add_vector(left_dist, left_dist, bin_dists[split_index]);

        total_samples_left += bin_counts[split_index];
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
            best_split_thres = bin_edges[split_index];
            best_split_bin = split_index;
            best_left_entropy = left_entropy;
            best_right_entropy = right_entropy;
        }
    }

    _t2 += CycleTimer::currentSeconds();
    // cerr << "find_split inner main loops entropy \t taking " << CycleTimer::currentSeconds() - _tt << "s" << endl;
    return {feature_id, min_entropy, best_split_thres, best_split_bin, best_left_entropy, best_right_entropy};
}

SplitInfo DecisionTree::find_split(const Dataset& d, vector<int>& indices, TreeNode* curr_node) {
    if (none_of(indices.begin(), indices.end(), [](bool x) {
        return x;
    })) {
        // indices all false - no data to split
        // should not happen
        cerr << "find_split ending up no data, still deciding what to do with this case";
        abort();
        return {MinSize, -1};
    }
    int best_feature = -1, best_split_bin = -1;
    float min_entropy = numeric_limits<float>::max();
    float best_left_entropy = -1, best_right_entropy = -1, best_split_thres = -1;

    pair<bool, NodeStatus> stop_result = should_stop(curr_node);
    if (stop_result.first) {
        return {stop_result.second, -1, -1, -1, -1, -1};
    }

    _t = 0, _t2 = 0;
    double _tt = CycleTimer::currentSeconds();
    for (int f = 0; f < d.num_features; f++) {
        auto curr_split_info = find_new_entropy_by_split_on_feature(d, indices, f, curr_node);
        if (curr_split_info.min_entropy < min_entropy) {
            min_entropy = curr_split_info.min_entropy;
            best_split_thres = curr_split_info.split_threshold;
            best_split_bin = curr_split_info.split_bin;
            best_left_entropy = curr_split_info.left_entropy;
            best_right_entropy = curr_split_info.right_entropy;
            best_feature = f;
        }
    }
    cerr << "find_split outer main loop \t taking " << CycleTimer::currentSeconds() - _tt << "s" << endl;
    cerr << "find_split outer initializing bin counts \t taking " << _t << "s" << endl;
    cerr << "find_split outer finding split index \t taking " << _t2 << "s" << endl;
    // some extra stuff to check if the SplitInfo meets our requirement
    //
    float info_gain = curr_node->get_entropy() - min_entropy;
    if (info_gain < INFO_GAIN_THRESHOLD) {
        return {NoGain, -1, -1, -1, -1, -1};
    }
    return {best_feature, min_entropy, best_split_thres, best_split_bin, best_left_entropy, best_right_entropy};
}

// Partitions indices according to the node.
//
// Returns the index of the split.
int DecisionTree::split_data(vector<int>& indices, const Dataset& d, TreeNode* curr_node) {

    int feature_id = curr_node->split_info.split_feature_id;
    int split_bin = curr_node->split_info.split_bin;

    double _t = CycleTimer::currentSeconds();
    print(feature_id, "split_data feature_id");

    const vector<int>& bins = d.bins[feature_id];
    auto begin = indices.begin() + curr_node->get_left();
    auto end = indices.begin() + curr_node->get_right();
    auto bound = stable_partition(begin, end,
            [&bins, &split_bin](const int index) {
                return bins[index] <= split_bin;
            });

    cerr << "split_data\t taking " << CycleTimer::currentSeconds() - _t << "s" << endl;

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

    cout << "starting building bins" << endl;
    float bin_start = CycleTimer::currentSeconds();

    d.build_bins(255, root);

    float bin_end = CycleTimer::currentSeconds();
    float bin_time = bin_end - bin_start;
    cout << "building bins took: " << bin_time << " seconds" << endl;

    vector<int>& class_dist = root->get_class_dist();
    class_dist.resize(d.num_classes,0);

    double _ttt = CycleTimer::currentSeconds();

    vector<int> indices(d.num_samples);
    /*
    for (int i = 0; i < d.num_samples; i++) {
      class_dist[d.y[i]]++;
      indices[i] = i;
    }
    */
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

    cout << "class_dist root " << CycleTimer::currentSeconds() - _ttt << endl;

    list<TreeNode*> work_queue;
    work_queue.push_back(root);

    // ptr to device memory
    // used to make sure d.bins and smaller_bin_dist persist in device memory
    int* device_bins = NULL;
    int* device_bin_dist = NULL;
    int* device_indices = NULL;
    int* device_y = NULL;
    int* host_bin_dist = NULL;

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
        print(curr->node_id, "working on splitting node id");
        cout << "working on splitting node id: " << curr->node_id << endl;
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
        print(curr->split_info.split_feature_id, "split on feature:");
        print(curr->split_info.split_threshold, "split on threshold:");

        /*
        cout << "split on feature: " << curr->split_info.split_feature_id << endl;
        cout << "split on threshold: " << curr->split_info.split_threshold << endl;
        cout << "split on bin: " << curr->split_info.split_bin << endl;
        */

        // split data into two halves
        int split_index = split_data(indices, d, curr);
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

        vector<vector<vector<int>>>& left_bin_dist = curr->left_child->setup_bin_dist(d.num_features, d.num_classes, d.num_bins);
        vector<vector<vector<int>>>& right_bin_dist = curr->right_child->setup_bin_dist(d.num_features, d.num_classes, d.num_bins);
        vector<vector<vector<int>>>& curr_bin_dist = curr->get_bin_dist();

        int left_size = split_index - curr->left;
        int right_size = curr->right - split_index;

        vector<vector<vector<int>>>& smaller_bin_dist = right_size > left_size ? left_bin_dist : right_bin_dist;
        vector<vector<vector<int>>>& larger_bin_dist = right_size > left_size ? right_bin_dist : left_bin_dist;
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

	int num_samples_here = end_index - start_index;
	int num_features_gpu = d.num_features / 2;
	// too few samples -> CPU
	// otherwise -> 50/50 hybrid
	if (num_samples_here < HYBRID_CUTOFF) {
	  // num_features_gpu = 0;
	} else {
	  // num_features_gpu = d.num_features / 2;
	}

	update_smaller_bin_dist(d.bins, smaller_bin_dist, indices, labels, d.y, d.num_bins,
				start_index, end_index,
				device_bins, device_bin_dist, device_indices, device_y, host_bin_dist,
				num_features_gpu, d.max_bins, d.num_classes);

        // Calculate right_dist by using left_dist
        subtract_vector(larger_dist, curr_dist, smaller_dist);

        // Similarly for bin dist
#pragma omp parallel for schedule(static)
        for (int f = 0; f < d.num_features; f++) {
            vector<vector<int>>& a = larger_bin_dist[f];
            vector<vector<int>>& b = curr_bin_dist[f];
            vector<vector<int>>& c = smaller_bin_dist[f];

            for (int bin = 0; bin < d.num_bins[f]; bin++) {
                subtract_vector(a[bin], b[bin],
                        c[bin]);
            }
        }

        double dist_end_time = CycleTimer::currentSeconds();

        cout << "calculating children dist took " << dist_end_time - dist_start_time << "s" << endl;

        //cout<< "split index: " << split_index << endl;
        print(curr->left_child->node_id, "new left TreeNode added to queue, node_id:");
        // cout<< "left size: " << curr->left_child->right - curr->left_child->left << endl;
        print(curr->right_child->node_id, "new right TreeNode added to queue, node_id:");
        // cout<< "right size: " << curr->right_child->right - curr->right_child->left << endl;
        work_queue.push_back(curr->left_child);
        work_queue.push_back(curr->right_child);
    }
    for (TreeNode* left_over_node : work_queue) {
        left_over_node->update_majority_label();
    }

    cout << "building bins took: " << bin_time << " seconds" << endl;
}

int DecisionTree::test_single_sample(const Dataset& d, int sample_id) {
    TreeNode* curr = root;
    while (curr) {
        int curr_feature = curr->split_info.split_feature_id;
        float curr_thres = curr->split_info.split_threshold;
        print(curr->node_id, "test_single_sample node_id");
        print(curr_feature, "test_single_sample curr_feature");
        print(curr_thres, "test_single_sample curr_thres");
        if (curr->is_leaf()) {
            return curr->majority_label;
        }
        if (d.x[curr_feature][sample_id] <= curr_thres) {
            if (curr->left_child) {
                curr = curr->left_child;
                print("turn left", "test_single_sample");
            } else {
                return curr->majority_label;
            }
        } else {
            if (curr->right_child) {
                curr = curr->right_child;
                print("turn right", "test_single_sample");
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
        print(predicted_label, "test on samples");
        if (predicted_label == d.y[sample_id]) {
            num_correct++;
        }
    }
    return 1. * num_correct / d.num_samples;
}

