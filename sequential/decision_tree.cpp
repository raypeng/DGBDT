//
// Created by Rui Peng on 4/15/17.
//

#include <list>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iterator>

#include "decision_tree.h"
#include "mypprint.hpp"
#include "util.h"
#include "CycleTimer.h"

#define INFO_GAIN_THRESHOLD 1e-3

using namespace std;

SplitInfo DecisionTree::find_new_entropy_by_split_on_feature(const Dataset& d, vector<int>& indices, int feature_id, TreeNode* curr_node) {
    // entropy before split does not affect comparing info gain values across different features to split
    // so only pick smallest total entropy after split
    // equivalent to taking entropy before split as zero
    print(feature_id, "find_split_feature feature_id");
    _t -= CycleTimer::currentSeconds();

    int N = curr_node->right - curr_node->left;
    int num_bins = d.num_bins[feature_id];
    const vector<int>& bins = d.bins[feature_id];
    vector<int> bin_counts(num_bins, 0);
    for (int i = 0; i < N; i++) {
        int index = indices[i + curr_node->left];
        bin_counts[bins[index]]++;
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
    const vector<vector<int>>& bin_dists = d.bin_dists[feature_id];
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

void DecisionTree::train(const Dataset &d) {

    // Setup root node.

    int curr_node_id = 0;

    float dummy_large_entropy = 1e3; // hack! coz we always split when we start anyways
    root = new TreeNode(curr_node_id++, 0, d.num_samples, dummy_large_entropy, 0, d.num_samples);
    vector<int>& class_dist = root->get_class_dist();
    class_dist.resize(d.num_classes,0);

    vector<int> indices(d.num_samples);
    for (int i = 0; i < d.num_samples; i++) {
        indices[i] = i;
    }

    // compute class_dist from bin distributions of first feature
    for (int i = 0; i < d.num_bins[0]; i++) {
        add_vector(class_dist, class_dist, d.bin_dists[0][i]);
    }

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
        print(curr->node_id, "working on splitting node id");
        _t = CycleTimer::currentSeconds();
        curr->split_info = find_split(d, indices, curr);
        cerr << "find_split for node " << curr->node_id << "\t taking " << CycleTimer::currentSeconds() - _t << "s" << endl;

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
        // split data into two halves
        int split_index = split_data(indices, d, curr);

        // create two child nodes and add to work queue
        // TODO: make the size of the node actually real after partitioning is implemented
        curr->left_child = new TreeNode(curr_node_id++, curr->get_depth()+1, d.num_samples, curr->split_info.left_entropy, curr->left, split_index);
        curr->right_child = new TreeNode(curr_node_id++, curr->get_depth()+1, d.num_samples, curr->split_info.right_entropy, split_index, curr->right);

        // Update class distributions
        vector<int>& left_dist = curr->left_child->get_class_dist();
        vector<int>& right_dist = curr->right_child->get_class_dist();
        vector<int>& curr_dist = curr->get_class_dist();
        left_dist.resize(d.num_classes, 0);
        right_dist.resize(d.num_classes, 0);

        for (int i = curr->left; i < split_index; i++) {
            int index = indices[i];
            left_dist[d.y[index]]++;
        }

        // Calculate right_dist by using left_dist
        subtract_vector(right_dist, curr_dist, left_dist);

        //cout<< "split index: " << split_index << endl;
        print(curr->left_child->node_id, "new left TreeNode added to queue, node_id:");
        //cout<< "left size: " << curr->left_child->right - curr->left_child->left << endl;
        print(curr->right_child->node_id, "new right TreeNode added to queue, node_id:");
        //cout<< "right size: " << curr->right_child->right - curr->right_child->left << endl;
        work_queue.push_back(curr->left_child);
        work_queue.push_back(curr->right_child);
    }
    for (TreeNode* left_over_node : work_queue) {
        left_over_node->update_majority_label();
    }
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
