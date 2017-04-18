//
// Created by Rui Peng on 4/15/17.
//

#include <list>
#include <cmath>
#include <limits>
#include <algorithm>

#include "decision_tree.h"
#include "mypprint.hpp"

#define INFO_GAIN_THRESHOLD 1e-3

using namespace std;

class FeatureComparator {
public:
    virtual bool compare(float val) {
        cerr << "ABC method should not be called";
        abort();
    }
};

class FeatureComparatorLT : public FeatureComparator {
public:
    float thres;
    FeatureComparatorLT(float thres_) {
        thres = thres_;
    }
    virtual bool compare(float val) {
        return val < thres;
    }
};

class FeatureComparatorGE : public FeatureComparator {
public:
    float thres;
    FeatureComparatorGE(float thres_) {
        thres = thres_;
    }
    virtual bool compare(float val) {
        return val >= thres;
    }
};

SplitInfo DecisionTree::find_new_entropy_by_split_on_feature(const Dataset& d, vector<bool> indices, int feature_id, TreeNode* curr_node) {
    // entropy before split does not affect comparing info gain values across different features to split
    // so only pick smallest total entropy after split
    // equivalent to taking entropy before split as zero
    print(feature_id, "find_split_feature feature_id");
    vector<pair<float, int>> ft_pairs; // feature_target_pairs
    vector<int> num_yes_before(d.num_classes, 0);
    for (int i = 0; i < d.num_samples; i++) {
        if (indices[i]) {
            ft_pairs.push_back(make_pair(d.x[feature_id][i], d.y[i]));
            num_yes_before[d.y[i]]++;
        }
    }
    // print(num_yes_before, "find_split_feature num_yes_before");
    int N = ft_pairs.size();
    sort(ft_pairs.begin(), ft_pairs.end(), [](const pair<float, int>& a, const pair<float, int>& b) {
        return a.first < b.first;
    });
    // print(ft_pairs, "find_split_feature ft_pairs");
    // get statistics of how many samples are from each class that says yes to feature < thres
    vector<vector<int>> num_yes(d.num_classes, vector<int>(N));
    for (int c = 0; c < d.num_classes; c++) {
        int count = 0;
        for (int i = 0; i < ft_pairs.size(); i++) {
            if (i > 0 && ft_pairs[i].first == ft_pairs[i - 1].first) {
                num_yes[c][i] = num_yes[c][i - 1];
            } else {
                num_yes[c][i] = count;
            }
            count += (ft_pairs[i].second == c);
        }
    }
    // print(num_yes, "find_split_feature num_yes");
    int first_nontrivial_index = 1;
    while (first_nontrivial_index < N &&
           ft_pairs[first_nontrivial_index].first == ft_pairs[first_nontrivial_index - 1].first) {
        first_nontrivial_index++;
    }
    print(first_nontrivial_index, "first non_trivial ");
    if (ft_pairs.front().first == ft_pairs.back().first) {
        // all values same, no proper way to split without ending up with an empty left child
        // in this the new entropy will be the same as the parent
        return {feature_id, curr_node->get_entropy(), -1, -1, -1}; // rest is dummy
    }
    float min_entropy = numeric_limits<float>::max();
    float best_left_entropy = -1, best_right_entropy = -1, best_split_thres = -1;
    for (int split_index = first_nontrivial_index; split_index < N; split_index++) {
        int total_samples_left = 0;
        for (int c = 0; c < d.num_classes; c++) {
            total_samples_left += num_yes[c][split_index];
        }
        int total_samples_right = N - total_samples_left;
        float left_entropy = 0, right_entropy = 0;
        for (int c = 0; c < d.num_classes; c++) {
            int left_samples_per_class = num_yes[c][split_index];
            if (left_samples_per_class != 0) {
                float left_frac_yes = 1. * left_samples_per_class / total_samples_left;
                left_entropy -= left_frac_yes * log2(left_frac_yes);
            }
            int right_samples_per_class = num_yes_before[c] - left_samples_per_class;
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
            best_split_thres = ft_pairs[split_index].first;
            best_left_entropy = left_entropy;
            best_right_entropy = right_entropy;
        }
    }
    return {feature_id, min_entropy, best_split_thres, best_left_entropy, best_right_entropy};
}

SplitInfo DecisionTree::find_split(const Dataset& d, vector<bool> indices, TreeNode* curr_node) {
    if (none_of(indices.begin(), indices.end(), [](bool x) {
        return x;
    })) {
        // indices all false - no data to split
        // should not happen
        cerr << "find_split ending up no data, still deciding what to do with this case";
        abort();
        return {DecisionTree::NoData, -1};
    }
    bool perfect_split = true;
    int last_label = -1;
    for (int i = 0; i < d.num_samples; i++) {
        if (indices[i]) {
            if (last_label == -1) { // not set yet
                last_label = d.y[i];
            } else if (last_label != d.y[i]) { // a different label than last, >= labels
                perfect_split = false;
                break;
            }
        }
    }
    if (perfect_split) { // all samples are from same label
        return {DecisionTree::PerfectSplit, -1, -1, -1, -1};
    }
    int best_feature = -1;
    float min_entropy = numeric_limits<float>::max();
    float best_left_entropy = -1, best_right_entropy = -1, best_split_thres = -1;
    for (int f = 0; f < d.num_features; f++) {
        auto curr_split_info = find_new_entropy_by_split_on_feature(d, indices, f, curr_node);
        if (curr_split_info.min_entropy < min_entropy) {
            min_entropy = curr_split_info.min_entropy;
            best_split_thres = curr_split_info.split_threshold;
            best_left_entropy = curr_split_info.left_entropy;
            best_right_entropy = curr_split_info.right_entropy;
            best_feature = f;
        }
    }
    // some extra stuff to check if the SplitInfo meets our requirement
    float info_gain = curr_node->get_entropy() - min_entropy;
    if (info_gain < INFO_GAIN_THRESHOLD) {
        return {DecisionTree::NoGain, -1, -1, -1, -1};
    }
    return {best_feature, min_entropy, best_split_thres, best_left_entropy, best_right_entropy};
}

void split_data(vector<bool>& indices, const Dataset& d, int feature_id, FeatureComparator* f) {
    print(feature_id, "split_data feature_id");
    const vector<float>& values = d.x[feature_id];
    for (int i = 0; i < d.num_samples; i++) {
        if (indices[i]) {
            indices[i] = f->compare(values[i]);
        }
    }
}

int get_majority_label(const Dataset& d, vector<bool>& indices) {
    vector<int> votes(d.num_classes);
    int num_active = 0;
    for (int i = 0; i < d.num_samples; i++) {
        if (indices[i]) {
            num_active++;
            votes[d.y[i]]++;
        }
    }
    print(num_active, "num_active ");
    return max_element(votes.begin(), votes.end()) - votes.begin();
}

DecisionTree::DecisionTree(int max_num_leaves_) {
    num_leaves = 0;
    max_num_leaves = max_num_leaves_;
    root = NULL;
}

void DecisionTree::train(const Dataset &d) {
    int curr_node_id = 0;
    vector<bool> all_indices(d.num_samples, true);
    float dummy_large_entropy = 1e3; // hack! coz we always split when we start anyways
    root = new TreeNode(curr_node_id++, all_indices, dummy_large_entropy);
    root->set_majority_label(get_majority_label(d, all_indices));
    list<TreeNode*> work_queue;
    work_queue.push_back(root);
    while (num_leaves + work_queue.size() < max_num_leaves) {
        if (work_queue.empty()) {
            print("done", "work queue empty, exit");
            return;
        }
        // remove a node from work queue to perform split on
        TreeNode* curr = work_queue.front();
        work_queue.pop_front();
        // find split according to the data in curr
        vector<bool>& curr_indices = curr->sample_indices;
        print(curr->node_id, "working on splitting node id");
        curr->split_info = find_split(d, curr_indices, curr);
        if (curr->split_info.split_feature_id < 0) { // no need to split
            num_leaves++;
            if (curr->split_info.split_feature_id == DecisionTree::NoProperSplit) {
                cout << "no proper way to split data for node " << curr->node_id << endl;
            } else {
                cout << "perfect split already for node " << curr->node_id << endl;
            }
            curr->set_majority_label(get_majority_label(d, curr->sample_indices));
            continue;
        }
        print(curr->split_info.split_feature_id, "split on feature:");
        print(curr->split_info.split_threshold, "split on threshold:");
        // split data into two halves
        // modify left_indices, right_indices in place
        vector<bool> left_indices = curr->sample_indices;
        vector<bool> right_indices = curr->sample_indices;
        auto left_comparator = new FeatureComparatorLT(curr->split_info.split_threshold);
        auto right_comparator = new FeatureComparatorGE(curr->split_info.split_threshold);
        split_data(left_indices, d, curr->split_info.split_feature_id, left_comparator);
        split_data(right_indices, d, curr->split_info.split_feature_id, right_comparator);
        // optional: remove indices in curr since it's no longer useful
        curr->sample_indices.clear();
        // create two child nodes and add to work queue
        curr->left_child = new TreeNode(curr_node_id++, left_indices, curr->split_info.left_entropy);
        curr->right_child = new TreeNode(curr_node_id++, right_indices, curr->split_info.right_entropy);
        print(curr->left_child->node_id, "new left TreeNode added to queue, node_id:");
        // print(curr->left_child->sample_indices, "with indices:");
        // print(curr->left_child->majority_label, "with majority vote:");
        print(curr->right_child->node_id, "new right TreeNode added to queue, node_id:");
        // print(curr->right_child->sample_indices, "with indices: ");
        // print(curr->right_child->majority_label, "with majority vote:");
        work_queue.push_back(curr->left_child);
        work_queue.push_back(curr->right_child);
    }
    for (TreeNode* left_over_node : work_queue) {
        left_over_node->set_majority_label(get_majority_label(d, left_over_node->sample_indices));
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
        if (d.x[curr_feature][sample_id] < curr_thres) {
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
