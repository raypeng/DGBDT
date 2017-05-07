//
// Created by Rui Peng on 4/15/17.
//

#ifndef SEQUENTIAL_TREE_H
#define SEQUENTIAL_TREE_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "bindist.h"

using namespace std;

struct SplitInfo {
    int split_feature_id; // going to be NodeStatus if < 0
    float min_entropy;
    float split_threshold;
    int split_bin;
    float left_entropy;
    float right_entropy;
};

enum NodeStatus { Ok = 0, PerfectSplit = -1, MaxDepth = -2, MinSize = -3, NoGain = -4 };

class TreeNode {
    friend class DecisionTree;
private:
    int node_id;
    SplitInfo split_info;
    vector<int> class_dist;
    float entropy;
    TreeNode* left_child;
    TreeNode* right_child;
    int majority_label;
    int depth;
    int size;
    int left;
    int right;
    BinDist bin_dists;

public:
    TreeNode(int node_id_, int depth_, int size_, float entropy_, int left_, int right_) :
            node_id(node_id_),
            depth(depth_),
            size(size_),
            entropy(entropy_),
            left(left_),
            right(right_),
            left_child(NULL),
            right_child(NULL),
            split_info({ -1, -999}),
            majority_label(-1) {
        cout << "node created with id " << node_id << endl;
    }

    vector<int>& get_class_dist() {
        return class_dist;
    }

    BinDist& get_bin_dist() {
        return bin_dists;
    }

    BinDist& setup_bin_dist(int num_features, int num_classes, int num_bins) {
        bin_dists.setup(num_features, num_bins, num_classes);
        return bin_dists;
    }

    void set_class_dist(vector<int> dist) {
        class_dist = dist;
    }

    void update_majority_label() {
        int* data = class_dist.data();
        majority_label = distance(data, max_element(data,
                    data + class_dist.size()));
    }

    void set_entropy(float entropy_) {
        entropy = entropy_;
    }

    float get_entropy() {
        return entropy;
    }

    bool is_leaf() {
        return left_child == NULL && right_child == NULL;
    }

    int get_depth() {
        return depth;
    }

    int get_size() {
        return size;
    }

    int get_left() {
        return left;
    }

    int get_right() {
        return right;
    }

    // for debugging
    void print_subtree() {
        cerr << "DEBUGGING INFO";
    }

};

#endif //SEQUENTIAL_TREE_H
