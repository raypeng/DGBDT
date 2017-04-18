//
// Created by Rui Peng on 4/15/17.
//

#ifndef SEQUENTIAL_TREE_H
#define SEQUENTIAL_TREE_H

#include <iostream>
#include <vector>

using namespace std;

struct SplitInfo {
    int feature_id; // going to be NodeStatus if < 0
    float threshold;
};

enum NodeStatus { Splitted = 0, PerfectSplit = -1, MaxDepth = -2, MinSize = -3};

class TreeNode {
    friend class DecisionTree;
private:
    int node_id;
    SplitInfo split_info;
    vector<bool> sample_indices;
    TreeNode* left_child;
    TreeNode* right_child;
    int majority_label;
    int depth;
    int size;

public:
    TreeNode(int node_id_, vector<bool>& sample_indices_, int depth_, int size_) :
            node_id(node_id_),
            sample_indices(sample_indices_),
            left_child(NULL),
            right_child(NULL),
            split_info({ -1, -999}),
            majority_label(-1),
            depth(depth_),
            size(size_) {
        cout << "node created with id " << node_id << endl;
    }

    void set_majority_label(int label) {
        majority_label = label;
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

    // for debugging
    void print_subtree() {
        cerr << "DEBUGGING INFO";
    }

};

#endif //SEQUENTIAL_TREE_H
