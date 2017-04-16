//
// Created by Rui Peng on 4/15/17.
//

#ifndef SEQUENTIAL_TREE_H
#define SEQUENTIAL_TREE_H

#include <iostream>
#include <vector>

using namespace std;

struct SplitInfo {
    int feature_id;
    float threshold;
};

class TreeNode {
    friend class DecisionTree;
private:
    int node_id;
    SplitInfo split_info;
    vector<bool> sample_indices;
    TreeNode* left_child;
    TreeNode* right_child;
    int majority_label;

public:
    TreeNode(int node_id_, vector<bool>& sample_indices_) :
            node_id(node_id_),
            sample_indices(sample_indices_),
            left_child(NULL),
            right_child(NULL),
            split_info({ -1, -999}),
            majority_label(-1) {
        cout << "node created with id " << node_id << endl;
    }

    void set_majority_label(int label) {
        majority_label = label;
    }

    bool is_leaf() {
        return left_child == NULL && right_child == NULL;
    }

    // for debugging
    void print_subtree() {
        cerr << "DEBUGGING INFO";
    }

};

#endif //SEQUENTIAL_TREE_H
