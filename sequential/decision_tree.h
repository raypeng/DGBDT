//
// Created by Rui Peng on 4/15/17.
//

#ifndef SEQUENTIAL_DECISION_TREE_H
#define SEQUENTIAL_DECISION_TREE_H

#include "tree.h"
#include "dataset.h"

#define MAX_DEPTH 1000000

class DecisionTree {
private:
    int num_leaves;
    int max_num_leaves;
    int max_depth;
    int min_node_size;
    TreeNode* root;

    // Private helper methods.
    SplitInfo find_split(const Dataset& d, vector<bool> indices,
            TreeNode* curr);
    pair<bool, NodeStatus> should_stop(const Dataset& d, vector<bool>&
            indices, TreeNode* curr);
public:
    // potentially more hyperparams
    DecisionTree(int max_num_leaves_, int max_depth_ = MAX_DEPTH,
        int min_node_size_ = 0);

    void train(const Dataset& d);

    float test(const Dataset& d);

    int test_single_sample(const Dataset& d, int sample_id);
};
#endif //SEQUENTIAL_DECISION_TREE_H
