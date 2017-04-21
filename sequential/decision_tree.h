//
// Created by Rui Peng on 4/15/17.
//

#ifndef SEQUENTIAL_DECISION_TREE_H
#define SEQUENTIAL_DECISION_TREE_H

#include "tree.h"
#include "dataset.h"

#define MAX_DEPTH 100000

class DecisionTree {
private:

    double _t, _t2;
    int num_leaves;
    int max_num_leaves;
    int max_depth;
    int min_node_size;
    TreeNode* root;

    SplitInfo find_new_entropy_by_split_on_feature(const Dataset& d, vector<int>& indices, int feature_id, TreeNode* curr_node);
    SplitInfo find_split(const Dataset& d, vector<int>& indices, TreeNode* curr_node);
    pair<bool, NodeStatus> should_stop(TreeNode* curr);
    int split_data(vector<int>& indices, const Dataset& d, TreeNode* curr_node);

public:
    // potentially more hyperparams
    DecisionTree(int max_num_leaves_, int max_depth_ = MAX_DEPTH, int min_node_size_ = 0);

    void train(const Dataset& d);

    float test(const Dataset& d);

    int test_single_sample(const Dataset& d, int sample_id);
};
#endif //SEQUENTIAL_DECISION_TREE_H
