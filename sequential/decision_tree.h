//
// Created by Rui Peng on 4/15/17.
//

#ifndef SEQUENTIAL_DECISION_TREE_H
#define SEQUENTIAL_DECISION_TREE_H

#include "tree.h"
#include "dataset.h"

class DecisionTree {
private:

    int num_leaves;
    int max_num_leaves;
    TreeNode* root;

    SplitInfo find_new_entropy_by_split_on_feature(const Dataset& d, vector<bool> indices, int feature_id, TreeNode* curr_node);
    SplitInfo find_split(const Dataset& d, vector<bool> indices, TreeNode* curr_node);

public:
    enum NodeStatus { NoGain = -5, NoProperSplit = -4, NoData = -3, PerfectSplit = -2 };

    // potentially more hyperparams
    DecisionTree(int max_num_leaves_);

    void train(const Dataset& d);

    float test(const Dataset& d);

    int test_single_sample(const Dataset& d, int sample_id);
};
#endif //SEQUENTIAL_DECISION_TREE_H
