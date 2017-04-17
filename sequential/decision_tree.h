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
public:
    enum NodeStatus { NoProperSplit = -4, NoData = -3, PerfectSplit = -2 };
    // potentially more hyperparams
    DecisionTree(int max_num_leaves_);

    void train(const Dataset& d);

    float test(const Dataset& d);

    int test_single_sample(const Dataset& d, int sample_id);
};
#endif //SEQUENTIAL_DECISION_TREE_H
