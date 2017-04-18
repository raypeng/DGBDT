//
// Created by Rui Peng on 4/15/17.
//

#include <list>
#include <cmath>
#include <limits>
#include <algorithm>

#include "decision_tree.h"
#include "mypprint.hpp"

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

float find_info_gain_by_split_on_feature(const Dataset& d, vector<bool> indices, int feature_id, float& thres) {
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
    vector<float> entropies(N, numeric_limits<float>::max());
    int first_nontrivial_index = 1;
    while (first_nontrivial_index < N &&
           ft_pairs[first_nontrivial_index].first == ft_pairs[first_nontrivial_index - 1].first) {
        first_nontrivial_index++;
    }
    print(first_nontrivial_index, "first non_trivial ");
    if (ft_pairs.front().first == ft_pairs.back().first) { // all values same, no proper way to split
        return numeric_limits<float>::max();
    }
    for (int split_index = 0; split_index < N; split_index++) {
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
        entropies[split_index] = 0;
        if (total_samples_left != 0) {
            entropies[split_index] += (1. * total_samples_left / N) * left_entropy;
        }
        if (total_samples_right != 0) {
            entropies[split_index] += (1. * total_samples_right / N) * right_entropy;
        }
    }
    // print(entropies, "find_split_feature entropies");
    int best_split_index = min_element(entropies.begin(), entropies.end()) - entropies.begin();
    thres = ft_pairs[best_split_index].first;
    return entropies[best_split_index];
}

SplitInfo DecisionTree::find_split(const Dataset& d, vector<bool> indices, TreeNode *curr) {
    if (none_of(indices.begin(), indices.end(), [](bool x) {
        return x;
    })) {
        // indices all false - no data to split
        // should not happen
        cerr << "find_split ending up no data, still deciding what to do with this case";
        abort();
        return {MinSize, -1};
    }

    pair<bool, NodeStatus> stop_result = should_stop(d, indices, curr);
    if (stop_result.first) {
        return {stop_result.second, -1};
    }

    vector<float> info_gains(d.num_features), thresholds(d.num_features);
    for (int f = 0; f < d.num_features; f++) {
        float thres;
        info_gains[f] = find_info_gain_by_split_on_feature(d, indices, f, thres);
        thresholds[f] = thres;
    }
    // print(info_gains, "find_split info_gains");
    int best_feature = min_element(info_gains.begin(), info_gains.end()) - info_gains.begin();
    if (info_gains[best_feature] == numeric_limits<float>::max()) {
        // best split is still not proper split, no more splitting possible
        // only happens when sample feature value all same for all features
        return {NoProperSplit, -1};
    }
    return {best_feature, thresholds[best_feature]};
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

pair<bool, NodeStatus> DecisionTree::should_stop(const Dataset& d, vector<bool>& indices, TreeNode* curr) {
    bool perfect_split = true;
    unordered_set<int> possible_labels;
    for (int i = 0; i < d.num_samples; i++) {
        if (indices[i]) {
            possible_labels.insert(d.y[i]);
            if (possible_labels.size() > 1) {
                perfect_split = false;
                break;
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

   // Haven't actualy "splitted" it yet, but we return it here just as a
   // placeholder..
   return {false, Splitted};
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

DecisionTree::DecisionTree(int max_num_leaves_, int max_depth_,
      int min_node_size_) {
    num_leaves = 0;
    max_num_leaves = max_num_leaves_;
    max_depth = max_depth_;
    min_node_size = min_node_size_;
    root = NULL;
}

void DecisionTree::train(const Dataset &d) {
    int curr_node_id = 0;
    vector<bool> all_indices(d.num_samples, true);
    root = new TreeNode(curr_node_id++, all_indices, 0, d.num_samples);
    root->set_majority_label(get_majority_label(d, all_indices));
    list<TreeNode*> work_queue;
    work_queue.push_back(root);
    while (num_leaves < max_num_leaves) {
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

        if (curr->split_info.feature_id < 0) { // no need to split
            num_leaves++;
            switch (curr->split_info.feature_id) {
                case PerfectSplit:
                    cout << "perfect split already for node " << curr->node_id << endl;
                    break;
                case MaxDepth:
                    cout << "node at max depth  " << curr->node_id << endl;
                    break;
                case MinSize:
                    cout << "node at min size  " << curr->node_id << endl;
                    break;
                default:
                    cout << "node became leaf for unkown reason " << curr->node_id << endl;
            }
            continue;
        }

        print(curr->split_info.feature_id, "split on feature:");
        print(curr->split_info.threshold, "split on threshold:");
        // split data into two halves
        // modify left_indices, right_indices in place
        vector<bool> left_indices = curr->sample_indices;
        vector<bool> right_indices = curr->sample_indices;
        auto left_comparator = new FeatureComparatorLT(curr->split_info.threshold);
        auto right_comparator = new FeatureComparatorGE(curr->split_info.threshold);
        split_data(left_indices, d, curr->split_info.feature_id, left_comparator);
        split_data(right_indices, d, curr->split_info.feature_id, right_comparator);
        // optional: remove indices in curr since it's no longer useful
        curr->sample_indices.clear();
        // create two child nodes and add to work queue

        // TODO: make the size of the node actually real after partitioning is implemented
        curr->left_child = new TreeNode(curr_node_id++, left_indices,
                curr->get_depth()+1, d.num_samples);
        curr->right_child = new TreeNode(curr_node_id++, right_indices,
                curr->get_depth()+1, d.num_samples);
        curr->left_child->set_majority_label(get_majority_label(d, left_indices));
        curr->right_child->set_majority_label(get_majority_label(d, right_indices));
        print(curr->left_child->node_id, "new left TreeNode added to queue, node_id:");
        // print(curr->left_child->sample_indices, "with indices:");
        // print(curr->left_child->majority_label, "with majority vote:");
        print(curr->right_child->node_id, "new right TreeNode added to queue, node_id:");
        // print(curr->right_child->sample_indices, "with indices: ");
        // print(curr->right_child->majority_label, "with majority vote:");
        work_queue.push_back(curr->left_child);
        work_queue.push_back(curr->right_child);
    }
}

int DecisionTree::test_single_sample(const Dataset& d, int sample_id) {
    TreeNode* curr = root;
    while (curr) {
        int curr_feature = curr->split_info.feature_id;
        float curr_thres = curr->split_info.threshold;
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
