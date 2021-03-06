//
// Created by Rui Peng on 4/15/17.
//

#include <list>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iterator>
#include <map>
#include <mpi.h>

#include "decision_tree.h"
#include "util.h"
#include "mypprint.hpp"
#include "CycleTimer.h"
#include "mpi_util.h"

#define INFO_GAIN_THRESHOLD 1e-3

#define NUM_VOTES 5
#define INVALID_FEATURE -1

using namespace std;

static double update_time = 0;
static double hist_time = 0;
static double stop_time = 0;
static double bcast_time = 0;
static double outer_loop_time = 0;
static double accum_count_time = 0;
static double scan_dbin_time = 0;
static double voting_comm_time = 0;

void DecisionTree::collect_top_k_features(const Dataset& d, TreeNode* curr_node,
    Heap& h) {

    BinDist& bin_dist = curr_node->get_bin_dist();
    for (int f = 0; f < d.num_features; f++) {
        int N = curr_node->right - curr_node->left;
        int num_bins = d.num_bins[f];
        const vector<int>& bins = d.bins[f];
        vector<int> bin_counts(num_bins, 0);

        for (int i = 0; i < num_bins; i++) {
            int count = 0;
            for (int j = 0; j < d.num_classes; j++) {
                count += bin_dist.get(f,i,j);

            }
            bin_counts[i] = count;
        }
        vector<int> left_dist(d.num_classes,0);
        vector<int>& class_dist = curr_node->get_class_dist();
        const vector<float>& bin_edges = d.bin_ends[f];

        int total_samples_left = 0;
        float min_entropy = numeric_limits<float>::max();

        for (int split_index = 0; split_index < num_bins - 1; split_index++) {
            while (split_index < num_bins - 1 && bin_counts[split_index] == 0) {
                split_index++;
            }

            if (split_index == num_bins - 1) {
                break;
            }

            add_vector(left_dist, left_dist, bin_dist.head(f,split_index));

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
            }
        }

        if (min_entropy < h.max()) {
            h.insert(f, min_entropy);
        }
    }
}

SplitInfo DecisionTree::find_new_entropy_by_split_on_feature(Dataset& d, int feature_id, TreeNode* curr_node) {
    // entropy before split does not affect comparing info gain values across different features to split
    // so only pick smallest total entropy after split
    // equivalent to taking entropy before split as zero
    _t -= CycleTimer::currentSeconds();

    BinDist& bin_dist = curr_node->get_bin_dist();

    vector<BinDist*> distributed_bin_dist = d.distributed_bin_dist;

    // Accumulate total sample counts, individual bin sample counts,
    // and class distribution across all nodes.
    double accum_start = CycleTimer::currentSeconds();
    int N = 0;
    vector<vector<int>> bin_counts(mpi_world_size(), vector<int>(d.max_bins));
    vector<int> class_dist(d.num_classes);

    for (int r = 0; r < mpi_world_size(); r++) {
        int num_bins = d.distributed_num_bins[r][feature_id];

        for (int i = 0; i < num_bins; i++) {
            int count = 0;

            for (int j = 0; j < d.num_classes; j++) {
                int val = distributed_bin_dist[r]->get(feature_id,i,j);
                count += val;
                class_dist[j] += val;
            }

            bin_counts[r][i] = count;
            N += count;
        }
    }
    double accum_end = CycleTimer::currentSeconds();
    accum_count_time += (accum_end - accum_start);

    _t += CycleTimer::currentSeconds();
    _t2 -= CycleTimer::currentSeconds();

    vector<int> left_dist(d.num_classes,0);
    const vector<float>& bin_ends = d.bin_ends[feature_id];

    float min_entropy = numeric_limits<float>::max();
    int total_samples_left = 0;
    float best_left_entropy = -1, best_right_entropy = -1, best_split_thres = -1;

    double dbin_start = CycleTimer::currentSeconds();
    vector<DistributedBin>& dbins = d.distributed_bins[feature_id];

    // Map (rank,bin) to the bin to store active bins while scanning.
    //
    // Since there are only O(num_bins * num_nodes) many bins, we use map here
    // to avoid the overhead of an unordered_map on repeated insertions,
    // deletes, and iterations across the map. unordered_map may or may not be
    // faster, but it's probably about the same.
    //map<pair<int,int>, DistributedBin> active_bins;

    for (int i = 0; i < dbins.size(); i++) {

        DistributedBin dbin = dbins[i];

        // Find first non-empty distributed bin.
        while (i < dbins.size() && bin_counts[dbin.rank][dbin.bin] == 0) {
            i++;
            dbin = dbins[i];
        }

        // No point splitting on the end.
        if (i == dbins.size()) {
            break;
        }

        BinDist* bin_dist = distributed_bin_dist[dbin.rank];

        vector<int> accumulate_dist(d.num_classes,0);

        // on bin ends, first find the first non-repeating bin end, then perform the
        // entropy calculation of a potential threshold split point.
        while (i < dbins.size() - 1
                && float_equal(dbin.v, dbins[i+1].v)) {
            add_vector(accumulate_dist, accumulate_dist, bin_dist->head(feature_id, dbin.bin));
            total_samples_left += bin_counts[dbin.rank][dbin.bin];

            //active_bins.erase({dbin.rank,dbin.bin});
            i++;
            dbin = dbins[i];
            bin_dist = distributed_bin_dist[dbin.rank];
        }

        add_vector(left_dist, left_dist, accumulate_dist);
        add_vector(left_dist, left_dist, bin_dist->head(feature_id,dbin.bin));

        // TODO: interpolation of active bins

        //active_bins.erase({dbin.rank,dbin.bin});
        total_samples_left += bin_counts[dbin.rank][dbin.bin];
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
            best_split_thres = dbin.v;
            best_left_entropy = left_entropy;
            best_right_entropy = right_entropy;
        }

    }


    double dbin_end = CycleTimer::currentSeconds();
    scan_dbin_time += (dbin_end - dbin_start);

    _t2 += CycleTimer::currentSeconds();
    // cerr << "find_split inner main loops entropy \t taking " << CycleTimer::currentSeconds() - _tt << "s" << endl;
    return {feature_id, min_entropy, best_split_thres, best_left_entropy, best_right_entropy};
}

static SplitInfo bcast_split_info(SplitInfo info) {
    double bcast_start = CycleTimer::currentSeconds();
	MPI_Bcast(&info, 1, split_info_type(), 0, MPI_COMM_WORLD);
    double bcast_end = CycleTimer::currentSeconds();
    bcast_time += (bcast_end - bcast_start);
	return info;
}

SplitInfo DecisionTree::find_split(Dataset& d, vector<int>& indices, TreeNode* curr_node) {
    if (none_of(indices.begin(), indices.end(), [](bool x) {
        return x;
    })) {
        // indices all false - no data to split
        // should not happen
        cerr << "find_split ending up no data, still deciding what to do with this case";
        abort();
        return {MinSize, -1};
    }

	// Stop if all nodes say stop.
    double stop_start = CycleTimer::currentSeconds();
	pair<bool, NodeStatus> stop_result = should_stop(curr_node);
	bool to_stop;
	MPI_Allreduce(&stop_result.first, &to_stop, 1, MPI_BYTE, MPI_LAND, MPI_COMM_WORLD);
	if (to_stop) {
		mpi_print("detected should stop");
		return {stop_result.second, -1, -1, -1, -1};
	}
    double stop_end = CycleTimer::currentSeconds();
    stop_time += (stop_end - stop_start);


    Heap h(NUM_VOTES);
    collect_top_k_features(d, curr_node, h);

    Heap feature_heap(2 * NUM_VOTES);
    vector<int> best_features;

    // Perform voting
    if (is_root()) {
        vector<int> all_votes(NUM_VOTES * mpi_world_size());
        double gather_start = CycleTimer::currentSeconds();
		MPI_Gather(h.data(), NUM_VOTES, MPI_INT, all_votes.data(),
				NUM_VOTES, MPI_INT, 0, MPI_COMM_WORLD);
        voting_comm_time += (CycleTimer::currentSeconds() - gather_start);

        // Top 2k will be requested

        vector<int> vote_counts(d.num_features);
        // Kinda hacky, since we have a max heap, but we want to keep
        // the max in the heap, we use negative votes
        for (int i = 0; i < all_votes.size(); i++) {
            vote_counts[all_votes[i]]--;
        }
        for (int f = 0; f < vote_counts.size(); f++) {
            int count = vote_counts[f];
            if (count < 0 && count < feature_heap.max()) {
                feature_heap.insert(f,count);
            }
        }

        best_features = feature_heap.get_ids();
        best_features.resize(feature_heap.get_num());

        double bcast_start = CycleTimer::currentSeconds();
		MPI_Bcast(best_features.data(), best_features.size(), MPI_INT, 0, MPI_COMM_WORLD);
        voting_comm_time += (CycleTimer::currentSeconds() - bcast_start);

    } else {
		MPI_Gather(h.data(), NUM_VOTES, MPI_INT, NULL,
				NUM_VOTES, MPI_INT, 0, MPI_COMM_WORLD);

        best_features.resize(NUM_VOTES * 2, INVALID_FEATURE);
		MPI_Bcast(best_features.data(), NUM_VOTES * 2, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (is_root()) {
        int best_feature = -1;
        float min_entropy = numeric_limits<float>::max();
        float best_left_entropy = -1, best_right_entropy = -1, best_split_thres = -1;

        _t = 0, _t2 = 0;
        double _tt = CycleTimer::currentSeconds();

		// avoid copying our own bin distribution
		BinDist& bin_dist = curr_node->get_bin_dist();
		d.distributed_bin_dist[0] = &bin_dist;


        for (int i = 0; i < best_features.size(); i++) {
            int f = best_features[i];

            double hist_start = CycleTimer::currentSeconds();
            for (int r = 1 ; r < d.distributed_bin_dist.size(); r++) {
                // Use feature_id as tag to differentiate between different features
                // if we parallelize this across different features.
                MPI_Recv(d.distributed_bin_dist[r]->head(f), d.distributed_num_bins[r][f] * d.num_classes, MPI_INT, r,
                        0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //mpi_print("receiving bin dists taking: ", CycleTimer::currentSeconds() - _tt);
            hist_time += CycleTimer::currentSeconds() - hist_start;

            auto curr_split_info = find_new_entropy_by_split_on_feature(d, f, curr_node);
            if (curr_split_info.min_entropy < min_entropy) {
                min_entropy = curr_split_info.min_entropy;
                best_split_thres = curr_split_info.split_threshold;
                best_left_entropy = curr_split_info.left_entropy;
                best_right_entropy = curr_split_info.right_entropy;
                best_feature = f;
            }
        }

        outer_loop_time += (CycleTimer::currentSeconds() - _tt);
        //mpi_print("find_split outer main loop \t taking ", CycleTimer::currentSeconds() - _tt);
        /*
        cerr << "find_split outer initializing bin counts \t taking " << _t << "s" << endl;
        cerr << "find_split outer finding split index \t taking " << _t2 << "s" << endl;
        */
        // some extra stuff to check if the SplitInfo meets our requirement
        //
        float info_gain = curr_node->get_entropy() - min_entropy;
        if (info_gain < INFO_GAIN_THRESHOLD) {
            return bcast_split_info({NoGain, -1, -1, -1, -1});
        }
        return bcast_split_info({best_feature, min_entropy, best_split_thres, best_left_entropy, best_right_entropy});
    } else {
		BinDist& bin_dist = curr_node->get_bin_dist();
        for (int i = 0; i < best_features.size(); i++) {
            int f = best_features[i];
            if (f != INVALID_FEATURE) {
                MPI_Send(bin_dist.head(f), d.num_bins[f] * d.num_classes, MPI_INT, 0,
                        0, MPI_COMM_WORLD);
            }
        }
		SplitInfo info;
		MPI_Bcast(&info, 1, split_info_type(), 0, MPI_COMM_WORLD);
		return info;
    }
}

// Partitions indices according to the node.
//
// Returns the index of the split.
int DecisionTree::split_data(vector<int>& indices, const Dataset& d, TreeNode* curr_node) {

    int feature_id = curr_node->split_info.split_feature_id;
    float split_point = curr_node->split_info.split_threshold;

    double _t = CycleTimer::currentSeconds();

    const vector<int>& bins = d.bins[feature_id];
    const vector<float>& feature_row = d.x[feature_id];
    auto begin = indices.begin() + curr_node->get_left();
    auto end = indices.begin() + curr_node->get_right();
    auto bound = stable_partition(begin, end,
            [&feature_row, &split_point](const int index) {
                return feature_row[index] <= split_point;
            });

    //cerr << "split_data\t taking " << CycleTimer::currentSeconds() - _t << "s" << endl;

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

void DecisionTree::train(Dataset &d) {

    // Setup root node.

    int curr_node_id = 0;

    float dummy_large_entropy = 1e3; // hack! coz we always split when we start anyways
    root = new TreeNode(curr_node_id++, 0, d.num_samples, dummy_large_entropy, 0, d.num_samples);

    mpi_print("starting building bins");
    double bin_start = CycleTimer::currentSeconds();

    d.build_bins(255, root);

    double bin_end = CycleTimer::currentSeconds();
    double bin_time = bin_end - bin_start;
    mpi_print("building bins took: ", bin_time);

    vector<int>& class_dist = root->get_class_dist();
    class_dist.resize(d.num_classes,0);

    double _ttt = CycleTimer::currentSeconds();

    vector<int> indices(d.num_samples);

#pragma omp parallel
    {
      vector<int> local_class_dist(d.num_classes);
#pragma omp for schedule(static) nowait
      for (int i = 0; i < d.num_samples; i++) {
        local_class_dist[d.y[i]]++;
        indices[i] = i;
      }
      for (int label = 0; label < d.num_classes; label++) {
#pragma omp atomic
	class_dist[label] += local_class_dist[label];
      }
    }

    mpi_print("class_dist root ", CycleTimer::currentSeconds() - _ttt);

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
        mpi_print("working on splitting node id: ", curr->node_id);
        _t = CycleTimer::currentSeconds();
        curr->split_info = find_split(d, indices, curr);
        /*
        cerr << "find_split for node " << curr->node_id << "\t taking " << CycleTimer::currentSeconds() - _t << "s" << endl;
        */

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
            double update_start = CycleTimer::currentSeconds();
            curr->update_majority_label();
            double update_end = CycleTimer::currentSeconds();
            update_time += (update_end - update_start);
            continue;
        }
        mpi_print("split on feature: ", curr->split_info.split_feature_id);
        mpi_print("split on threshold: ", curr->split_info.split_threshold);

        /*
        cout << "split on feature: " << curr->split_info.split_feature_id << endl;
        cout << "split on threshold: " << curr->split_info.split_threshold << endl;
        */

        mpi_print("splitting data");
        // split data into two halves
        int split_index = split_data(indices, d, curr);
        mpi_print("making children");
        // create two child nodes and add to work queue
        // TODO: make the size of the node actually real after partitioning is implemented
        curr->left_child = new TreeNode(curr_node_id++, curr->get_depth()+1, d.num_samples, curr->split_info.left_entropy, curr->left, split_index);
        curr->right_child = new TreeNode(curr_node_id++, curr->get_depth()+1, d.num_samples, curr->split_info.right_entropy, split_index, curr->right);

        double dist_start_time = CycleTimer::currentSeconds();

        // Update class and bin distributions
        vector<int>& left_dist = curr->left_child->get_class_dist();
        vector<int>& right_dist = curr->right_child->get_class_dist();
        vector<int>& curr_dist = curr->get_class_dist();
        left_dist.resize(d.num_classes, 0);
        right_dist.resize(d.num_classes, 0);

        BinDist& left_bin_dist = curr->left_child->setup_bin_dist(d.num_features, d.max_bins, d.num_classes);
        BinDist& right_bin_dist = curr->right_child->setup_bin_dist(d.num_features, d.max_bins, d.num_classes);
        BinDist& curr_bin_dist = curr->get_bin_dist();

        int left_size = split_index - curr->left;
        int right_size = curr->right - split_index;

        BinDist& smaller_bin_dist = right_size > left_size ? left_bin_dist : right_bin_dist;
        BinDist& larger_bin_dist = right_size > left_size ? right_bin_dist : left_bin_dist;
        vector<int>& smaller_dist = right_size > left_size ? left_dist : right_dist;
        vector<int>& larger_dist = right_size > left_size ? right_dist : left_dist;
        int start_index;
        int end_index;

        if (right_size > left_size) {
            start_index = curr->left;
            end_index = split_index;
        } else {
            start_index = split_index;
            end_index = curr->right;
        }

        vector<int> labels(end_index - start_index);
        for (int i = start_index; i < end_index; i++) {
            int index = indices[i];
            int label = d.y[index];
            labels[i - start_index] = label;
            smaller_dist[label]++;
        }

	/*
#pragma omp parallel
	    {
	      vector<int> local_smaller_dist(d.num_classes);
#pragma omp for schedule(static) nowait
	      for (int i = start_index; i < end_index; i++) {
		int index = indices[i];
		int label = d.y[index];
		labels[i - start_index] = label;
		local_smaller_dist[label]++;
	      }
	      for (int label = 0; label < d.num_classes; label++) {
#pragma omp atomic
		smaller_dist[label] += local_smaller_dist[label];
	      }
	    }
	*/

#pragma omp parallel for schedule(static)
        for (int f = 0; f < d.num_features; f++) {

            vector<int>& bins = d.bins[f];

            for (int i = start_index; i < end_index; i++) {
                int index = indices[i];
                int label = labels[i - start_index];

                int bin = bins[index];
                smaller_bin_dist.inc(f,bin,label);
            }
        }


        // Calculate right_dist by using left_dist
        subtract_vector(larger_dist, curr_dist, smaller_dist);

        // Similarly for bin dist
        larger_bin_dist.diff(curr_bin_dist, smaller_bin_dist);

        double dist_end_time = CycleTimer::currentSeconds();

        mpi_print("calculating children dist took ", dist_end_time - dist_start_time);

		mpi_print("left size: ", left_size);
		mpi_print("right size: ", right_size);
        //cout<< "split index: " << split_index << endl;
        // cout<< "left size: " << curr->left_child->right - curr->left_child->left << endl;
        // cout<< "right size: " << curr->right_child->right - curr->right_child->left << endl;
        work_queue.push_back(curr->left_child);
        work_queue.push_back(curr->right_child);
    }
    double update_start = CycleTimer::currentSeconds();
    for (TreeNode* left_over_node : work_queue) {
        left_over_node->update_majority_label();
    }
    double update_end = CycleTimer::currentSeconds();
    update_time += (update_end - update_start);

    if (is_root()) {
        mpi_print("building bins took: ", bin_time);
        mpi_print("communicating histogram s", hist_time);
        mpi_print("communicating updates ", update_time);
        mpi_print("communicating stops ", stop_time);
        mpi_print("communicating bcast ", bcast_time);
        mpi_print("communicating votes ", voting_comm_time);
        mpi_print("dbin scan ", scan_dbin_time);
        mpi_print("accum counts ", accum_count_time);
        mpi_print("outer loop total ", outer_loop_time);
        mpi_print("total comm time ", hist_time + update_time + stop_time
                + bcast_time + voting_comm_time);
    }
}

int DecisionTree::test_single_sample(const Dataset& d, int sample_id) {
    TreeNode* curr = root;
    while (curr) {
        int curr_feature = curr->split_info.split_feature_id;
        float curr_thres = curr->split_info.split_threshold;
        if (curr->is_leaf()) {
            return curr->majority_label;
        }
        if (d.x[curr_feature][sample_id] <= curr_thres) {
            if (curr->left_child) {
                curr = curr->left_child;
            } else {
                return curr->majority_label;
            }
        } else {
            if (curr->right_child) {
                curr = curr->right_child;
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
        if (predicted_label == d.y[sample_id]) {
            num_correct++;
        }
    }
    return 1. * num_correct / d.num_samples;
}
