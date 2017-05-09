//
// Created by Rui Peng on 4/15/17.
//

#include "decision_tree_util.h"

#define UPDIV(n, d) (((n)+(d)-1)/(d))


__global__ void
kernel_update_bin_dist(int* d_bins, int* d_bin_dist, int* d_indices, int* d_labels,
		       int start_index, int end_index,
		       int num_features, int num_bins, int num_classes) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  int num_threads = blockDim.x;
  int f = block_id;
  for (int i = start_index + thread_id; i < end_index; i += num_threads) {
    int index = d_indices[i - start_index];
    int label = d_labels[i - start_index];
    int bin = d_bins[index];
    atomicAdd(&d_bin_dist[f * num_bins * num_classes + bin * num_classes + label], 1);
  }
}

void update_smaller_bin_dist(vector<vector<int>>& bins,
			     vector<vector<vector<int>>>& smaller_bin_dist,
			     vector<int>& indices,
			     vector<int>& labels,
			     int start_index,
			     int end_index,
			     int*& d_bins,
			     int*& d_bin_dist) {
  int num_features = bins.size();
  int num_bins = 255;
  int num_classes = 5;
  int num_samples = bins.front().size();

  if (d_bins == NULL) {
    cudaMalloc(&d_bins, sizeof(int) * num_features * num_samples);
    for (int f = 0; f < num_features; f++) {
      // cout << "bin size " << f << " " << bins[f].size() << endl;
      cudaMemcpy(d_bins + f * num_samples, bins[f].data(), sizeof(int) * num_samples, cudaMemcpyHostToDevice);
    }
    cout << "cudaMalloc(d_bins)" << endl;
  }
  if (d_bin_dist == NULL) {
    cudaMalloc(&d_bin_dist, sizeof(int) * num_features * num_bins * num_classes);
  }
  cudaMemset(d_bin_dist, 0, sizeof(int) * num_features * num_bins * num_classes);

  int* d_indices;
  cudaMalloc(&d_indices, sizeof(int) * (end_index - start_index));
  cudaMemcpy(d_indices, indices.data() + start_index, sizeof(int) * (end_index - start_index), cudaMemcpyHostToDevice);
  int* d_labels;
  cudaMalloc(&d_labels, sizeof(int) * (end_index - start_index));
  cudaMemcpy(d_labels, labels.data(), sizeof(int) * (end_index - start_index), cudaMemcpyHostToDevice);

  cout << "entering kernel" << endl;
  double _t = CycleTimer::currentSeconds();
  int num_threads_per_block = 32;
  int num_blocks = UPDIV(num_features, num_threads_per_block);
  kernel_update_bin_dist<<<num_blocks, num_threads_per_block>>>
    (d_bins, d_bin_dist, d_indices, d_labels,
     start_index, end_index,
     num_features, num_bins, num_classes);
  cudaThreadSynchronize();
  cout << "exiting kernel: " << CycleTimer::currentSeconds() - _t << endl;

  int* h_bin_dist = new int[num_features * num_bins * num_classes];
  cudaMemcpy(h_bin_dist, d_bin_dist, sizeof(int) * num_features * num_bins * num_classes, cudaMemcpyDeviceToHost);
  cout << "h_bin_dist done" << endl;
  for (int f = 0; f < num_features; f++) {
    for (int bin = 0; bin < num_bins; bin++) {
      for (int label = 0; label < num_classes; label++) {
	int i = f * num_bins * num_classes + bin * num_classes + label;
	if (h_bin_dist[i] > 0) {
	  if (bin >= smaller_bin_dist[f].size() || label >= smaller_bin_dist[f][bin].size()) {
	    cout << i << " " << f << " " << bin << " " << label << " " << h_bin_dist[i] << endl;
	    cout << smaller_bin_dist[f].size() << " " << smaller_bin_dist[f][bin].size() << endl;
	    continue;
	  }
	  smaller_bin_dist[f][bin][label] = h_bin_dist[i];
	}
      }
    }
  }
  /*
  for (int i = 0; i < num_features * num_bins * num_classes; i++) {
    if (h_bin_dist[i] > 0) {
      int f = i / (num_bins * num_classes);
      int bin = (i / num_classes) % num_bins;
      int label = i % num_classes;
      if (bin >= smaller_bin_dist[f].size() || label >= smaller_bin_dist[f][bin].size()) {
	cout << i << " " << f << " " << bin << " " << label << " " << h_bin_dist[i] << endl;
	cout << smaller_bin_dist[f].size() << " " << smaller_bin_dist[f][bin].size() << endl;
	continue;
      }
      smaller_bin_dist[f][bin][label] = h_bin_dist[i];
    }
  }
  */
  cout << "smaller_bin_dist done" << endl;

  cudaFree(d_indices);
  cudaFree(d_labels);
  delete[] h_bin_dist;
}


/*
void update_smaller_bin_dist(vector<vector<int>>& bins,
			     vector<vector<vector<int>>>& smaller_bin_dist,
			     vector<int>& indices,
			     vector<int>& labels,
			     int start_index,
			     int end_index) {

  using namespace thrust;

  device_vector<int> d_indices(indices.begin(), indices.end());
  device_vector<int> d_labels(labels.begin(), labels.end());
  int num_features = smaller_bin_dist.size();

  device_vector<int> label_indices(end_index - start_index);
  thrust::gather(thrust::device,
		 d_indices.begin() + start_index, d_indices.begin() + end_index,
		 d_labels.begin(),
		 label_indices.begin());

  device_vector<int> bin_indices(end_index - start_index);
  for (int f = 0; f < num_features; f++) {
    device_vector<int> d_bins(bins[f].begin(), bins[f].end());
    thrust::gather(thrust::device,
		   d_indices.begin() + start_index, d_indices.begin() + end_index,
		   d_bins.begin(),
		   bin_indices.begin());
    for (int i = 0; i < end_index - start_index; i++) {
      // smaller_bin_dist[f][bin_indices[i]][label_indices[i]]++;
    }
  }

  // int num_classes = smaller_bin_dist.front().size();

  // device_vector<device_vector<int>> d_bins(bins.begin(), bins.end());
  // device_vector<device_vector<device_vector<int>>> d_smaller_bin_dist(smaller_bin_dist.begin(), smaller_bin_dist.end());

  // device_vector<int> f_bins = d_bins.front();
  // device_vector<device_vector<int>> f_bin_dists = d_smaller_bin_dist.front();
  
}
*/
