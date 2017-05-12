//
// Created by Rui Peng on 4/15/17.
//

#include "decision_tree_util.h"

#define UPDIV(n, d) (((n)+(d)-1)/(d))
#define NUM_THREADS_PER_BLOCK 256
#define NUM_CLASSES 5
#define MAX_BINS 255

__global__ void
kernel_update_bin_dist(int* d_bins, int* d_bin_dist, int* d_indices, int* d_y,
		       int start_index, int end_index,
		       int num_bins, int num_classes, int num_samples) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  int num_threads = blockDim.x;
  int f = block_id;
  __shared__ int local_counts[MAX_BINS * NUM_CLASSES];

  for (int i = thread_id; i < MAX_BINS * NUM_CLASSES; i+= num_threads) {
      local_counts[i] = 0;
  }

  __syncthreads();

  // printf("kernel block_id %d [%d - %d]\n", f, start_index, end_index);
  for (int i = start_index + thread_id; i < end_index; i += num_threads) {
    int index = d_indices[i - start_index];
    int label = d_y[index];
    int bin = d_bins[f * num_samples + index];
    atomicAdd(&local_counts[bin * num_classes + label], 1);
  }

  __syncthreads();

  for (int i = thread_id; i < MAX_BINS * NUM_CLASSES; i+= num_threads) {
      d_bin_dist[f * MAX_BINS * NUM_CLASSES + i] += local_counts[i];
  }
}

void initialize_device_memory(vector<vector<int>>& bins,
			      vector<vector<vector<int>>>& smaller_bin_dist,
			      vector<int>& y,
			      int max_bins,
			      int num_classes,
			      int*& d_bins,
			      int*& d_bin_dist,
			      int*& d_indices,
			      int*& d_y,
			      int*& h_bin_dist) {

  int num_features = bins.size();
  int num_samples = bins.front().size();

  double gpu_t = CycleTimer::currentSeconds();

  cudaMalloc(&d_bins, sizeof(int) * num_features * num_samples);
  for (int f = 0; f < num_features; f++) {
    // cout << "bin size " << f << " " << bins[f].size() << endl;
    cudaMemcpy(d_bins + f * num_samples, bins[f].data(), sizeof(int) * num_samples, cudaMemcpyHostToDevice);
  }
  cudaMalloc(&d_bin_dist, sizeof(int) * num_features * max_bins * num_classes);
  cudaMalloc(&d_indices, sizeof(int) * num_samples);
  cudaMalloc(&d_y, sizeof(int) * num_samples);
  cudaMemcpy(d_y, y.data(), sizeof(int) * num_samples, cudaMemcpyHostToDevice);

  h_bin_dist = new int[num_features * max_bins * num_classes];

  cout << "intialize cuda memory: " << CycleTimer::currentSeconds() - gpu_t << endl;

}

void update_smaller_bin_dist(vector<vector<int>>& bins,
			     vector<vector<vector<int>>>& smaller_bin_dist,
			     vector<int>& indices,
			     vector<int>& labels,
			     vector<int>& y,
			     vector<int>& num_bins,
			     int start_index,
			     int end_index,
			     int*& d_bins,
			     int*& d_bin_dist,
			     int*& d_indices,
			     int*& d_y,
			     int*& h_bin_dist,
			     int num_features_gpu,
			     int max_bins,
			     int num_classes) {

  cout << "update bin_dist num_samples: " << end_index - start_index << endl;

  int num_features = bins.size();
  int num_samples = bins.front().size();

  double _t = CycleTimer::currentSeconds();

  if (num_features_gpu > 0) {
    if (d_bins == NULL) {
      initialize_device_memory(bins, smaller_bin_dist, y,
			       max_bins, num_classes,
			       d_bins, d_bin_dist, d_indices, d_y, h_bin_dist);
    } else {
      assert(d_bin_dist != NULL);
      assert(d_indices != NULL);
      assert(d_y != NULL);
      assert(h_bin_dist != NULL);
    }
  }

  double gpu_t = CycleTimer::currentSeconds();

  if (num_features_gpu > 0) {
    cudaMemset(d_bin_dist, 0, sizeof(int) * num_features * max_bins * num_classes);
    cudaMemcpy(d_indices, indices.data() + start_index, sizeof(int) * (end_index - start_index), cudaMemcpyHostToDevice);

    int num_threads_per_block = NUM_THREADS_PER_BLOCK;
    int num_blocks = num_features_gpu;
    kernel_update_bin_dist<<<num_blocks, num_threads_per_block>>>
      (d_bins, d_bin_dist, d_indices, d_y,
       start_index, end_index,
       max_bins, num_classes, num_samples);
  }

  if (num_features_gpu < num_features) {
    update_smaller_bin_dist_cpu(bins, smaller_bin_dist, indices, labels,
				start_index, end_index,
				num_features_gpu, num_features);
  }

  if (num_features_gpu > 0) {
    cudaThreadSynchronize();

    cudaMemcpy(h_bin_dist, d_bin_dist, sizeof(int) * num_features * max_bins * num_classes, cudaMemcpyDeviceToHost);

    for (int f = 0; f < num_features_gpu; f++) {
      for (int bin = 0; bin < num_bins[f]; bin++) {
	for (int label = 0; label < num_classes; label++) {
	  int i = f * max_bins * num_classes + bin * num_classes + label;
	  if (h_bin_dist[i] > 0) {
	    smaller_bin_dist[f][bin][label] = h_bin_dist[i];
	  }
	}
      }
    }

    cout << "exiting kernel: " << num_features_gpu << " features\t" << CycleTimer::currentSeconds() - gpu_t << endl;

  }

  cout << "update time: " << CycleTimer::currentSeconds() - _t << endl;
}

/*
// abandoned thrust code
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
