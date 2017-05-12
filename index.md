---
layout: default
---

# Distributed Decision Trees with Heterogeneous Parallelism

Alex Xiao (axiao@andrew.cmu.edu)

Rui Peng (ruip@andrew.cmu.edu)

* [link to proposal](proposal.html)
* [link to checkpoint](checkpoint.html)
* [link to code](https://github.com/raypeng/DGBDT)


## Summary

Decision tree learning is one of the most popular supervised classification
algorithms used in machine learning. In our project, we attempted to optimize decision tree learning
by parallelizing training on a single machine (using multi-core CPU parallelism, GPU parallelism, and a hybrid of the two) and
across multiple machines in a cluster. Initial results show performance gains from all forms of parallelism.
In particular, our hybrid, single machine implementation on GHC achieves an 8 second training
time for a [dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS) with over 11 million samples, which is
**60 times faster** than sci-kit learn and **24 times faster** than our optimized
sequential version, with similar accuracy.


![Summary](assets/runtime-summary.png)


## Background

Decision trees are a common model used in machine learning and data mining to approximate regression or classification functions.
They model functions of the form:

![equation](assets/equation.png)

Where x~1 through x~n are the features of the input and y is the output.

A decision tree models this function with a series of queries of the form "x~i < v?", where v is a value of the feature.
For example, below is an example of a decision tree used to predict whether or not a
passenger survived the Titanic, based on the features gender, age, and
number of siblings/spouses.

<br>

![tree image](assets/tree.png)

<br>

For the purposes of this project, we will focus on classification decision
trees, where we are trying to predict the class label of an input, such as in
the example above.

While building the tree, decision tree training algorithms would need to
evaluate potential split points in the form of "feature f > x?" for each node.
The data will then be partitioned on that split point and this process is
repeated until the
tree becomes large enough. The evaluation for a split point is usually based on some kind of metric that captures the distribution of the class labels of the data after the split.
For example, a common criteria to use is entropy, which is defined below
for when there are J class labels.

![entropy](assets/entropy.png)

The criteria for a split point for feature f and value x would then be the weighted sum of the entropy of the left and right child.


the When the feature values are continuous, it is more efficient to compute this weighted sum
for each split point by first sorting the values based on feature f and scanning
through the sorted list. This way we can maintain the left and right
distributions of the class labels and evaluate all split points for a feature.
Below is pseudo-code for a (binary) decision tree training algorithm that achieves this.

<pre>

// root contains all the data
work_queue.add(root)

while (!work_queue.empty()) {
  node = work_queue.remove_head()

  if (node.is_terminal()) continue

  best_split_point = nil

  for f in features {
    sort(node.data, comparator = f)

    for d in node.data {
      // check this split point based off some criteria, like entropy
      check_best_split_point(best_split_point,f,d)
    }
  }

  // partition data based on split point
  left,right = split(node, best_split_point)

  work_queue.add(left)
  work_queue.add(right)
}
</pre>

Since decision trees are created by splitting on feature values, which
can often be continuous numbers, sorting the
data is required to efficiently compute distribution statistics of
the split while scanning through data in the inner loop. Unfortunately,
the standard decision tree construction algorithm is slow even for sequential
standards, since repeated sorting of data becomes a bottleneck. One common
optimization for this is to first preprocess the dataset by constructing
histograms to compactly describe the distribution of the data for each feature.

For example, consider the image below showing the datapoints ordered by some
feature. Instead of

![Histogram binning](assets/hist_bin.png)

### Challenges

* Building an optimized sequential implementation of decision tree learning to use as
  a baseline requires some work, since the default decision tree training algorithm
  is slow and requires repeatedly sorting the dataset, which can be massive.

* Parallelizing training with a machine on CPU cores is also tricky, since the
  shape of the decision tree is irregular and determined at runtime, making
  static partitioning of the workload across tree nodes ineffective.

* Distributing training across machines in a cluster requires significant
  communication between machines, since the decision on which feature to split
  on requires a global view of the dataset.

* The standard algorithm for decision tree learning does not translate well to GPU or
  hybrid implementations. To quote the creator of XGBoost, a widely used decision tree
  boosting framework: “The execution pattern of decision tree training relies heavily
  on conditional branches and thus has high ratio of divergent execution,
  which makes the algorithm have less benefit from SPMD architecture”. Our
  implementation of decision tree learning must not have the same problems.

* Scheduling GPU and CPU computation on a heterogenous machine is difficult,
  since it is crucial to identify scenarios in which one is preferred over
  the other or if the overhead of using both is worth the trouble.

## Optimizing a Sequential Implementation

The standard sequential implementation for decision tree learning looks
something like this:





To improve upon this, we implemented an algorithm that first builds a
histogram of every feature that roughly captures the distribution statistics
of the data. Using this algorithm, training roughly looks like this:

<pre>
build_histograms()

while (!work_queue.empty()) {
  node = work_queue.remove_head()

  if (node.is_terminal()) continue

  best_split_point = nil

  for f in features {
    for bin in node.histogram(f) {
      check_best_split_point(best_split_point,f,bin)
    }
  }

  left,right = split(node, best_split_point)

  left.compute_histograms(node.histogram, best_split_point)
  right.compute_histograms(node.histogram, best_split_point)

  work_queue.add(left)
  work_queue.add(right)

}
</pre>

This eliminates sorting the data and also scans over histogram
bins instead of data points. Since number of bins (set to a constant value like
255) <<<< number of datapoints, this provides a big performance gain. The main
computation is now offloaded to building the initial histograms and constructing new
histograms from old histograms. To do this efficiently, we use an adaptive
histogram construction algorithm from this
[paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/boosttreerank.pdf) and compute the left/right
child histograms by first computing the smaller one, then performing histogram
subtraction to get the larger one.

Below are performance comparisons between our implementation of the
above two algorithms and the popular decision tree learning framework sci-kit learn.
We benchmarked on the [Microsoft Learn to
Rank](https://www.microsoft.com/en-us/research/project/mslr/) dataset, which contains 2
million (query,url) pairs, each with 136 features and a label denoting
the relevance of the query to the url. Experiments were performed on GHC.


![Sequential](assets/runtime-sequential.png)


Our final sequential version observes massive improvements over both sci-kit learn and
the traditional decision tree learning algorithm. This is the optimized sequential version
we will use as a baseline later.
Note that although our accuracy has also decreased slightly due to the approximate
nature of our histogram binning, the reduction is small (on the order of 0.1)
and not much of a concern if we used our algorithm in an ensemble method (which
people often do with decision trees). Furthermore, since our focus is on
performance, we decided to not spend too
much time on sophisticated splitting heuristics and pruning techniques that are found in mature
frameworks as long as our accuracy is competitive.

## Parallelizing with Multiple CPU Cores

As mentioned previously parallelizing across tree nodes leads to the problem of
an imbalanced workload. So we want to instaed parallelize finding a feature
to split on within a single node. After profiling our code, we determined two
computationally expensive areas that might be opportunities for parallelism:

1. Initial building of histograms.
2. Constructing child histograms from each node.

We used OpenMP to parallelize these two areas across features. Since building
histograms and constructing child histograms requires scanning over
the distributions of every histogram bin of every feature, this leads to a
roughly balanced workload. The speedup graphs are shown below (experiments on
GHC).


![OpenMP](assets/runtime-openmp.png)


The speedup graph above at first displays near-linear speedup, especially for
histogram construction across different features. The eventual
dissipation of speedup is very likely due to memory bandwidth issues, since
histogram construction is trivially parallelizable across features and needs
minimal syncrhonization. We aim to solve this
problem through using the GPU and hybrid parallelism mentioned later.

## Distributing Training with Multiple Machines

A major concern we had initially, communication efficiency of distributed
training, is somewhat alleviated by our histogram representation of the dataset.
This allows multiple machines to communicate with histograms instead of
their partition of the dataset, drastically reducing the communication
requirements.

Our distributed training algorithm is basically to assign each machine on latedays
to have a partition of the dataset, and construct local histograms. By
doing this in a data parallel fashion, we should gain great speedups for
histogram construction and hopefully good performance gains as well
in the tree building phase (where communication and synchronization costs might dominate).
Whenever we decide how to split a tree node, the workers send their histograms to the
master, which merges them together to search for a split point. The master
then sends the split point to the workers, and each worker
builds local child histograms individually.
Initial results, however, show that communication efficiency
is still a problem. The experiment below was run on the latedays cluster on a
varying number of workers/machines.


![OpenMPI](assets/runtime-openmpi.png)


As you can see from the graph, histogram construction scales well due to the
trivial communication requirements necessary for it. On the other hand,
communication during tree building is expensive, since the root
machine must communicate with all other machines on every node split for every
feature. We plan to further optimize our distributed training
code by reducing the amount of information each machine needs to send to
the root by performing some local computation first to get this to scale beyond
2 nodes.

## GPU and Hybrid Implementation

Another advantage of our histogram implementation is that the main bottleneck during
tree construction is computing child histograms, which requires a lot of moving
data around and incrementing counters in memory. This kind of computation lends
itself well to a GPU implementation. This also motivates a hybrid
algorithm: build the initial histograms using multi-threaded CPU (getting
GPU to work for adapative histogram building will take some work), and
use both the GPU and CPU to accelerate child histogram computation (pseudo-code
below):

<pre>
gpu_features, cpu_features = assign_features(features)
gpu_result = []
cpu_result = []

// Asynchronously compute on gpu
kernel_gpu_compute_histograms(gpu_features, gpu_result)

cpu_compute_histograms(cpu_features, cpu_result)

cudaThreadSynchronize()

merge_results(gpu_result, cpu_result)
</pre>

Since the speedup graph for CPU suggests that our algorithm may be bandwidth
bound, an implementation that uses both the memory bandwidth of GPU and CPU will
likely be faster. Initial results show that hybrid reduces tree building time by 20%
over GPU only when running on the [HIGGS Data Set](https://archive.ics.uci.edu/ml/datasets/HIGGS),
which has 11 million samples. Both the GPU only and hybrid only implementation
are improvements over a multi-core CPU implementation with 16 threads.
likely be faster. We are currently playing around with scheduling strategies (such
as scheduling based on the size of the node we are woking with).
Initial results show that hybrid reduces tree building time by 20%
over GPU only when running on the [HIGGS Data Set](https://archive.ics.uci.edu/ml/datasets/HIGGS),
which has 11 million samples. Both the GPU only and hybrid only implementations
are improvements over a multi-core CPU implementation with 16 threads.
We are working on optimizing this further with a better scheduling strategy.

## Further Work

We have two main goals to focus on:

1. Improve our GPU implementation and hybrid scheduling. Currently the GPU
   implementation is pretty simple, which might make hybrid scheduling
   not as effective as it could be. We would
   like to look into further improving the GPU implementation to show the
   advantages of hybrid scheduling. Specifically, we are planning to reduce
   the memory movement between host and device that occurs when training.
2. Improve the communication efficiency of our distributed implementation. We
   found a [paper](https://arxiv.org/abs/1611.01276) recently published at
   NIPS that will help us in this regard.
   We hope that implementing their idea will allow us to scale beyond two
   machines.

Picture sources:

https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png
