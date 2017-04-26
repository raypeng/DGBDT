---
layout: default
---

# Distributed Gradient Boosted Decision Tree on GPU

Alex Xiao (axiao@andrew.cmu.edu)

Rui Peng (ruip@andrew.cmu.edu)



# Project Checkpoint



## Updated Schedule

April 10 - April 16

Read papers on decision tree algorithms. Finished dataset I/O and parsing code. Got basic sequential version implementation of decision tree working.

April 17 - April 25

Analyzed and profiled basic implementation and compared to some popular decision tree implementations. Implemented histogram binning as a effective preprocessing step to reduce decition tree building time. Further optimized code and beat performance of scikit-learn decision tree implementation.

April 26 - April 30

Both Alex and Rui will work on getting large datasets on GHC and Latedays servers and benchmark performance with other popular libraries (e.g. xgboost, lightGBM). We will also seek to optimize histogram building step on CPUs with multi-threading support from OpenMP. We may potentially further optimize parameter settings to improve performance.

May 1 - May 7

Rui will work on getting GPU implementation of decision tree training algorithm and Alex will work on adapting current algorithm to work on distributed cluster using MPI.

May 8 - May 12

We will merge our implementations to work on distributed settings leveraging multi-threaded CPUs and GPUs on each node. We will then prepare performance graphs for final presentation and writeup.

## Work Completed So Far

So far, we have done a thorough research of ideas that other researchers have tried to parallelize decision trees on a single machine. We’ve discovered that most people agree that finding split points for a single node provides the best opportunity for parallelism rather than trying to split multiple nodes at once. Furthermore, based on the papers we have read that have been published in the last few decades, there are two primary techniques used to optimize parallel decision tree training: using pre-sorted data structures to speed up finding split points and using histogram binning techniques to quantize the training data. We elected to implement the latter, since there have been more recent and successful implementations of these.

In terms of implementation work, we’ve coded up a pretty optimized sequential version of decision tree training that utilizes histogram binning. We’ve benchmarked this against scikit-learn and beat their decision tree training times by over a factor of two.


## Goals and Deliverables

From profiling the current implementation, we found that the histogram binning step is taking more than half of the total runtime. Instead of putting significant effort to making the algorithm gradient boosted, we decided to further optimize the sequential version first. We planned to use multi-threading to speed up histogram building as the workload is trivially parallelizable across threads. And we planned to use GPU to speed up the step of finding best feature as we can put finding best threshold for a feature to a CUDA thread block. A distributed version of the algorithm remains to be part of our goal and deliverable.

#### Plan to achieve:

* Efficient distributed decision tree training algorithm with histogram binning accelerated by multi-threading on CPUs and parallel feature split finding on GPUs.

#### Nice to have:

* Further adapt the algorithm to use gradient boosting.




## Parallel Competition

We plan to show a couple of graphs detailing how our optimizations impact runtime. Specifically, we are thinking of a graph showing how we scale across multiple nodes,
a graph comparing GPU vs no GPU on a single node, and a graph showing speedups of
histogram binning with multiple CPU threads.

## Preliminary Results

We've been benchmarking our sequential version against scikit-learn and now have
optimized it such that when training a single tree with max leaves of 32 our implementation
beats it scikit-learn by over a factor of 2. Initial profiling shows that our histogram binning
takes up a good portion of our training time, so we will look into parallelizing that.

## Concerns

We are mainly concerned about getting enough space on GHC and latedays to run full experiements. We couldn't get our datasets copied over, so hopefully we will be able to
work with course staff to resolve this.

