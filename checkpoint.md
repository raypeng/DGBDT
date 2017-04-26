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



## Goals and Deliverables

From profiling the current implementation, we found that the histogram binning step is taking more than half of the total runtime. Instead of putting significant effort to making the algorithm gradient boosted, we decided to further optimize the sequential version first. We planned to use multi-threading to speed up histogram building as the workload is trivially parallelizable across threads. And we planned to use GPU to speed up the step of finding best feature as we can put finding best threshold for a feature to a CUDA thread block. A distributed version of the algorithm remains to be part of our goal and deliverable.

#### Plan to achieve:

* Efficient distributed decision tree training algorithm with histogram binning accelerated by multi-threading on CPUs and parallel feature split finding on GPUs.

#### Nice to have:

* Further adapt the algorithm to use gradient boosting.




## Parallel Competition




## Preliminary Results



## Concerns

