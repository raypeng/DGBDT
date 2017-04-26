---
layout: default
---

# Distributed Gradient Boosted Decision Tree on GPU

Alex Xiao (axiao@andrew.cmu.edu)

Rui Peng (ruip@andrew.cmu.edu)



# Project Checkpoint



## Updated Schedule

April 10 - April 16

Research and brainstorm potential parallel algorithms, setup codebase, start working on distributed sequential version as a reference baseline.

April 17 - April 24

Complete distributed sequential version, measure accuracy and performance, start implementing parallel GPU version.

April 25 - April 30

Work on GPU version, measure accuracy and performance, optimize as necessary.

May 1 - May 7

If finished with GPU version, start implementation of CPU+GPU hybrid algorithm. Start performing measurements needed to analyze speedup and training accuracy.

May 8 - May 12

Wrap up, perform all measurements needed to analyze speedup and training accuracy, prepare presentation.



## Work Completed So Far

So far, we have done a thorough research of ideas that other researchers have tried to parallelize decision trees on a single machine. We’ve discovered that most people agree that finding split points for a single node provides the best opportunity for parallelism rather than trying to split multiple nodes at once. Furthermore, based on the papers we have read that have been published in the last few decades, there are two primary techniques used to optimize parallel decision tree training: using pre-sorted data structures to speed up finding split points and using histogram binning techniques to quantize the training data. We elected to implement the latter, since there have been more recent and successful implementations of these.

In terms of implementation work, we’ve coded up a pretty optimized sequential version of decision tree training that utilizes histogram binning. We’ve benchmarked this against scikit-learn and beat their decision tree training times by over a factor of two.


## Goals and Deliverables

Baseline: a sequential CPU implementation with naive communication scheme of the algorithm on distributed setting.

We will measure the performance of various implementations as the speed up versus the baseline and report the speedup under different settings.

#### Plan to achieve:

* Significant speedup of parallel GPU implementation and efficient communication scheme over baseline version

#### Nice to have:

* Further speedup versus baseline version with an algorithm leveraging CPU+GPU hybrid execution architecture




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

