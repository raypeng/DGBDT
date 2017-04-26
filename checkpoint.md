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



## Goals and Deliverables

Baseline: a sequential CPU implementation with naive communication scheme of the algorithm on distributed setting.

We will measure the performance of various implementations as the speed up versus the baseline and report the speedup under different settings.

#### Plan to achieve:

* Significant speedup of parallel GPU implementation and efficient communication scheme over baseline version

#### Nice to have:

* Further speedup versus baseline version with an algorithm leveraging CPU+GPU hybrid execution architecture




## Parallel Competition




## Preliminary Results



## Concerns

