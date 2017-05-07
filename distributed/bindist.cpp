#include "bindist.h"
#include <algorithm>

BinDist::BinDist() {
    // Default constructor does nothing
}

void BinDist::setup(int num_features_, int num_bins_, int num_classes_) {
    num_features = num_features_;
    num_bins = num_bins_;
    num_classes = num_classes_;

    data = new int[num_features * num_bins * num_classes]();
}


void BinDist::reset(int f) {
    int *dist = head(f);
    std::fill(dist, dist + num_bins * num_classes, 0);
}

int* BinDist::head(int f) {
    return &(data[f * num_bins * num_classes]);;
}

int* BinDist::head(int f, int b) {
    return &(data[f * num_bins * num_classes + b * num_classes]);;
}

void BinDist::sum(BinDist& a, BinDist& b) {

    int* this_head = head(0);
    int* ahead = a.head(0);
    int* bhead = b.head(0);

//#pragma omp for schedule(static)
    for (int i = 0; i <  num_features * num_bins * num_classes; i++) {
        this_head[i] = ahead[i] + bhead[i];
    }
}

void BinDist::diff(BinDist& a, BinDist& b) {

    int* this_head = head(0);
    int* ahead = a.head(0);
    int* bhead = b.head(0);

//#pragma omp for schedule(static)
    for (int i = 0; i < num_features * num_bins * num_classes; i++) {
        this_head[i] = ahead[i] - bhead[i];
    }
}

int BinDist::get(int f, int b, int c) {
    return data[f * num_bins * num_classes + b * num_classes + c];
}

void BinDist::inc(int f, int b, int c, int delta) {
    data[f * num_bins * num_classes + b * num_classes + c] += delta;
}

BinDist::~BinDist() {
    delete[] data;
}
