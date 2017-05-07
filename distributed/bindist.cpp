#include "bindist.h"
#include <algorithm>

BinDist::BinDist() {
    // Default constructor does nothing
}

void BinDist::setup(int num_features_, int num_bins_, int num_classes_) {
    num_features = num_features_;
    num_bins = num_bins_;
    num_classes = num_classes_;

    data = new int**[num_features];

    for (int i = 0; i < num_features; i++) {
        data[i] = new int*[num_bins]();
        for (int j = 0; j < num_classes; num_classes++) {
            data[i][j] = new int[num_classes]();
        }
    }
}

void BinDist::reset(int f) {
    int *dist = *(data[f]);
    std::fill(dist, dist + num_bins * num_classes, 0);
}

int* BinDist::head() {
    return data[0][0];
}

void BinDist::sum(BinDist& a, BinDist& b) {

    int* this_head = head();
    int* ahead = a.head();
    int* bhead = b.head();

//#pragma omp for schedule(static)
    for (int i = 0; i <  num_features * num_bins * num_classes; i++) {
        this_head[i] = ahead[i] + bhead[i];
    }
}

void BinDist::diff(BinDist& a, BinDist& b) {

    int* this_head = head();
    int* ahead = a.head();
    int* bhead = b.head();

//#pragma omp for schedule(static)
    for (int i = 0; i < num_features * num_bins * num_classes; i++) {
        this_head[i] = ahead[i] - bhead[i];
    }
}

BinDist::~BinDist() {

    for (int i = 0; i < num_features; i++) {
        for (int j = 0; j < num_bins; j++) {
            delete[] data[i][j];
        }
        delete data[i];
    }
    delete[] data;
}
