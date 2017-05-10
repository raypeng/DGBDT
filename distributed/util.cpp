#include "util.h"

// a = b - c
void subtract_vector(vector<int>& a, const vector<int>& b, const vector<int>& c) {
    for (int i = 0; i < a.size(); i++) {
        a[i] = b[i] - c[i];
    }
}

// a = b + c
void add_vector(vector<int>& a, const vector<int>& b, const vector<int>& c) {
    for (int i = 0; i < a.size(); i++) {
        a[i] = b[i] + c[i];
    }
}

// a = b + c
void add_vector(vector<int>& a, const vector<int>& b, int* c) {
    for (int i = 0; i < a.size(); i++) {
        a[i] = b[i] + c[i];
    }
}

bool float_equal(float a, float b) {
    float diff = a - b;

    return -1e-10 < diff && diff < 1e-10;
}
