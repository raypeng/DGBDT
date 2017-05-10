#ifndef UTIL_H
#define UTIL_H

#include <vector>

using namespace std;

// a = b - c
void subtract_vector(vector<int>& a, const vector<int>& b, const vector<int>& c);

// a = b + c
void add_vector(vector<int>& a, const vector<int>& b, const vector<int>& c);

// a = b + c
void add_vector(vector<int>& a, const vector<int>& b, int* c);

bool float_equal(float a, float b);

#endif
