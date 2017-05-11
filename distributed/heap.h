#ifndef HEAP_H
#define HEAP_H

#include <vector>

using namespace std;

// Naive heap implementation

class Heap {
    private:
        int size;
        int num;
        vector<int> ids;
        vector<float> vals;
    public:
        Heap(int size);

        float max();

        // Removes the max
        void insert(int id, float val);

        vector<int>& get_ids();

        int* data();

        int get_num();
};

#endif
