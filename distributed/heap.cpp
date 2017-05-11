#include "heap.h"
#include <limits>


Heap::Heap(int size_) {
    size = size_;
    num = 0;
    ids.resize(size);
    vals.resize(size, numeric_limits<float>::max());
}

float Heap::max() {
    return vals[size - 1];
}

void Heap::insert(int id, float val) {
    if (num < size) {
        vals[num] = val;
        ids[num] = id;
        num++;
    } else {
        int i = num - 1;
        while (i > 0) {
            if (vals[i - 1] > val) {
                vals[i] = vals[i - 1];
                ids[i] = ids[i - 1];
                i--;
            } else {
                vals[i] = val;
                ids[i] = id;
                return;
            }
        }

        // new min
        vals[0] = val;
        ids[0] = id;
    }
}

vector<int>& Heap::get_ids() {
    return ids;
}

int* Heap::data() {
    return ids.data();
}
