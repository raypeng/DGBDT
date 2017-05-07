#ifndef BINDIST_H
#define BINDIST_H

class BinDist {
    private:
        int ***data;
        int num_features;
        int num_bins;
        int num_classes;
    public:
        BinDist();

        void setup(int num_features, int num_bins, int num_classes);

        void reset(int f);

        void sum(BinDist& a, BinDist& b);

        void diff(BinDist& a, BinDist& b);

        int* head();

        int** operator [](int i) const {return data[i];}

        int** & operator [](int i) {return data[i];}

        ~BinDist();
};

#endif

