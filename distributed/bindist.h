#ifndef BINDIST_H
#define BINDIST_H

class BinDist {
    private:
        int *data;
        int num_features;
        int num_bins;
        int num_classes;
    public:
        BinDist();

        BinDist(int num_features, int num_bins, int num_classes);

        void setup(int num_features, int num_bins, int num_classes);

        void reset(int f);

        void sum(BinDist& a, BinDist& b);

        void diff(BinDist& a, BinDist& b);

        int size();

        int get(int f, int b, int c);

        void inc(int f, int b, int c, int delta = 1);

        int* head(int f);

        int* head(int f, int b);

        ~BinDist();
};

#endif

