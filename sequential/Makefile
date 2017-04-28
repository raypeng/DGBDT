CXXFLAGS='-std=c++11'

OBJS = main.o dataset.o decision_tree.o
OBJS2 = main2.o dataset2.o decision_tree2.o

main: $(OBJS)
	g++ $(CXXFLAGS) -o $@ $(OBJS) -lm

openmp: $(OBJS2)
	g++ $(CXXFLAGS) -fopenmp -o $@ $(OBJS2) -lm

main.o: main.cpp dataset.h decision_tree.h mypprint.hpp
	g++ $(CXXFLAGS) -c main.cpp

dataset.o: dataset.cpp dataset.h mypprint.hpp
	g++ $(CXXFLAGS) -c dataset.cpp

decision_tree.o: decision_tree.cpp decision_tree.h mypprint.hpp
	g++ $(CXXFLAGS) -c decision_tree.cpp

main2.o: main.cpp dataset.h decision_tree.h mypprint.hpp
	g++ $(CXXFLAGS) -fopenmp -c main.cpp -o main2.o

dataset2.o: dataset.cpp dataset.h mypprint.hpp
	g++ $(CXXFLAGS) -fopenmp -c dataset.cpp -o dataset2.o

decision_tree2.o: decision_tree.cpp decision_tree.h mypprint.hpp
	g++ $(CXXFLAGS) -fopenmp -c decision_tree.cpp -o decision_tree2.o

clean:
	rm -f $(OBJS) $(OBJS2) main *~
