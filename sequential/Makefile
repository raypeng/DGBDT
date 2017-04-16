CXXFLAGS='-std=c++11'
g++='/opt/gcc/4.9.2/bin/g++'

OBJS = main.o dataset.o decision_tree.o

main: $(OBJS)
	g++ $(CXXFLAGS) -o $@ $(OBJS) -lm

main.o: main.cpp dataset.h decision_tree.h mypprint.hpp
	g++ $(CXXFLAGS) -c main.cpp

dataset.o: dataset.cpp dataset.h mypprint.hpp
	g++ $(CXXFLAGS) -c dataset.cpp

decision_tree.o: decision_tree.cpp decision_tree.h mypprint.hpp
	g++ $(CXXFLAGS) -c decision_tree.cpp

clean:
	rm -f $(OBJS) main *~
