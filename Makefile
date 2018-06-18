all: vecbench hashbench


vecbench: vecbench.cpp
	c++ -O3 -std=c++11 -o vecbench vecbench.cpp

hashbench: hashbench.cpp
	c++ -O3 -std=c++11 -o hashbench hashbench.cpp


clean:
	rm -r -f vecbench hashbench
