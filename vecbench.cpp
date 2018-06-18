#include <inttypes.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <ctime>

#include "synth.h"


//g++ -std=c++11 -o vecbench vecbench.cpp


typedef std::vector<uint32_t>  vector;

static vector unite(vector& h1, vector& h2) {
  vector v;
  std::set_union(h1.begin(), h1.end(),h2.begin(), h2.end(),std::back_inserter(v));
  return v;
}


vector getRandomSet(ClusteredDataGenerator &c, uint32_t N, uint32_t Max) {
  return c.generateClustered(N,Max);
}


int main(int argc, char* argv[]) {
  if(argc < 2) {
    printf("provide numerical value.\n");
    return EXIT_FAILURE;
  }
  uint64_t end=atol(argv[1]);
  ClusteredDataGenerator c;
  printf("# maxrange N1 N2 finalcard time(s)\n");
  for (int t = 0; t < end; t++) {
    uint32_t Max = 0;
    while(Max < 1000000) Max = rand()% 20000000;
    uint32_t N1 = rand() % Max;
    uint32_t N2 = rand() % Max;
    vector set1 = getRandomSet(c, N1, Max);
    vector set2 = getRandomSet(c, N1, Max);

    auto start = std::chrono::system_clock::now();

    vector answer = unite(set1, set2);
    size_t card = answer.size();

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;


    printf("%u %u %u %zu %f \n", Max, N1, N2, card, elapsed_seconds.count());

  }
  return EXIT_SUCCESS;
}
