#include <inttypes.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <ctime>
#include "synth.h"

//g++ -std=c++11 -o hashbench hashbench.cpp

typedef std::unordered_set<uint32_t>  hashset;


static hashset unite(hashset& h1, hashset& h2) {
  hashset answer;
  answer.insert(h1.begin(), h1.end());
  answer.insert(h2.begin(), h2.end());
  return answer;
}

hashset getRandomSet(ClusteredDataGenerator &c, uint32_t N, uint32_t Max) {
  std::vector<uint32_t> rv = c.generateClustered(N,Max);
  hashset answer;
  answer.insert(rv.begin(), rv.end());
  return answer;
}

int main(int argc, char* argv[]) {
  if(argc < 2) {
    printf("provide numerical value.\n");
    return EXIT_FAILURE;
  }
  uint64_t end=atol(argv[1]);
  ClusteredDataGenerator c;
  printf("# maxrange N1 N2 finalcard time(s)\n");
  for (int t = 0; t < 1000; t++) {
    uint32_t Max = 0;
    while(Max < 1000000) Max = rand()% 20000000;
    uint32_t N1 = rand() % Max;
    uint32_t N2 = rand() % Max;
    hashset set1 = getRandomSet(c, N1, Max);
    hashset set2 = getRandomSet(c, N1, Max);

    auto start = std::chrono::system_clock::now();

    hashset answer = unite(set1, set2);
    size_t card = answer.size();

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;


    printf("%u %u %u %zu %f \n", Max, N1, N2, card, elapsed_seconds.count());

  }
  return EXIT_SUCCESS;
}
