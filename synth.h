#pragma once

#include <inttypes.h>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <cassert>

class ZRandom {

public:
  enum { N = 624, M = 397 };
  unsigned int MT[N + 1];
  unsigned int *map[N];
  int nValues;

  ZRandom(unsigned int iSeed = 20070102);
  void seed(unsigned iSeed);
  unsigned int getValue();
  unsigned int getValue(const uint32_t MaxValue);
  double getDouble();
  bool test(const double p);
};

ZRandom::ZRandom(unsigned iSeed) : nValues(0) { seed(iSeed); }

void ZRandom::seed(unsigned iSeed) {
  nValues = 0;
  // Seed the array used in random number generation.
  MT[0] = iSeed;
  for (int i = 1; i < N; ++i) {
    MT[i] = 1 + (69069 * MT[i - 1]);
  }
  // Compute map once to avoid % in inner loop.
  for (int i = 0; i < N; ++i) {
    map[i] = MT + ((i + M) % N);
  }
}

inline bool ZRandom::test(const double p) { return getDouble() <= p; }
inline double ZRandom::getDouble() {
  return double(getValue()) * (1.0 / 4294967296.0);
}

unsigned int ZRandom::getValue(const uint32_t MaxValue) {
  unsigned int used = MaxValue;
  used |= used >> 1;
  used |= used >> 2;
  used |= used >> 4;
  used |= used >> 8;
  used |= used >> 16;

  // Draw numbers until one is found in [0,n]
  unsigned int i;
  do
    i = getValue() & used; // toss unused bits to shorten search
  while (i > MaxValue);
  return i;
}

unsigned int ZRandom::getValue() {
  if (0 == nValues) {
    MT[N] = MT[0];
    for (int i = 0; i < N; ++i) {
      unsigned y = (0x80000000 & MT[i]) | (0x7FFFFFFF & MT[i + 1]);
      unsigned v = *(map[i]) ^ (y >> 1);
      if (1 & y)
        v ^= 2567483615;
      MT[i] = v;
    }
    nValues = N;
  }
  unsigned y = MT[N - nValues--];
  y ^= y >> 11;
  y ^= static_cast<unsigned int>((y << 7) & 2636928640);
  y ^= static_cast<unsigned int>((y << 15) & 4022730752);
  y ^= y >> 18;
  return y;
}


class UniformDataGenerator {
public:
  UniformDataGenerator(uint32_t seed = static_cast<uint32_t>(time(NULL)))
      : rand(seed) {}

  void negate(std::vector<uint32_t> &in, std::vector<uint32_t> &out, uint32_t Max) {
    out.resize(Max - in.size());
    in.push_back(Max);
    uint32_t i = 0;
    size_t c = 0;
    for (size_t j = 0; j < in.size(); ++j) {
      const uint32_t v = in[j];
      for (; i < v; ++i)
        out[c++] = i;
      ++i;
    }
    assert(c == out.size());
  }

  /**
   * fill the std::vector with N numbers uniformly picked from  from 0 to Max, not
   * including Max
   * if it is not possible, an exception is thrown
   */
  std::vector<uint32_t> generateUniformHash(uint32_t N, uint32_t Max,
                                       std::vector<uint32_t> &ans) {
    if (Max < N)
      return ans;
    ans.clear();
    if (N == 0)
      return ans; // nothing to do
    ans.reserve(N);
    assert(Max >= 1);
    std::unordered_set<uint32_t> s;
    while (s.size() < N)
      s.insert(rand.getValue(Max - 1));
    ans.assign(s.begin(), s.end());
    sort(ans.begin(), ans.end());
    assert(N == ans.size());
    return ans;
  }

  void fastgenerateUniform(uint32_t N, uint32_t Max, std::vector<uint32_t> &ans) {
    if (2 * N > Max) {
      std::vector<uint32_t> buf(N);
      fastgenerateUniform(Max - N, Max, buf);
      negate(buf, ans, Max);
      return;
    }
    generateUniformHash(N, Max, ans);
  }

  // Max value is excluded from range
  std::vector<uint32_t> generate(uint32_t N, uint32_t Max) {
    std::vector<uint32_t> ans;
    ans.reserve(N);
    fastgenerateUniform(N, Max, ans);
    return ans;
  }
  ZRandom rand;
};

/*
 * Reference: Vo Ngoc Anh and Alistair Moffat. 2010. Index compression using
 * 64-bit words. Softw. Pract. Exper.40, 2 (February 2010), 131-147.
 */
class ClusteredDataGenerator {
public:
  std::vector<uint32_t> buffer;
  UniformDataGenerator unidg;
  ClusteredDataGenerator(uint32_t seed = static_cast<uint32_t>(time(NULL)))
      : buffer(), unidg(seed) {}

  // Max value is excluded from range
  template <class iterator>
  void fillUniform(iterator begin, iterator end, uint32_t Min, uint32_t Max) {
    unidg.fastgenerateUniform(static_cast<uint32_t>(end - begin), Max - Min,
                              buffer);
    for (size_t k = 0; k < buffer.size(); ++k)
      *(begin + k) = Min + buffer[k];
  }

  // Max value is excluded from range
  // throws exception if impossible
  template <class iterator>
  void fillClustered(iterator begin, iterator end, uint32_t Min, uint32_t Max) {
    const uint32_t N = static_cast<uint32_t>(end - begin);
    const uint32_t range = Max - Min;
    if (range < N)
      return;
    assert(range >= N);
    if ((range == N) or (N < 10)) {
      fillUniform(begin, end, Min, Max);
      return;
    }
    const uint32_t cut = N / 2 + unidg.rand.getValue(range - N);
    assert(cut >= N / 2);
    assert(Max - Min - cut >= N - N / 2);
    const double p = unidg.rand.getDouble();
    assert(p <= 1);
    assert(p >= 0);
    if (p <= 0.25) {
      fillUniform(begin, begin + N / 2, Min, Min + cut);
      fillClustered(begin + N / 2, end, Min + cut, Max);
    } else if (p <= 0.5) {
      fillClustered(begin, begin + N / 2, Min, Min + cut);
      fillUniform(begin + N / 2, end, Min + cut, Max);
    } else {
      fillClustered(begin, begin + N / 2, Min, Min + cut);
      fillClustered(begin + N / 2, end, Min + cut, Max);
    }
  }

  // Max value is excluded from range
  std::vector<uint32_t> generate(uint32_t N, uint32_t Max) {
    return generateClustered(N, Max);
  }

  // Max value is excluded from range
  std::vector<uint32_t> generateClustered(uint32_t N, uint32_t Max) {
    std::vector<uint32_t> ans(N);
    fillClustered(ans.begin(), ans.end(), 0, Max);
    return ans;
  }
};
