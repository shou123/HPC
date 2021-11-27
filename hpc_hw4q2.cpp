#include <algorithm>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "mpi.h"

constexpr auto MaxNum{54};

template <typename T> auto populateRandomVector(std::vector<T> &Vec) {
  std::random_device RndDevice;
  std::mt19937 MersenneEngine{RndDevice()};
  std::normal_distribution<double> Dist{32, 2};
  std::generate(Vec.begin(), Vec.end(), [&Dist, &MersenneEngine]() {
    return std::round(Dist(MersenneEngine));
  });
}

template <typename T> auto getHistogram(std::vector<T> &Vec) {
  // std::map<int, int> Hist{};
  std::vector<int> Hist(MaxNum + 1);
  for (auto V : Vec) {
    ++Hist[std::round(V)];
  }
  return Hist;
}

int main(int argc, char **argv) {
  using T = int;
  auto MPI_T{MPI_INT};

  MPI_Init(&argc, &argv);

  int Processes, Rank;
  MPI_Comm_size(MPI_COMM_WORLD, &Processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);

  // Step 1 -- populate the vector and broadcast it
  auto N{1024 * 1024 * 8};
  std::vector<T> Vec(N);
  populateRandomVector(Vec);
  // MPI_Bcast(Vec.data(), Vec.size(), MPI_T, 0, MPI_COMM_WORLD);

  // Step 2 -- scatter the numbers to multiple vectors
  auto Portion{N / Processes};
  std::vector<T> PVec(Portion);
  MPI_Scatter(Vec.data(), Portion, MPI_T, PVec.data(), Portion, MPI_T, 0,
              MPI_COMM_WORLD);

  // Step 3 -- perform the histogram operation
  auto PartialHist{getHistogram(PVec)};

  // Step 4 -- reduce the data
  std::vector<T> Hist;
  if (0 == Rank) {
    Hist.resize((MaxNum + 1));
  }
  MPI_Reduce(PartialHist.data(), Hist.data(), MaxNum + 1, MPI_T, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // Step 5 -- simple graph
  if (0 == Rank) {
    std::map<int, int> HistMap{};
    for (auto V : Vec) {
      ++HistMap[std::round(V)];
    }
    for (auto P : HistMap) {
      std::cout << std::setw(2) << P.first << ' '
                << std::string(P.second / 65536, '*') << '\n';
    }
    auto HistSerial{getHistogram(Vec)};
    std::cout << "Serial equals to parallel? " << std::boolalpha
              << (HistSerial == Hist) << std::endl;
  }
  MPI_Finalize();

  return 0;
}