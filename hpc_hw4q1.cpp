/*
 *  To compile: mpicxx src/hpc_hw4q1.cpp
 *  To run: mpirun -n 64 a.out
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "mpi/mpi.h"

auto calculatePi(int N) {
  std::random_device RndDevice;
  std::mt19937 MersenneEngine{RndDevice()};
  std::uniform_real_distribution<double> Dist{0, 1};

  auto Count{0};
  for (auto I{0}; I < N; ++I) {
    auto X{Dist(MersenneEngine)};
    auto Y{Dist(MersenneEngine)};
    Count += (std::pow(X, 2) + std::pow(Y, 2) <= 1);
  }
  return Count * 1.0 / N * 4;
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int Processes, Rank;
  MPI_Comm_size(MPI_COMM_WORLD, &Processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);

  auto N{100000000};
  auto Total{0.};
  auto SubTotal{calculatePi(N) * 1.0};

  MPI_Reduce(&SubTotal, &Total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (Rank == 0) {
    auto Pi{Total / Processes};
    std::cout << ' ' << std::setprecision(30) << Pi << ' '
              << std::fabs(std::acos(-1.0) - Pi) << std::endl;
  }
  MPI_Finalize();

  return 0;
}