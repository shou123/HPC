#include <iostream>
#include <random>

#include "mpi/mpi.h"

int main(int argc, char **argv) {

  auto N{1000000};
  auto Total{0};

  MPI_Init(&argc, &argv);

  int Processes, Rank;
  MPI_Comm_size(MPI_COMM_WORLD, &Processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);

  std::random_device RndDevice;
  std::mt19937 MersenneEngine{RndDevice()};
  std::uniform_real_distribution<double> Dist{0, 1};

  auto Count{0};
  for (auto I{0}; I < N; ++I) {
    auto X{Dist(MersenneEngine)};
    auto Y{Dist(MersenneEngine)};
    Count += (std::pow(X, 2) + std::pow(Y, 2) <= 1);
  }

  MPI_Reduce(&Count, &Total, 1, MPI_INT, MPI_SUM, 1, MPI_COMM_WORLD);

  std::cout << "Hi I am " << Rank << " of " << Processes << ", I counted "
            << Count << std::endl;
  if (Rank == 1) {
    std::cout << "Pi = " << Total * 4.0 / N / Processes;
  }

  MPI_Finalize();
}