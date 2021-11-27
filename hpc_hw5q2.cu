#include <assert.h>
#include <numeric>
#include <random>
#include <stdio.h>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#ifdef PARALLEL
#include <omp.h>
#endif

constexpr auto N{256};
constexpr auto MaxNum{100};

template <typename T>
auto stencilCpu(T A[][N][N], const T B[][N][N], const std::size_t N) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
  for (auto I{1}; I < N - 1; I++)
    for (auto J{1}; J < N - 1; J++)
      for (auto K{1}; K < N - 1; K++) {
        A[I][J][K] = 0.8 * (B[I - 1][J][K] + B[I + 1][J][K] + B[I][J - 1][K] +
                            B[I][J + 1][K] + B[I][J][K - 1] + B[I][J][K + 1]);
        // A[I][J][K] = I + J + K;
      }
}

// template <typename T> __global__ void stencilGpuTiled(T *A, const T *B) {
//   __shared__ T sA[TailSize][TailSize];
//   __shared__ T sB[TailSize][TailSize];

//   auto Row{blockDim.y * blockIdx.y + threadIdx.y};
//   auto Col{blockDim.x * blockIdx.x + threadIdx.x};
// }

template <typename T>
__global__ void stencilGpuNonTiled(T A[][N][N], const T B[][N][N],
                                   const std::size_t N) {
  auto I{blockDim.x * blockIdx.x + threadIdx.x};
  auto J{blockDim.y * blockIdx.y + threadIdx.y};
  auto K{blockDim.z * blockIdx.z + threadIdx.z};

  if ((I > N - 2) || (J > N - 2) || (K > N - 2) || (I < 1) || (J < 1) ||
      (K < 1))
    return;

  A[I][J][K] = 0.8 * (B[I - 1][J][K] + B[I + 1][J][K] + B[I][J - 1][K] +
                      B[I][J + 1][K] + B[I][J][K - 1] + B[I][J][K + 1]);
}

int main(int argc, char **argv) {

  auto Err{cudaSuccess};

  // This will pick the best possible CUDA capable device
  int DevId;
  DevId = findCudaDevice(argc, (const char **)argv);

  // Get GPU information
  cudaDeviceProp Props;
  checkCudaErrors(cudaGetDevice(&DevId));
  checkCudaErrors(cudaGetDeviceProperties(&Props, DevId));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", DevId, Props.name,
         Props.major, Props.minor);

  using T = float;
  typedef T AT[N][N];
  auto NumberSize{N * N * N * sizeof(T)};

  // Populate matrix
  AT *HostA = nullptr;
  AT *HostB = nullptr;
  HostA = (AT *)malloc(NumberSize);
  HostB = (AT *)malloc(NumberSize);
  for (auto I{0}; I < N; ++I)
    for (auto J{0}; J < N; ++J)
      for (auto K{0}; K < N; ++K) {
        // HostB[I][J][K] = 1;
        HostB[I][J][K] = static_cast<T>(rand() % MaxNum);
        HostA[I][J][K] = 0;
      }

  float GpuElapsedTimeMs[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  cudaEvent_t Start, Stop;
  cudaEventCreate(&Start);
  cudaEventCreate(&Stop);
  AT *DeviceA;
  AT *DeviceB;

  // Allocate memory
  cudaEventRecord(Start, 0);
  Err = cudaMalloc(reinterpret_cast<void **>(&DeviceA), NumberSize);
  Err = cudaMemset(DeviceA, 0.0f, NumberSize);
  Err = cudaMalloc(reinterpret_cast<void **>(&DeviceB), NumberSize);
  Err = cudaMemset(DeviceB, 0.0f, NumberSize);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&GpuElapsedTimeMs[0], Start, Stop);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorString(Err) << std::endl;
  checkCudaErrors(Err);

  // Copy to device
  cudaEventRecord(Start, 0);
  Err = cudaMemcpy(DeviceB, HostB, NumberSize, cudaMemcpyHostToDevice);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&GpuElapsedTimeMs[1], Start, Stop);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorName(Err) << std::endl;
  checkCudaErrors(Err);

  // Compute
  const dim3 BlockSize(9, 9, 9);
  const dim3 GridSize((N + 8) / 9, (N + 8) / 9, (N + 8) / 9);
  cudaEventRecord(Start, 0);
  stencilGpuNonTiled<<<GridSize, BlockSize>>>(DeviceA, DeviceB, N);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&GpuElapsedTimeMs[2], Start, Stop);

  // Copy back
  AT *FromDeviceA = nullptr;
  FromDeviceA = (AT *)malloc(NumberSize);
  cudaEventRecord(Start, 0);
  Err = cudaMemcpy(FromDeviceA, DeviceA, NumberSize, cudaMemcpyDeviceToHost);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&GpuElapsedTimeMs[3], Start, Stop);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorString(Err) << std::endl;
  checkCudaErrors(Err);

  // CPU
  auto CpuElapsedTimeMs{0.0f};
  cudaEventRecord(Start, 0);
  stencilCpu(HostA, HostB, N);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&CpuElapsedTimeMs, Start, Stop);

  auto ErrFlag{false};
  for (auto I{0}; I < N; ++I)
    for (auto J{0}; J < N; ++J)
      for (auto K{0}; K < N; ++K)
        if (FromDeviceA[I][J][K] != HostA[I][J][K]) {
          ErrFlag = true;
          break;
        }
  std::cout << "Check passed? " << std::boolalpha << !ErrFlag << std::endl;
  std::cout << "GPU Time [ms]: ";
  std::for_each(std::begin(GpuElapsedTimeMs), std::end(GpuElapsedTimeMs),
                [&](float I) { std::cout << I << " "; });
  std::cout << std::endl;
  std::cout << "CPU Time [ms]: " << CpuElapsedTimeMs << std::endl;
  std::cout << "Speed up [xN]: "
            << CpuElapsedTimeMs /
                   (std::accumulate(std::begin(GpuElapsedTimeMs),
                                    std::end(GpuElapsedTimeMs), 0.0f))
            << " " << CpuElapsedTimeMs / GpuElapsedTimeMs[2] << std::endl;

  cudaFree(DeviceA);
  cudaFree(DeviceB);
  free(FromDeviceA);
  free(HostA);
  free(HostB);

  return EXIT_SUCCESS;
}
