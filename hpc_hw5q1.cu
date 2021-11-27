#include <assert.h>
#include <numeric>
#include <random>
#include <stdio.h>
#include <vector>

#ifdef PARALLEL
#include <omp.h>
#endif

#include <../common/helper_cuda.h>
#include <../common/helper_functions.h>
#include <cuda_runtime.h>

constexpr auto MaxNum{1000000};
constexpr auto NumClass{1 << 8};
constexpr auto Interval{MaxNum / NumClass};

template <typename T> auto populateRandomVector(std::vector<T> &Vec) {
  std::random_device RndDevice;
  std::mt19937 MersenneEngine{RndDevice()};
  std::uniform_int_distribution<T> Dist{0, MaxNum};
  std::generate(Vec.begin(), Vec.end(),
                [&Dist, &MersenneEngine]() { return (Dist(MersenneEngine)); });
}

template <typename T> auto histogramCpu(const std::vector<T> &Vec) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
  std::vector<int> Hist(NumClass);
  for (auto I{0}; I < Vec.size(); ++I) {
    Hist[Vec[I] / Interval]++;
  }
  return Hist;
}

template <typename T>
__global__ void histogramGpu(const T *Vec, const std::size_t N,
                             const int Interval, int *Hist) {
  auto Tid{blockIdx.x * blockDim.x + threadIdx.x};
  if (Tid >= N)
    return;
  atomicAdd(&Hist[Vec[Tid] / Interval], 1);
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

  using T = int;

  auto N{1 << 28};
  auto NumberSize{N * sizeof(T)};
  auto HistSize{NumClass * sizeof(int)};
  std::vector<T> HostVec(N);
  populateRandomVector(HostVec);

  T *DeviceVec = NULL;
  T *DeviceHist = NULL;

  float GpuElapsedTimeMs[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  cudaEvent_t Start, Stop;
  cudaEventCreate(&Start);
  cudaEventCreate(&Stop);

  // Allocate memory
  cudaEventRecord(Start, 0);
  Err = cudaMalloc(reinterpret_cast<void **>(&DeviceVec), NumberSize);
  checkCudaErrors(Err);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorString(Err) << std::endl;
  Err = cudaMalloc(reinterpret_cast<void **>(&DeviceHist), HistSize);
  checkCudaErrors(Err);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorString(Err) << std::endl;
  Err = cudaMemset(DeviceHist, 0, HistSize);
  checkCudaErrors(Err);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorString(Err) << std::endl;

  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&GpuElapsedTimeMs[0], Start, Stop);

  // Copy data to device
  cudaEventRecord(Start, 0);
  Err =
      cudaMemcpy(DeviceVec, HostVec.data(), NumberSize, cudaMemcpyHostToDevice);
  checkCudaErrors(Err);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorString(Err) << std::endl;
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&GpuElapsedTimeMs[1], Start, Stop);

  // Kernel configuration
  cudaEventRecord(Start, 0);
  int ThreadsPerBlock = 128;
  int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

  histogramGpu<<<BlocksPerGrid, ThreadsPerBlock>>>(DeviceVec, N, Interval,
                                                   DeviceHist);
  cudaDeviceSynchronize();
  Err = cudaGetLastError();
  checkCudaErrors(Err);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorString(Err) << std::endl;
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&GpuElapsedTimeMs[2], Start, Stop);

  // copy it back
  cudaEventRecord(Start, 0);
  std::vector<T> FromDeviceHist(NumClass);
  Err = cudaMemcpy(FromDeviceHist.data(), DeviceHist, HistSize,
                   cudaMemcpyDeviceToHost);
  checkCudaErrors(Err);
  if (Err != cudaSuccess)
    std::cout << cudaGetErrorString(Err) << std::endl;
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&GpuElapsedTimeMs[3], Start, Stop);

  // CPU
  auto CpuElapsedTimeMs{0.0f};
  cudaEventRecord(Start, 0);
  auto HostHist{histogramCpu(HostVec)};
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&CpuElapsedTimeMs, Start, Stop);

  std::cout << "Check passed? " << std::boolalpha
            << (FromDeviceHist == HostHist) << std::endl;
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

  cudaFree(DeviceVec);
  cudaFree(DeviceHist);

  return EXIT_SUCCESS;
}
