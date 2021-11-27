#include <assert.h>
#include <iostream>
#include <numeric>
#include <random>
#include <stdio.h>
#include <vector>

#ifdef PARALLEL
#include <omp.h>
#endif
#include <cuda_runtime.h>

constexpr int N{1024};
constexpr int NumHist{16};
constexpr int MaxNum{100};
constexpr int Interval{MaxNum / NumHist};

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void checkError(cudaError Err) {
  checkCudaErrors(Err);
  if (Err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(Err));
}

__global__ void histGpu(const int *Numbers, int *Hist) {
  int Index = blockDim.x * blockIdx.x + threadIdx.x;
  if (Index > N)
    return;
  // Hist[Numbers[Index] / Interval]++;
  atomicAdd(&Hist[Numbers[Index] / Interval], 1);
}

void histCpu(const int *Numbers, int *Hist) {
  for (int Index = 0; Index < N; ++Index)
    Hist[Numbers[Index] / Interval]++;
}

int main() {

  cudaError Err = cudaSuccess;

  int *DeviceNumbers = nullptr;
  int *DeviceHist = nullptr;

  int NumSize = N * sizeof(int);
  int HistSize = NumHist * sizeof(int);

  // Step 1 -- allocate memory on the GPU
  cudaMalloc((void **)&DeviceNumbers, NumSize);
  cudaMalloc((void **)&DeviceHist, HistSize);

  // Step 2 -- zero all mem blocks
  cudaMemset(DeviceNumbers, 0, NumSize);
  cudaMemset(DeviceHist, 0, HistSize);

  // Step 3 -- generate host numbers
  int HostNumbers[N] = {};
  for (int I = 0; I < N; ++I) {
    HostNumbers[I] = rand() % MaxNum;
    // printf("%d ", HostNumbers[I]);
  }
  puts("");

  // Step 4 -- copy from host to device
  cudaMemcpy(DeviceNumbers, HostNumbers, NumSize, cudaMemcpyHostToDevice);

  // Step 5 -- kernel computation
  dim3 GridSpec(128);
  dim3 BlockSpec((N + 127) / 128);
  histGpu<<<GridSpec, BlockSpec>>>(DeviceNumbers, DeviceHist);

  // Step 6 -- copy hist from device to host
  int HostHist[N] = {};
  cudaMemcpy(HostHist, DeviceHist, HistSize, cudaMemcpyDeviceToHost);

  int HostHistTruth[NumHist] = {};
  histCpu(HostNumbers, HostHistTruth);
  for (int I = 0; I < NumHist; ++I) {
    printf("%d-%d ", HostHist[I], HostHistTruth[I]);
  }

  return 0;
}