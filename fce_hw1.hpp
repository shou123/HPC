#ifndef HW1_HPP_INCLUDED
#define HW1_HPP_INCLUDED

#include "util.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

template <typename T>
void merge(std::vector<T> &A, std::size_t Low, std::size_t Mid,
           std::size_t High) {
  auto I = Low, J = Mid + 1, K = Low;
  std::vector<T> B(A.size());

  // do the loop while i and j less than their subarray
  while (I <= Mid && J <= High) {
    // compare two array's content, which one is small then
    // put into B array
    if (A[I] < A[J])
      B[K++] = A[I++];
    else
      B[K++] = A[J++];
  }
  // check whether remain content in Array A
  for (; I <= Mid; I++)
    B[K++] = A[I];
  // check whether remain content in Array B
  for (; J <= High; J++)
    B[K++] = A[J];
  // copy array B to A
  for (auto It = Low; It <= High; It++)
    A[It] = B[It];
}

template <typename T>
void mergeSort(std::vector<T> &A, std::size_t Low, std::size_t High) {
  if (Low < High) {
    std::size_t Mid = (Low + High) / 2; // decide the middle place in array
    mergeSort(A, Low, Mid);             // keep dividing the front part
    mergeSort(A, Mid + 1, High);        // keep dividing the end part
    merge(A, Low, Mid, High); // compare two sorted subarray and merge into
                              // one array
  }
}

template <class ForwardIt> void quickSort(ForwardIt First, ForwardIt Last) {
  if (First == Last) {
    return;
  }
  auto Pivot = *std::next(First, std::distance(First, Last) / 2);
  auto Middle1 = std::partition(First, Last,
                                [Pivot](const auto &Em) { return Em < Pivot; });
  auto Middle2 = std::partition(
      Middle1, Last, [Pivot](const auto &Em) { return !(Pivot < Em); });
  quickSort(First, Middle1);
  quickSort(Middle2, Last);
}

#endif /* HW1_HPP_INCLUDED */