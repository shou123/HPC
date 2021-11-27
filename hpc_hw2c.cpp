#include <cmath>
#include <iostream>
#include <omp.h>

double f(double X, double Y) {
  double Result = 0.002;
  for (int I = -2; I <= 2; I++) {
    for (int J = -2; J <= 2; J++) {
      Result += std::abs(1 / (5 * (I + 2) + J + 3 + std::pow(X - 16 * J, 6) +
                              std::pow(Y - 16 * I, 6)));
    }
  }
  return 1.0 / Result;
}

double integrate(double XFrom, double XTo, double YFrom, double YTo,
                 double Precision) {
  int Threads = omp_get_max_threads();
  double XInterval = std::abs((XFrom - XTo)) / (double)Threads;
  double Result = 0;
#pragma omp parallel for
  for (int I = 0; I < Threads; I++) {
    double XFromVal = I * XInterval;
    double XToVal = (I + 1) * XInterval;
    double YFromVal =
        YFrom; // we create these variables, to avoid race condtions between
               // different threads and moreover braking the data. Now, this is
               // a thread-local variable.
    double YToVal = YTo;
    double Sum = 0;
    while (XFromVal <= XToVal) {
      double Y0 = YFromVal;
      while (Y0 <= YToVal) {
        Sum += f((2 * XFromVal + Precision) / 2, (2 * Y0 + Precision) / 2) *
               Precision * Precision;
        Y0 += Precision;
      }
      XFromVal += Precision;
    }
#pragma omp critical
    Result += Sum;
  }
  return Result;
}

int main() {
  auto Results = integrate(0, 10, 1, 12, 0.0001);
  std::cout << Results << std::endl;
}