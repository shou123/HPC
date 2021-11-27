#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>

std::mutex GLockPrint;
constexpr int NoOfPhilosophers = 5;

struct Fork {
  std::mutex Mutex;
};

struct Table {
  bool Ready{false};
  std::mutex FinishMutex;
  std::array<bool, NoOfPhilosophers> Finish{};
  std::array<Fork, NoOfPhilosophers> Forks;

  auto allFinished() {
    return Finish.at(0) &&
           std::equal(Finish.begin() + 1, Finish.end(), Finish.begin());
  }
};

struct Philosopher {
  std::string const Name;
  std::size_t Id;

  Table &DinnerTable;
  Fork &LeftFork;
  Fork &RightFork;
  std::thread Lifethread;
  std::mt19937 Rng{std::random_device{}()};

  auto print(std::string_view Text) {
    std::lock_guard<std::mutex> CoutLock(GLockPrint);
    std::cout << std::left << std::setw(3) << std::setfill(' ') << Name << Text
              << std::endl;
  }

  auto eat() {
    std::lock(LeftFork.Mutex, RightFork.Mutex);

    std::lock_guard<std::mutex> LeftLock(LeftFork.Mutex, std::adopt_lock);
    std::lock_guard<std::mutex> RightLock(RightFork.Mutex, std::adopt_lock);

    print(" started eating.");

#ifdef CONST_EATING_TIME
    auto EatingTime = 100;
#else
    static thread_local std::uniform_int_distribution<> Dist(100, 600);
    auto EatingTime{Dist(Rng)};
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(EatingTime));

    print(" finished eating " + std::to_string(EatingTime) + "ms.");
  }

  auto think() {
#ifdef CONST_THINKING_TIME
    auto ThinkingTime{100};
#else
    static thread_local std::uniform_int_distribution<> Dist(100, 600);
    auto ThinkingTime{Dist(Rng)};
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(ThinkingTime));
    print(" is thinking for " + std::to_string(ThinkingTime) + "ms.");
  }

  auto dine() {
    // while (!DinnerTable.Ready)
    //   ;

    // while (DinnerTable.Ready) {
    //   DinnerTable.Finish[Id] = true;
    //   think();
    //   eat();
    // }

    auto CountDown{0};
    static thread_local std::uniform_int_distribution<> Dist(10, 60);
    while (!DinnerTable.Ready) {
      CountDown = Dist(Rng);
      std::this_thread::sleep_for(std::chrono::milliseconds(CountDown));
    }
    while (DinnerTable.Ready) {
      DinnerTable.Finish[Id] = true;
      think();
      eat();
    }
  }

  Philosopher(std::string_view N, std::size_t Id, Table &T, Fork &L, Fork &R)
      : Name(N), Id(Id), DinnerTable(T), LeftFork(L), RightFork(R),
        Lifethread(&Philosopher::dine, this) {}

  ~Philosopher() { Lifethread.join(); }
};

int main() {
  Table Table;
  std::array<Philosopher, NoOfPhilosophers> Philosophers{{
      {"1", 0, Table, Table.Forks[0], Table.Forks[1]},
      {"2", 1, Table, Table.Forks[1], Table.Forks[2]},
      {"3", 2, Table, Table.Forks[2], Table.Forks[3]},
      {"4", 3, Table, Table.Forks[3], Table.Forks[4]},
      {"5", 4, Table, Table.Forks[4], Table.Forks[0]},
  }};

  Table.Ready = true;
  while (!Table.allFinished())
    ;
  Table.Ready = false;

  return 0;
}
