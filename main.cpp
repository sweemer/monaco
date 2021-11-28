#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <random>
#include <string_view>
#include "montecarlo.hpp"

namespace monaco {

  class black_scholes_analytical {
    const float sign;     // +1 for call, -1 for put
    const float s0;       // initial spot price
    const float sigma;    // volatility
    const float k;        // strike
    const float r;        // interest rate
    const float t;        // term in years

    float cndf(float x) {
      return std::erfc(-x / std::sqrt(float(2))) / float(2);
    }

  public:
    black_scholes_analytical(const float sign,
                             const float s0,
                             const float sigma,
                             const float k,
                             const float r,
                             const float t)
        : sign(sign), s0(s0), sigma(sigma), k(k), r(r), t(t) {
    }

    float calculate() {
      const auto d1 = (std::log(s0 / k) + (r + sigma * sigma / float(2)) * t) / (sigma * std::sqrt(t));
      const auto d2 = d1 - sigma * std::sqrt(t);
      return sign * s0 * cndf(sign * d1) - sign * std::exp(-r * t) * k * cndf(sign * d2);
    }
  };

  class black_scholes_path_generator final : public monaco::path_generator {
    const std::size_t num_steps;
    const float s0;
    const float sigma;
    const float k;
    const float r;
    const float T;

  public:
    black_scholes_path_generator(const std::size_t num_steps,
                                 const float s0,
                                 const float sigma,
                                 const float k,
                                 const float r,
                                 const float T)
        : num_steps(num_steps), s0(s0), sigma(sigma), k(k), r(r), T(T) {
    }

    const monaco::path generate() const final {
      std::random_device random_device;
      std::mt19937 generator(random_device());
      std::normal_distribution<float> gaussian(0, 1);
      monaco::path path;
      path.reserve(num_steps + 1);
      auto s = s0;
      auto t = float(0);
      // path.push_back(std::make_pair(t, s));
      const auto dt = T / static_cast<float>(num_steps);
      const auto sqrt_dt = std::sqrt(dt);
      for (auto i = 0ul; i < num_steps; i++) {
        t += dt;
        s += r * s * dt + sigma * s * sqrt_dt * gaussian(generator);
        path.push_back(std::make_pair(t, s));
      }
      return path;
    }
  };

  class european_call_path_evaluator final : public monaco::path_evaluator {
    const float sign;
    const float k;

  public:
    european_call_path_evaluator(const float sign, const float k)
        : sign(sign), k(k) {
    }

    float evaluate(const monaco::path& path) const final {
      return std::max(sign * (path.back().second - k), float(0));
    }
  };

  template<typename Lambda>
  void time_it(const std::string_view description, Lambda&& lambda) {
    const auto start = std::chrono::system_clock::now();
    const auto return_value = lambda();
    const auto end = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << description << return_value << " (" << millis << "ms)" << std::endl;
  }

}

int main(const int argc, const char** const argv) {
  if (argc < 2) {
    std::cerr << "usage:\n\nmontecarlo <num paths>" << std::endl;
    return -1;
  }

  const auto num_paths_long = std::atol(argv[1]);
  const auto num_paths = static_cast<std::size_t>(num_paths_long);
  const auto num_steps = 100;
  const auto s0 = float(100);
  const auto sigma = float(0.2);
  const auto k = float(110);
  const auto r = float(0.01);
  const auto t = float(0.5);
  std::array<float, 2> signs {float(-1), float(1)};

  if (num_paths_long <= 0L) {
    std::cerr << "number of paths must be greater than zero" << std::endl;
    return -1;
  }

  for (const auto sign : signs) {
    std::cout << "-- sign = " << sign << std::endl;
    auto path_generator = std::make_shared<monaco::black_scholes_path_generator>(num_steps, s0, sigma, k, r, t);
    auto path_evaluator = std::make_shared<monaco::european_call_path_evaluator>(sign, k);

    monaco::black_scholes_analytical analytical(sign, s0, sigma, k, r, t);
    monaco::montecarlo sequential(std::execution::seq, num_paths, path_generator, path_evaluator);
    monaco::montecarlo parallel(std::execution::par, num_paths, path_generator, path_evaluator);

    monaco::time_it("analytical solution    : ", [&analytical]() { return analytical.calculate(); });
    monaco::time_it("monte-carlo sequential : ", [&sequential]() { return sequential.calculate(); });
    monaco::time_it("monte-carlo parallel   : ", [&parallel]()   { return parallel.calculate(); });
  }
}
