#include <cmath>
#include <execution>
#include <iostream>
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
    const float t;

  public:
    black_scholes_path_generator(const std::size_t num_steps,
                                 const float s0,
                                 const float sigma,
                                 const float k,
                                 const float r,
                                 const float t)
        : num_steps(num_steps), s0(s0), sigma(sigma), k(k), r(r), t(t) {
    }

    const monaco::path generate() const final {
      monaco::path path;
      for (auto i = 0ul; i < num_steps; i++) {

      }
      return path;
    }
  };

  class european_call_path_evaluator final : public monaco::path_evaluator {
    const float strike;

  public:
    european_call_path_evaluator(const float strike)
        : strike(strike) {
    }

    float evaluate(const monaco::path& path) const final {
      return std::max(path.back().second - strike, float(0));
    }
  };

}

int main(const int argc, const char** const argv) {
  if (argc < 2) {
    std::cerr << "usage:\n\nmontecarlo <num paths>" << std::endl;
    return -1;
  }
  const auto num_paths_long = std::atol(argv[1]);
  if (num_paths_long <= 0L) {
    std::cerr << "number of paths must be greater than zero" << std::endl;
    return -1;
  }
  // const auto num_paths = static_cast<std::size_t>(num_paths_long);
  const auto sign = float(1);
  const auto s0 = float(100);
  const auto sigma = float(0.2);
  const auto k = float(110);
  const auto r = float(0.01);
  const auto t = float(0.5);
  monaco::black_scholes_analytical analytical(sign, s0, sigma, k, r, t);
  std::cout << "analytical solution:    " << analytical.calculate() << std::endl;
  // monaco::black_scholes_path_generator path_generator(num_paths, 100, 0.2, 0.01, 0.5);
  // monaco::european_call_path_evaluator path_evaluator(110);
  // monaco::montecarlo sequential(std::execution::seq, num_paths, path_generator, path_evaluator);
  // std::cout << "monte-carlo sequential: " << sequential.calculate() << std::endl;
  // monaco::montecarlo parallel(std::execution::par, num_paths, path_generator, path_evaluator);
  // std::cout << "monte-carlo parallel:   " << parallel.calculate() << std::endl;
}
