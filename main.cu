#include <chrono>
#include <cmath>
#include <cstddef>
#include <execution>
#include <iostream>
#include <string_view>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

namespace monaco {

  class black_scholes_analytical {
    const float sign;     // +1 for call, -1 for put
    const float s0;       // initial spot price
    const float sigma;    // volatility
    const float k;        // strike
    const float r;        // interest rate
    const float T;        // term in years

    static float cndf(float x) {
      return std::erfc(-x / std::sqrt(float(2))) / float(2);
    }

  public:
    black_scholes_analytical(const float sign,
                             const float s0,
                             const float sigma,
                             const float k,
                             const float r,
                             const float T)
        : sign(sign), s0(s0), sigma(sigma), k(k), r(r), T(T) {
    }

    float calculate() const {
      const auto d1 = (std::log(s0 / k) + (r + sigma * sigma / float(2)) * T) / (sigma * std::sqrt(T));
      const auto d2 = d1 - sigma * std::sqrt(T);
      return sign * s0 * cndf(sign * d1) - sign * std::exp(-r * T) * k * cndf(sign * d2);
    }
  };

  class black_scholes_montecarlo final : public thrust::unary_function<std::size_t, float> {
    const float sign;     // +1 for call, -1 for put
    const float s0;       // initial spot price
    const float sigma;    // volatility
    const float k;        // strike
    const float r;        // interest rate
    const float T;        // term in years
    const std::size_t num_steps;

  public:
    black_scholes_montecarlo(const float sign,
                             const float s0,
                             const float sigma,
                             const float k,
                             const float r,
                             const float T,
                             const std::size_t num_steps)
        : sign(sign), s0(s0), sigma(sigma), k(k), r(r), T(T), num_steps(num_steps) {
    }

    __host__ __device__
    float operator()(const std::size_t thread_id) const {
      thrust::default_random_engine generator(0u);
      thrust::normal_distribution<float> gaussian(0, 1);
      generator.discard(num_steps * thread_id); // don't reuse subsequences
      auto s = s0;
      const auto dt = T / static_cast<float>(num_steps);
      const auto sqrt_dt = sqrtf(dt);
      for (auto i = 0ul; i < num_steps; i++) {
        s += r * s * dt + sigma * s * sqrt_dt * gaussian(generator);
      }
      return fmaxf(sign * (s - k), float(0));
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

  template<typename ExecutionPolicy, typename UnaryFunction>
  class montecarlo {
    const ExecutionPolicy& execution_policy;
    const UnaryFunction& unary_function;
    const std::size_t num_paths;

  public:
    montecarlo(const ExecutionPolicy& execution_policy,
               const UnaryFunction& unary_function,
               const std::size_t num_paths)
        : execution_policy(execution_policy)
        , unary_function(unary_function)
        , num_paths(num_paths) {
    }

    float calculate() const {
      return thrust::transform_reduce(execution_policy,
                                      thrust::counting_iterator<std::size_t>(0ul),
                                      thrust::counting_iterator<std::size_t>(num_paths),
                                      unary_function,
                                      float(),
                                      thrust::plus<float>()) / num_paths;
    }
  };

}

int main(const int argc, const char** const argv) {
  if (argc < 3) {
    std::cerr << "usage:\n\nmontecarlo <num paths> <num steps>" << std::endl;
    return -1;
  }

  const auto num_paths_long = std::atol(argv[1]);
  const auto num_steps_long = std::atol(argv[2]);

  if (num_paths_long <= 0L || num_steps_long <= 0L) {
    std::cerr << "number of paths and steps must be greater than zero" << std::endl;
    return -1;
  }

  cudaDeviceSynchronize(); // warmup device

  const auto num_paths = static_cast<std::size_t>(num_paths_long);
  const auto num_steps = static_cast<std::size_t>(num_steps_long);
  const auto s0 = float(100);
  const auto sigma = float(0.2);
  const auto k = float(110);
  const auto r = float(0.01);
  const auto t = float(0.5);

  std::array<float, 2> signs {float(-1), float(1)};

  for (const auto sign : signs) {
    std::cout << "-- sign = " << sign << std::endl;

    const monaco::black_scholes_analytical analytical(sign, s0, sigma, k, r, t);
    const monaco::black_scholes_montecarlo montecarlo(sign, s0, sigma, k, r, t, num_steps);
    const monaco::montecarlo seq(thrust::seq, montecarlo, num_paths);
    const monaco::montecarlo host(thrust::host, montecarlo, num_paths);
    const monaco::montecarlo device(thrust::device, montecarlo, num_paths);

    monaco::time_it("analytical           : ", [&analytical]() { return analytical.calculate(); });
    // monaco::time_it("monte-carlo (seq)    : ", [&seq]() { return seq.calculate(); });
    // monaco::time_it("monte-carlo (host)   : ", [&host]() { return host.calculate(); });
    monaco::time_it("monte-carlo (device) : ", [&device]()   { return device.calculate(); });
  }
}
