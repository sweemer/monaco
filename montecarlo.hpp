#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>

namespace monaco {

  using point = std::pair<float, float>; // date in years, spot value
  using path = std::vector<monaco::point>;

  struct path_generator {
    virtual const monaco::path generate() const = 0;
    virtual ~path_generator() = default;
  };

  struct path_evaluator {
    virtual float evaluate(const monaco::path& path) const = 0;
    virtual ~path_evaluator() = default;
  };

  template<typename ExecutionPolicy>
  class montecarlo {
    const ExecutionPolicy execution_policy;
    const std::size_t num_paths;
    const std::shared_ptr<const monaco::path_generator> path_generator;
    const std::shared_ptr<const monaco::path_evaluator> path_evaluator;

  public:
    montecarlo(const ExecutionPolicy execution_policy,
               const std::size_t num_paths,
               std::shared_ptr<const monaco::path_generator> path_generator,
               std::shared_ptr<const monaco::path_evaluator> path_evaluator)
        : num_paths(num_paths)
        , path_generator(std::move(path_generator))
        , path_evaluator(std::move(path_evaluator)) {
    }

    float calculate() {
      std::vector<int> path_ids(num_paths);
      const auto sum = std::transform_reduce(execution_policy, path_ids.begin(), path_ids.end(), float(), std::plus<float>(), [this](const auto path_id) {
        const auto path = path_generator->generate();
        return path_evaluator->evaluate(path);
      });
      return sum / num_paths;
    }
  };

}
