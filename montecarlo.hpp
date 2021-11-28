#pragma once

#include <algorithm>
#include <numeric>
#include <tuple>

namespace monaco {

  using point = std::pair<float, float>; // date in years, spot value
  using path = std::vector<monaco::point>;

  template<typename ExecutionPolicy, typename PathGenerator, typename PathEvaluator>
  class montecarlo {
    const ExecutionPolicy execution_policy;
    const PathGenerator& path_generator;
    const PathEvaluator& path_evaluator;
    const std::size_t num_paths;

  public:
    montecarlo(const ExecutionPolicy execution_policy,
               const PathGenerator& path_generator,
               const PathEvaluator& path_evaluator,
               const std::size_t num_paths)
        : execution_policy(execution_policy)
        , path_generator(path_generator)
        , path_evaluator(path_evaluator)
        , num_paths(num_paths) {
    }

    float calculate() const {
      std::vector<int> path_ids(num_paths);
      const auto sum = std::transform_reduce(execution_policy, path_ids.begin(), path_ids.end(), float(), std::plus<float>(), [this](const auto path_id) {
        const auto path = path_generator.generate();
        return path_evaluator.evaluate(path);
      });
      return sum / num_paths;
    }
  };

}
