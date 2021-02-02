#include <cmath>
#include <vector>

#include "algorithms.hpp"
#include "utils.hpp"

using namespace std;
using namespace bandits;

size_t
MedianElimination::solve(const vector<shared_ptr<IBanditArm>> &bandit) const
{
    double epsilon = this->_epsilon / 4;
    double delta = this->_delta / 2;
    vector<size_t> current_arms;

    // Initialize current arms indexes to [0, arms.size()).
    for (size_t i = 0; i < bandit.size(); i++) {
        current_arms.push_back(i);
    }
    
    while (current_arms.size() > 1) {
        // Evaluate each arm.
        // TODO(pj): Don't dump pulls from the previous round.
        MedianHeap empirical_values;
        int num_pulls = ceil(1 / pow(epsilon / 2, 2) * log(3 / delta));
        for (auto &idx : current_arms) {
            auto &arm = bandit[idx];
            double total_return = 0;
            for (int i = 0; i < num_pulls; i++) {
                total_return += arm->pull();
            }
            empirical_values.push(make_pair(total_return / num_pulls, idx));
        }

        // Pick arms above the median empirical value.
        vector<size_t> subset_arms;
        for (auto &value_arm_pair : empirical_values.get_upper_half()) {
            subset_arms.push_back(value_arm_pair.second);
        }

        // Bookkeeping.
        epsilon = 0.75 * epsilon;
        delta = delta / 2.0;
        swap(current_arms, subset_arms);
    }

    return current_arms[0];
}

size_t
ExpGapElimination::solve(const vector<shared_ptr<IBanditArm>> &bandit) const
{
    int round = 0;
    vector<shared_ptr<IBanditArm>> current_arms;
    vector<size_t> current_idxs;

    // Initialize current arms.
    for (size_t i = 0; i < bandit.size(); i++) {
        current_arms.push_back(bandit[i]);
        current_idxs.push_back(i);
    }

    while (current_arms.size() > 1 &&
           (this->_epsilon == 0 || round < ceil(log2(1 / this->_epsilon)))) {
        double epsilon = pow(2, -round) / 4;
        double delta = this->_delta / (50.0 * pow(round, 3));

        // Evaluate each arm.
        vector<double> empirical_values;
        int num_pulls = ceil(2 / pow(epsilon, 2) * log(2 / delta));
        for (auto &arm : current_arms) {
            double total_return = 0;
            for (int i = 0; i < num_pulls; i++) {
                total_return += arm->pull();
            }
            empirical_values.push_back(total_return / num_pulls);
        }

        // Find (epsilon_r, delta_r)-optimal arm.
        // TODO(pj): Use pulls (empirical values) from the above eval.
        MedianElimination med_elim_algo(epsilon / 2, delta);
        auto best_arm_idx = med_elim_algo.solve(current_arms);
        auto best_value = empirical_values[best_arm_idx];

        // Pick arms above the epsilon-best value.
        vector<shared_ptr<IBanditArm>> subset_arms;
        vector<size_t> subset_idxs;
        for (size_t i = 0; i < empirical_values.size(); i++) {
            if (empirical_values[i] >= best_value - epsilon) {
                subset_arms.push_back(current_arms[i]);
                subset_idxs.push_back(current_idxs[i]);
            }
        }

        // Bookkeeping.
        round += 1;
        swap(current_arms, subset_arms);
        swap(current_idxs, subset_idxs);
    }

    return current_idxs[0];
}
