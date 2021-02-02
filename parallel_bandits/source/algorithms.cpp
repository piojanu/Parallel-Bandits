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
