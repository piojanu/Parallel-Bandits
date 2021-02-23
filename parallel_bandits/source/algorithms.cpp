#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <random>
#include <vector>

#include "algorithms.hpp"
#include "utils.hpp"

using namespace std;
using namespace bandits;

size_t
MedianElimination::solve(const vector<shared_ptr<IBanditArm>> &bandit,
                         size_t &total_pulls) const
{
    double epsilon = this->_epsilon / 4;
    double delta = this->_delta / 2;
    vector<size_t> current_arms;

    // Initialize current arms indexes to [0, arms.size()).
    for (size_t i = 0; i < bandit.size(); i++) {
        current_arms.push_back(i);
    }
    
    while (current_arms.size() > 1) {
        // TODO(pj): Don't dump pulls from the previous round.
        MedianHeap empirical_values;
        int num_pulls = ceil(1 / pow(epsilon / 2, 2) * log(3 / delta));

        total_pulls += current_arms.size() * num_pulls;
        if (total_pulls > this->_limit_pulls) {
            // Halt and return an arbitrary arm;
            return current_arms[0];
        }

        // Evaluate each arm.
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
ExpGapElimination::solve(const vector<shared_ptr<IBanditArm>> &bandit,
                         size_t &total_pulls) const
{
    int round = 1;
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

        vector<double> empirical_values;
        int num_pulls = ceil(2 / pow(epsilon, 2) * log(2 / delta));

        total_pulls += current_arms.size() * num_pulls;
        if (total_pulls > this->_limit_pulls) {
            // Halt and return an arbitrary arm;
            return current_idxs[0];
        }

        // Evaluate each arm.
        for (auto &arm : current_arms) {
            double total_return = 0;
            for (int i = 0; i < num_pulls; i++) {
                total_return += arm->pull();
            }
            empirical_values.push_back(total_return / num_pulls);
        }

        // Find (epsilon_r, delta_r)-optimal arm.
        // TODO(pj): Use pulls (empirical values) from the above eval.
        MedianElimination med_elim_algo(epsilon / 2, delta, this->_limit_pulls);
        auto best_arm_idx = med_elim_algo.solve(current_arms, total_pulls);
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

size_t
OneRoundBestArm::solve(const vector<shared_ptr<IBanditArm>> &bandit,
                       size_t &total_pulls) const
{

    vector<pair<double, size_t>> empirical_values(this->_num_players);
    #pragma omp parallel \
        num_threads(this->_num_players) \
        shared(bandit, total_pulls, empirical_values)
    {
        auto num_pulls = this->_time_horizon / 2;

        // Choose a subset of arms uniformly at random.
        default_random_engine rnd_gen;
        vector<size_t> all_idxs(bandit.size());
        iota(all_idxs.begin(), all_idxs.end(), 0);
        shuffle(all_idxs.begin(), all_idxs.end(), rnd_gen);

        size_t num_sub_arms =
            min<size_t>(ceil(6.0 * bandit.size() / sqrt(this->_num_players)),
                        bandit.size());
        vector<size_t> sub_idxs(all_idxs.begin(),
                                all_idxs.begin() + num_sub_arms);

        vector<shared_ptr<IBanditArm>> sub_arms;
        for (auto &idx : sub_idxs) {
            sub_arms.push_back(bandit[idx]);
        }

        // Explore
        size_t _total_pulls = 0;
        ExpGapElimination expgap_algo(0, 1.0 / 3.0, num_pulls);

        auto solution_idx = expgap_algo.solve(sub_arms, _total_pulls);
        auto best_arm_idx = sub_idxs[solution_idx];
        auto best_arm = sub_arms[solution_idx];

        #pragma omp atomic
        total_pulls += _total_pulls;

        // Exploit
        double total_return = 0;
        for (size_t i = 0; i < num_pulls; i++) {
            total_return += best_arm->pull();
        }

        #pragma omp atomic
        total_pulls += num_pulls;

        // Communicate the best arm idx and value.
        auto my_idx = omp_get_thread_num();
        empirical_values[my_idx] =
            make_pair(total_return / num_pulls, best_arm_idx);
    }

    vector<double> arms_total(bandit.size(), 0);
    vector<size_t> arms_count(bandit.size(), 0);
    for (auto &arm_pair : empirical_values) {
        auto arm_value = arm_pair.first;
        auto arm_idx = arm_pair.second;

        arms_total[arm_idx] += arm_value;
        arms_count[arm_idx] += 1;
    }

    auto best_arm_idx = bandit.size() - 1;
    auto best_arm_value = 0.0;
    for (size_t i = 0; i < bandit.size(); i++) {
        if (arms_count[i] > sqrt(this->_num_players)) {
            auto arm_value = arms_total[i] / arms_count[i];
            if (arm_value > best_arm_value) {
                best_arm_value = arm_value;
                best_arm_idx = i;
            }
        }
    }

    return best_arm_idx;
}

size_t
MultiRoundEpsilonArm::solve(const vector<shared_ptr<IBanditArm>> &bandit,
                       size_t &total_pulls) const
{
    int round = 1;
    double epsilon = 1, time = 0;

    vector<size_t> current_idxs(bandit.size());
    iota(current_idxs.begin(), current_idxs.end(), 0);

    // 2D vector of the shape: num. players x num. arms.
    vector<vector<double>> empirical_values(this->_num_players,
                                            vector<double>(bandit.size(), 0));
    
    while (current_idxs.size() > 1 && epsilon > (this->_epsilon / 2)) {
        auto time_old = time;
        epsilon = pow(2, -round);
        time = (2 / (this->_num_players * pow(epsilon, 2))) *
            log((4 * bandit.size() * pow(round, 2)) / this->_delta);
        auto num_pulls = ceil(time - time_old);

        total_pulls += this->_num_players * current_idxs.size() * num_pulls;
        if (total_pulls > this->_limit_pulls) {
            // Halt and return an arbitrary arm;
            return current_idxs[0];
        }

        #pragma omp parallel \
            num_threads(this->_num_players) \
            firstprivate(bandit, current_idxs, num_pulls) \
            shared(empirical_values)
        {
            auto my_idx = omp_get_thread_num();
            for (auto &arm_idx : current_idxs) {
                auto &arm = bandit[arm_idx];

                double total_return = 0;
                for (auto i = 0; i < num_pulls; i++) {
                    total_return += arm->pull();
                }

                auto average_return = total_return / num_pulls;
                empirical_values[my_idx][arm_idx] +=
                    (average_return - empirical_values[my_idx][arm_idx]) /
                    round;
            }
        }

        vector<double> average_values(bandit.size(), 0);
        for (auto &arm_idx : current_idxs) {
            for (auto p_idx = 0; p_idx < this->_num_players; p_idx++) {
                average_values[arm_idx] += empirical_values[p_idx][arm_idx];
            }
            average_values[arm_idx] /= this->_num_players;
        }
        auto best_value = *max_element(average_values.begin(),
                                       average_values.end());
        
        vector<size_t> subset_idxs;
        for (auto &arm_idx : current_idxs) {
            if (average_values[arm_idx] >= (best_value - epsilon)) {
                subset_idxs.push_back(arm_idx);
            }
        }
        
        // Bookkeeping.
        round += 1;
        swap(current_idxs, subset_idxs);
    }
    
    return current_idxs[0];
}
