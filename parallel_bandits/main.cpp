#include <cstdlib>
#include <chrono>
#include <iostream>
#include <omp.h>

#include "algorithms.hpp"
#include "bandits.hpp"

using namespace std;
using namespace bandits;
using namespace chrono;

int main(int argc, char **argv) {
    // Seed the random generator.
    srand(static_cast<unsigned int>(time(NULL)));

    int num_agents = 10;
    vector<pair<double, size_t>> pairs_vector(num_agents);
    #pragma omp parallel num_threads(num_agents) shared(pairs_vector)
    {
        auto my_idx = omp_get_thread_num();
        pairs_vector[my_idx] = make_pair(my_idx, my_idx);

        #pragma omp critical
        {
            cout << "Hello World from thread: ";
            cout << omp_get_thread_num();
            cout << "!" << endl;
        }
    }

    cout << endl << "Pairs values: ";
    for (auto &value : pairs_vector) {
        cout << value.first << ", " << value.second << "; ";
    }
    cout << endl;


    vector<double> expected_values = {0.5, 0.75, 0.1, 0.1, 0.45, 0.45, 0.45};
    auto bandit = make_bernoulli_bandit(expected_values);

    MedianElimination median_algo(0.01, 0.01, (size_t) -1);
    size_t median_total_pulls = 0;
    auto begin = steady_clock::now();
    auto median_arm = median_algo.solve(bandit, median_total_pulls);
    auto end = steady_clock::now();
    auto median_epalsed = duration_cast<milliseconds>(end - begin).count();

    ExpGapElimination expgap_algo(0.01, 0.01, (size_t) -1);
    size_t expgap_total_pulls = 0;
    begin = steady_clock::now();
    auto expgap_arm = expgap_algo.solve(bandit, expgap_total_pulls);
    end = steady_clock::now();
    auto expgap_epalsed = duration_cast<milliseconds>(end - begin).count();

    cout << endl << "Bandit arms: ";
    for (auto &value : expected_values) {
        cout << value << ", ";
    }
    cout << endl << endl << "Params: epsilon = 0.01, delta = 0.01." << endl;
    cout << "Median Elimination result arm " << median_arm;
    cout << " in " << median_epalsed << " [ms]";
    cout << ", " << median_total_pulls << " [pulls]" << endl;
    cout << "ExpGap Elimination result arm " << expgap_arm;
    cout << " in " << expgap_epalsed << " [ms]";
    cout << ", " << expgap_total_pulls << " [pulls]" << endl;
    
    return 0;
}
