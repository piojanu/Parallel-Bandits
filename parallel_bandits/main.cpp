#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <omp.h>

#include "algorithms.hpp"
#include "bandits.hpp"

using namespace std;
using namespace bandits;
using namespace chrono;

struct Result {
    milliseconds elapsed;
    size_t total_pulls;
    bool is_solved;
};

Result measure_expgap(int num_arms, double min_gap,
                      double epsilon, double delta)
{
    auto bandit = make_bernoulli_bandit(num_arms, min_gap);
    ExpGapElimination expgap_algo(epsilon, delta, (size_t) -1);
    size_t total_pulls = 0;

    auto begin = steady_clock::now();
    auto solution_arm = expgap_algo.solve(bandit, total_pulls);
    auto end = steady_clock::now();

    Result result = {
        duration_cast<milliseconds>(end - begin),
        total_pulls,
        solution_arm == (num_arms - 1)
    };

    return result;
}

Result measure_multiround(int num_arms, double min_gap, int num_threads,
                          double epsilon, double delta)
{
    auto bandit = make_bernoulli_bandit(num_arms, min_gap);
    MultiRoundEpsilonArm multiround_algo(num_threads, epsilon, delta,
                                         (size_t) -1);
    size_t total_pulls = 0;

    auto begin = steady_clock::now();
    auto solution_arm = multiround_algo.solve(bandit, total_pulls);
    auto end = steady_clock::now();

    Result result = {
        duration_cast<milliseconds>(end - begin),
        total_pulls,
        solution_arm == (num_arms - 1)
    };

    return result;
}

int main(int argc, char **argv) {
    // Seed the random generator.
    srand(static_cast<unsigned int>(time(NULL)));

    #pragma omp parallel num_threads(10)
    {
        auto my_idx = omp_get_thread_num();
        #pragma omp critical
        {
            cout << "Hello World from thread: ";
            cout << omp_get_thread_num();
            cout << "!" << endl;
        }
    }
    cout << endl;

    vector<int> num_arms_params = {100, 1000, 10000, 100000, 1000000};
    vector<double> min_gap_params = {0.4, 0.2, 0.1, 0.01, 0.001};
    vector<int> num_threads_params = {1, 8, 16, 32, 64, 128};
    vector<double> epsilon_params = {0.2, 0.1, 0.01, 0.001};
    vector<double> delta_params = {0.1, 0.05, 0.01};
    const auto total_runs = (num_arms_params.size() *
                             min_gap_params.size() *
                             num_threads_params.size() *
                             epsilon_params.size() *
                             delta_params.size());
    int run_count = 0;

    ofstream expgap_results;
    expgap_results.open("expgap_results.csv", fstream::out);
    expgap_results << "num_arms,min_gap,num_threads,epsilon,delta"
                   << ",elapsed,pulls,solved" << endl;

    ofstream multiround_results;
    multiround_results.open("multiround_results.csv", fstream::out);
    multiround_results << "num_arms,min_gap,num_threads,epsilon,delta"
                       << ",elapsed,pulls,solved" << endl;

    auto begin = steady_clock::now();
    for (auto &num_arms : num_arms_params) {
    for (auto &min_gap : min_gap_params) {
    for (auto &epsilon : epsilon_params) {
    for (auto &delta : delta_params) {
        auto result = measure_expgap(num_arms, min_gap, epsilon, delta);
        expgap_results << num_arms << ","
                       << min_gap << ","
                       << 1 << "," // Num. threads
                       << epsilon << ","
                       << delta << ","
                       << result.elapsed.count() << ","
                       << result.total_pulls << ","
                       << result.is_solved << endl;

        for (auto &num_threads : num_threads_params) {
            auto result = measure_multiround(num_arms, min_gap, num_threads,
                                             epsilon, delta);
            multiround_results << num_arms << ","
                               << min_gap << ","
                               << num_threads << ","
                               << epsilon << ","
                               << delta << ","
                               << result.elapsed.count() << ","
                               << result.total_pulls << ","
                               << result.is_solved << endl;
        }

        expgap_results.flush();
        multiround_results.flush();

        auto total_time = duration_cast<seconds>(steady_clock::now() - begin);
        run_count++;
        cout << "Elapsed: " << total_time.count() << "s | "
             << "Run: " << run_count << "/" << total_runs << endl;
    }}}}
    
    expgap_results.close();
    multiround_results.close();
    return 0;
}
