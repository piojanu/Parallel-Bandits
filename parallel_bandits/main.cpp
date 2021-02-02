#include <cstdlib>
#include <iostream>
#include <omp.h>

#include "algorithms.hpp"
#include "bandits.hpp"

using namespace std;
using namespace bandits;

int main(int argc, char **argv) {
    // Seed the random generator.
    srand(static_cast<unsigned int>(time(NULL)));

    #pragma omp parallel
    {
        #pragma omp critical
        {
            cout << "Hello World from thread: ";
            cout << omp_get_thread_num();
            cout << "!" << endl;
        }
    }

    vector<double> expected_values = {0.5, 0.75, 0.1, 0.1, 0.45, 0.45, 0.45};
    auto bandit = make_bernoulli_bandit(expected_values);
    MedianElimination algo(0.1, 0.05);
    auto best_arm = algo.solve(bandit);

    cout << "Bandit arms: ";
    for (auto &value : expected_values) {
        cout << value << ", ";
    }
    cout << endl << "Best arm idx: " << best_arm << endl;
    
    return 0;
}
