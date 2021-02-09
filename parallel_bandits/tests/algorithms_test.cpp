#include <vector>

#include "algorithms.hpp"
#include "bandits.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace bandits;

class PACAlgorithmTest: public ::testing::Test { 
public: 
    void SetUp() { 
        vector<double> expected_values =
            {0.6, 0.7, 0.45, 0.45, 0.45, 0.45, 0.45};
        this->bandit = make_bernoulli_bandit(expected_values);
    }

    vector<shared_ptr<IBanditArm>> bandit;
};

TEST_F(PACAlgorithmTest, GIVENMedianEliminationWHENSolveMABTHENReturnBestArm) {
    // Set Up
    MedianElimination algo(0.1, 0.01, (size_t) -1);

    // Run
    auto arm = algo.solve(bandit);

    // Test
    EXPECT_EQ(arm, 1);
}

TEST_F(PACAlgorithmTest, GIVENExpGapEliminationWHENSolveMABTHENReturnBestArm) {
    // Set Up
    ExpGapElimination algo(0.1, 0.01, (size_t) -1);

    // Run
    auto arm = algo.solve(bandit);

    // Test
    EXPECT_EQ(arm, 1);
}
