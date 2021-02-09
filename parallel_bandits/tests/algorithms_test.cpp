#include <vector>

#include "algorithms.hpp"
#include "bandits.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace bandits;

class MABAlgorithmTest: public ::testing::Test {
public: 
    void SetUp() { 
        vector<double> expected_values =
            {0.6, 0.7, 0.45, 0.45, 0.45, 0.45, 0.45};
        this->bandit = make_bernoulli_bandit(expected_values);
    }

    vector<shared_ptr<IBanditArm>> bandit;
};

TEST_F(MABAlgorithmTest, GIVENMedianEliminationWHENSolveMABTHENReturnBestArm) {
    // Set Up
    MedianElimination algo(0.1, 0.01, (size_t) -1);

    // Run
    auto arm = algo.solve(bandit);

    // Test
    EXPECT_EQ(arm, 1);
}

TEST_F(MABAlgorithmTest, GIVENExpGapEliminationWHENSolveMABTHENReturnBestArm) {
    // Set Up
    ExpGapElimination algo(0.1, 0.01, (size_t) -1);

    // Run
    auto arm = algo.solve(bandit);

    // Test
    EXPECT_EQ(arm, 1);
}

TEST_F(MABAlgorithmTest, GIVENOneRoundBestArmWHENSolveMABTHENReturnBestArm) {
    // Set Up
    auto num_agents = 5;
    auto limit_pulls = 8000000;
    OneRoundBestArm algo(num_agents, limit_pulls);

    // Run
    auto arm = algo.solve(bandit);

    // Test
    EXPECT_EQ(arm, 1);
}
