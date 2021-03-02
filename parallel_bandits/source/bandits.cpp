#include <cstdlib>
#include <memory>

#include "bandits.hpp"

using namespace std;
using namespace bandits;

double BernoulliArm::pull() const
{
    double r = (double) rand() / (double) RAND_MAX;
    return (double) (r >= (1 - this->_value));
}

vector<shared_ptr<IBanditArm>>
bandits::make_bernoulli_bandit(const vector<double> &expected_values)
{
    vector<shared_ptr<IBanditArm>> bandit;
    for (auto &value : expected_values) {
        auto arm = make_shared<BernoulliArm>(value);
        bandit.push_back(arm);
    }

    return bandit;
}

vector<shared_ptr<IBanditArm>>
bandits::make_bernoulli_bandit(const int num_arms, const double min_gap)
{
    double optimal_value = 1 - ((1 - min_gap) / 2);
    double others_value = (1 - min_gap) / 2;

    vector<shared_ptr<IBanditArm>> bandit;
    for (auto i = 0; i < (num_arms - 1); i++) {
        auto arm = make_shared<BernoulliArm>(others_value);
        bandit.push_back(arm);
    }
    auto arm = make_shared<BernoulliArm>(optimal_value);
    bandit.push_back(arm);

    return bandit;
}
