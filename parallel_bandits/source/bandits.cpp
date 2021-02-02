#include <cstdlib>

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
