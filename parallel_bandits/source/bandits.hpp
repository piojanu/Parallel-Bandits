#pragma once
#include <vector>

using namespace std;

namespace bandits
{
    class IBanditArm
    {
    public:
        virtual double pull() const = 0;
        virtual ~IBanditArm() = default;
    };
    
    class BernoulliArm : public IBanditArm
    {
    public:
        BernoulliArm() = delete;
        BernoulliArm(double expected_value) : _value(expected_value) { };
        double pull() const override;

    private:
        const double _value;
    };

    vector<shared_ptr<IBanditArm>>
    make_bernoulli_bandit(const vector<double> &expected_values);

    vector<shared_ptr<IBanditArm>>
    make_bernoulli_bandit(const int num_arms, const double min_gap);
}
