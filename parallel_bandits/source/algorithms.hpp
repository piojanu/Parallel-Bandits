#pragma once
#include <vector>

#include "bandits.hpp"

using namespace std;

namespace bandits
{
    class IAlgorithm
    {
    public:
        virtual size_t
        solve(const vector<shared_ptr<IBanditArm>> &bandit) const = 0;

        virtual ~IAlgorithm() = default;
    };

    class PACAlgorithm : public IAlgorithm
    {
    public:
        PACAlgorithm() = delete;
        PACAlgorithm(double epsilon, double delta) :
            _epsilon(epsilon), _delta(delta) { };

    protected:
        const double _epsilon, _delta;
    };

    class MedianElimination : public PACAlgorithm
    {
    public:
        MedianElimination(double epsilon, double delta) :
            PACAlgorithm(epsilon, delta) { };

        size_t
        solve(const vector<shared_ptr<IBanditArm>> &bandit) const override;
    };
}
