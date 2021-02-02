#pragma once
#include <vector>

#include "bandits.hpp"

using namespace std;

namespace bandits
{
    class IAlgorithm
    {
    public:
        /**
         * Solve the Multi-Armed Bandit problem.
         *
         * @param bandit Vector of bandit arms to pull.
         * @return The (ε-)optimal arm index.
         */
        virtual size_t
        solve(const vector<shared_ptr<IBanditArm>> &bandit) const = 0;

        /**
         * Solve the Multi-Armed Bandit problem.
         *
         * @param[in] bandit Vector of bandit arms to pull.
         * @param[in, out] total_pulls Total number of arm pulls to this moment.
         * @return The (ε-)optimal arm index.
         */
        virtual size_t
        solve(const vector<shared_ptr<IBanditArm>> &bandit,
              size_t &total_pulls) const = 0;

        virtual ~IAlgorithm() = default;
    };

    class PACAlgorithm : public IAlgorithm
    {
    public:
        /**
         * Initialize the (ε, δ)-PAC solver.
         *
         * @param epsilon Find an arm that is at most ε worse than the optimal
         *     arm in terms of the expected value (bounded between [0, 1]).
         * @param delta With probability of at least 1-δ find an ε-optimal arm.
         * @param limit_pulls Don't pull all arms more then this amount.
         *     If the limit is exceeded, then `solve` returns an arbitrary arm!
         */
        PACAlgorithm(double epsilon, double delta, size_t limit_pulls) :
            _epsilon(epsilon), _delta(delta), _limit_pulls(limit_pulls) { }

        PACAlgorithm() = delete;

        // Note: It is required because `solve` in this class hides the other
        //       `solve` in the IAlgorithm interface.
        // Source: https://stackoverflow.com/a/1896864/7983111
        using IAlgorithm::solve;

        size_t
        solve(const vector<shared_ptr<IBanditArm>> &bandit) const override
        {
            size_t total_pulls = 0;
            return this->solve(bandit, total_pulls);
        }

    protected:
        const double _epsilon, _delta;
        const size_t _limit_pulls;
    };

    class MedianElimination : public PACAlgorithm
    {
    public:
        MedianElimination(double epsilon, double delta, size_t limit_pulls) :
            PACAlgorithm(epsilon, delta, limit_pulls) { }

        using PACAlgorithm::solve; // Use the base class implementation;

        size_t
        solve(const vector<shared_ptr<IBanditArm>> &bandit,
              size_t &total_pulls) const override;
    };

    class ExpGapElimination : public PACAlgorithm
    {
    public:
        ExpGapElimination(double epsilon, double delta, size_t limit_pulls) :
            PACAlgorithm(epsilon, delta, limit_pulls) { }
        
        using PACAlgorithm::solve; // Use the base class implementation;

        size_t
        solve(const vector<shared_ptr<IBanditArm>> &bandit,
              size_t &total_pulls) const override;
    };
}
