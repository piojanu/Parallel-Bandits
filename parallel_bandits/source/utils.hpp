#pragma once
#include <vector>

using namespace std;

namespace bandits
{
    class MedianHeap {
    public:
        typedef pair<double, size_t> value_type;

        void push(value_type value);
        const auto& get_upper_half() const {return min_heap;};

    private:
        vector<value_type> max_heap;
        vector<value_type> min_heap;
    };
};
