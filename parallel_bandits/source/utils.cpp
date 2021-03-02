#include <algorithm>
#include <functional>

#include "utils.hpp"

using namespace std;
using namespace bandits;

void MedianHeap::push(value_type value)
{
    // Decide on which heap to put the value.
    if (this->min_heap.empty() || value.first > this->min_heap.front().first) {
        this->min_heap.push_back(value);
        push_heap(this->min_heap.begin(),
                  this->min_heap.end(),
                  greater<value_type>());
    } else {
        this->max_heap.push_back(value);
        push_heap(this->max_heap.begin(),
                  this->max_heap.end());
    }

    // Balance the heaps.
    if (this->max_heap.size() > this->min_heap.size()) {
        this->min_heap.push_back(this->max_heap.front());
        push_heap(this->min_heap.begin(),
                  this->min_heap.end(),
                  greater<value_type>());

        pop_heap(this->max_heap.begin(),
                 this->max_heap.end());
        this->max_heap.pop_back();
    } else if (this->min_heap.size() - 1 > this->max_heap.size()) {
        this->max_heap.push_back(this->min_heap.front());
        push_heap(this->max_heap.begin(),
                  this->max_heap.end());

        pop_heap(this->min_heap.begin(),
                 this->min_heap.end(),
                 greater<value_type>());
        this->min_heap.pop_back();
    }
}
