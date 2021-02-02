#include <vector>

#include "utils.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace bandits;

TEST(MedianHeap, GIVENEmptyHeapWHENPushedOddAmountTHENMedianInTopOfMinHeap) {
    // Set Up
    vector<MedianHeap::value_type> data = {
        make_pair(17.0, 1),
        make_pair(21.0, 2),
        make_pair(6.0, 3),
        make_pair(10.5, 4), // Median
        make_pair(5.0, 5),
        make_pair(-11.0, 6),
        make_pair(100.0, 7)
    };
    MedianHeap median_heap;

    // Run
    for (auto &value : data) {
        median_heap.push(value);
    }
    auto min_heap = median_heap.get_upper_half();
    auto median = min_heap.front();

    // Test
    EXPECT_EQ(median.first, 10.5);
    EXPECT_EQ(median.second, 4);
    for (auto &value : min_heap) {
        EXPECT_GE(value.first, median.first);
    }
}

TEST(MedianHeap, GIVENEmptyHeapWHENPushedEvenAmoundTHENMedianInTopOfMinHeap) {
    // Set Up
    vector<MedianHeap::value_type> data = {
        make_pair(17.0, 1),
        make_pair(21.0, 2),
        make_pair(6.0, 3),
        make_pair(1.1, 4),
        make_pair(10.5, 5), // Median
        make_pair(5.0, 6),
        make_pair(-11.0, 7),
        make_pair(100.0, 8)
    };
    MedianHeap median_heap;

    // Run
    for (auto &value : data) {
        median_heap.push(value);
    }
    auto min_heap = median_heap.get_upper_half();
    auto median = min_heap.front();

    // Test
    EXPECT_EQ(median.first, 10.5);
    EXPECT_EQ(median.second, 5);
    for (auto &value : min_heap) {
        EXPECT_GE(value.first, median.first);
    }
}

TEST(MedianHeap, GIVENEmptyHeapWHENPushedDuplicatesTHENMedianInTopOfMinHeap) {
    // Set Up
    vector<MedianHeap::value_type> data = {
        make_pair(10.5, 1),
        make_pair(21.0, 2),
        make_pair(21.0, 3),
        make_pair(10.5, 4), // Median
        make_pair(5.0, 5),
        make_pair(5.0, 6),
        make_pair(100.0, 7)
    };
    MedianHeap median_heap;

    // Run
    for (auto &value : data) {
        median_heap.push(value);
    }
    auto min_heap = median_heap.get_upper_half();
    auto median = min_heap.front();

    // Test
    EXPECT_EQ(median.first, 10.5);
    EXPECT_EQ(median.second, 4);
    for (auto &value : min_heap) {
        EXPECT_GE(value.first, median.first);
    }
}
