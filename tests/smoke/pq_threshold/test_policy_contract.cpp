#include "fine_search_strategy.h"

#include <cassert>
#include <cmath>
#include <string>

int main() {
    using namespace paraselect::search;
    static_assert(static_cast<int>(PQThresholdPolicy::Bucket) >= 0);

    PQAdcThresholdParams p;
    p.policy = PQThresholdPolicy::Bucket;
    assert(p.candidate_capacity >= p.candidate_k);
    assert(p.candidate_k == 512);
    assert(p.candidate_capacity <= 8192);
    assert(p.bucket_edge0 < p.bucket_edge1);
    assert(p.bucket_alpha0 >= 1.0f);
    assert(p.bucket_alpha1 >= p.bucket_alpha0);
    assert(p.bucket_alpha2 >= p.bucket_alpha1);

    return 0;
}
