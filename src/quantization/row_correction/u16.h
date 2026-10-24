#pragma once

#include "common/row_correction.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace paraselect::row_correction {

// A fixed-model affine encoding. Page storage transports the codes; only the
// quantization layer chooses their range. Changing the model invalidates it.
struct U16Quantizer {
    float scale = 0;
    float offset = 0;

    std::uint16_t encode(float v) const {
        if (!std::isfinite(v) || !(scale > 0) || !std::isfinite(scale) || !std::isfinite(offset))
            throw std::invalid_argument("row_correction: non-finite value or invalid U16 scale");
        const double code = std::ceil((double(v) - double(offset)) / double(scale));
        if (v < offset || !(code >= 0 && code <= 65534))
            throw std::out_of_range("row_correction: value outside fixed model U16 range");
        return static_cast<std::uint16_t>(code);
    }

    float decode(std::uint16_t code) const { return std::fma(scale, float(code), offset); }

    static U16Quantizer for_pq(const float* centers, int nlist,
                              const float* codebook, int m, int k, int dsub) {
        if (!centers || !codebook || nlist <= 0 || m <= 0 || k <= 0 || k > 256 || dsub <= 0)
            throw std::invalid_argument("row_correction: invalid PQ model");
        const std::int64_t dim = std::int64_t(m) * dsub;
        // Bound sequential FP32 products, norm/dot sums and final addition.
        // epsilon (rather than epsilon/2) also leaves headroom for the host
        // bound calculation. The absolute term handles subnormal underflow.
        const long double operations = 2.L * dim + 8;
        const long double error = operations * std::numeric_limits<float>::epsilon();
        if (error >= 0.25L)
            throw std::invalid_argument("row_correction: PQ dimension exceeds error-bound contract");
        const long double gamma = error / (1 - error);
        long double lower = std::numeric_limits<long double>::infinity();
        long double upper = -lower;
        for (int c = 0; c < nlist; ++c) {
            long double lo = 0, hi = 0, magnitude = 0;
            for (int sub = 0; sub < m; ++sub) {
                long double sub_lo = std::numeric_limits<long double>::infinity();
                long double sub_hi = -sub_lo, sub_abs = 0;
                for (int code = 0; code < k; ++code) {
                    long double score = 0, absolute = 0;
                    for (int j = 0; j < dsub; ++j) {
                        const long double p = codebook[(std::size_t(sub)*k + code)*dsub+j];
                        const long double centroid = centers[std::size_t(c)*dim + sub*dsub+j];
                        if (!std::isfinite(p) || !std::isfinite(centroid))
                            throw std::invalid_argument("row_correction: non-finite PQ model");
                        score += p*p + 2*centroid*p;
                        absolute += p*p + 2*std::abs(centroid*p);
                    }
                    sub_lo = std::min(sub_lo, score);
                    sub_hi = std::max(sub_hi, score);
                    sub_abs = std::max(sub_abs, absolute);
                }
                lo += sub_lo;
                hi += sub_hi;
                magnitude += sub_abs;
            }
            // Reject models whose original FP32 producer could overflow,
            // even when cancellation would make the final value finite.
            if (magnitude * (1 + gamma) >= std::numeric_limits<float>::max())
                throw std::overflow_error("row_correction: model can overflow FP32 producer");
            const long double padding = gamma*magnitude +
                operations*std::numeric_limits<float>::min();
            lower = std::min(lower, lo-padding);
            upper = std::max(upper, hi+padding);
        }
        const float inf = std::numeric_limits<float>::infinity();
        U16Quantizer out;
        out.offset = std::nextafter(static_cast<float>(lower), -inf);
        const float high = std::nextafter(static_cast<float>(upper), inf);
        out.scale = std::nextafter(static_cast<float>((double(high)-out.offset)/65534.), inf);
        if (!std::isfinite(out.offset) || !std::isfinite(high) ||
            !std::isfinite(out.scale) || !(out.scale > 0))
            throw std::overflow_error("row_correction: model has no finite U16 encoding");
        if (!std::isfinite(out.decode(65534)))
            throw std::overflow_error("row_correction: U16 endpoint overflows FP32 decode");
        (void)out.encode(high);
        return out;
    }
};

} // namespace paraselect::row_correction
