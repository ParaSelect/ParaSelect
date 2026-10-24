#include "quantization/row_correction/u16.h"
#include "quantization/row_correction/pq.h"
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>

void require(bool ok) { if (!ok) throw std::runtime_error("U16 row correction contract failed"); }

int main() {
    using namespace paraselect::row_correction;
    const float centers[] = {3, -5, -11, 7};
    const float words[] = {-10, 2, 19, -13, 4, 25};
    const auto codec = U16Quantizer::for_pq(centers, 2, words, 2, 3, 1);
    // Exhaust every possible future code combination, not just initial rows.
    for (int c = 0; c < 2; ++c)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                float reconstruction[] = {words[a], words[3+b]};
                float v = value(centers + c*2, reconstruction, 2);
                const std::uint8_t pq_code[] = {std::uint8_t(a), std::uint8_t(b)};
                require(pq_value(centers+c*2, words, pq_code, 2, 3, 1) == v);
                auto code = codec.encode(v);
                require(code <= 65534);
                require(std::abs(codec.decode(code) - v) <= codec.scale * 1.01f);
            }
    for (float v : {std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::max()}) {
        bool rejected = false;
        try { (void)codec.encode(v); } catch (const std::exception&) { rejected = true; }
        require(rejected);
    }
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> distribution(-1000, 1000);
    for (int dim : {16, 128, 160}) {
        const int m = 16, k = 8, dsub = dim/m;
        std::vector<float> centroids(3*dim), book(m*k*dsub), reconstruction(dim);
        for (auto& v : centroids) v = distribution(rng);
        for (auto& v : book) v = distribution(rng);
        auto quantizer = U16Quantizer::for_pq(centroids.data(), 3, book.data(), m, k, dsub);
        for (int row = 0; row < 10000; ++row) {
            for (int sub = 0; sub < m; ++sub) {
                int code = rng()%k;
                for (int j = 0; j < dsub; ++j)
                    reconstruction[sub*dsub+j] = book[(sub*k+code)*dsub+j];
            }
            float v = value(centroids.data()+(row%3)*dim, reconstruction.data(), dim);
            auto code = quantizer.encode(v);
            require(std::abs(quantizer.decode(code)-v) <= quantizer.scale*1.01f);
        }
    }
    for (float invalid : {std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::max()}) {
        bool rejected = false;
        try { (void)U16Quantizer::for_pq(&invalid, 1, &invalid, 1, 1, 1); }
        catch (const std::exception&) { rejected = true; }
        require(rejected);
    }
    const float zero[] = {0};
    auto constant = U16Quantizer::for_pq(zero, 1, zero, 1, 1, 1);
    require(std::isfinite(constant.decode(constant.encode(0))));
}
