#include "search/ivf_common/quantized/res_query_lut_bias.h"
#include "dataset/cache/res_row_correction_assets.h"
#include <cmath>
#include <fstream>
#include <unistd.h>

void require(bool condition) {
    if (!condition) throw std::runtime_error("Res ROW_CORRECTION test failed");
}

int main() {
    const paraselect::search::QueryLutBias absent{};
    require(absent.empty());
    require(!absent.enabled());

    const float c[] = {2.0f, -1.0f, 3.0f};
    const float q[] = {1.0f, 4.0f, -2.0f};
    // Test both a dense reconstruction and a scaled sign reconstruction.
    const float reconstructions[][3] = {{0.25f, -2.0f, 1.0f}, {0.5f, -0.5f, 0.5f}};
    for (const auto& r : reconstructions) {
        float raw = 0, coarse = 0, dot = 0;
        for (int d = 0; d < 3; ++d) {
            raw += (q[d] - c[d] - r[d]) * (q[d] - c[d] - r[d]);
            coarse += (q[d] - c[d]) * (q[d] - c[d]);
            dot += q[d] * r[d];
        }
        const float row_correction = paraselect::search::query_lut_bias_value(c, r, 3);
        require(std::abs(raw - (coarse + row_correction - 2 * dot)) < 1e-5f);
        const paraselect::search::QueryLutBias view{&row_correction, nullptr, 0, 0, 1};
        require(!view.empty());
        require(view.enabled());
        require(view.count == 1);
    }
    char path[] = "/tmp/paraselect-res-row_correction-XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0);
    close(fd);
    try {
        const float raw[] = {1, 2, 3};
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(raw), sizeof(raw));
        }
        auto fp = paraselect::dataset::RowCorrectionAsset::open(path, 3);
        require(fp.view().fp32[2] == 3 && !fp.view().u16);
        fp = {};
        paraselect::row_correction::U16Header header{};
        std::memcpy(header.magic, paraselect::row_correction::kU16Magic, 8);
        header.version = paraselect::row_correction::kU16Version; header.header_bytes = sizeof(header); header.count = 3;
        header.scale = 0.5f; header.offset = -1;
        const std::uint16_t codes[] = {0, 2, 4};
        for (bool valid : {true, false}) {
            header.scale = valid ? 0.5f : 0;
            {
                std::ofstream out(path, std::ios::binary | std::ios::trunc);
                out.write(reinterpret_cast<const char*>(&header), sizeof(header));
                out.write(reinterpret_cast<const char*>(codes), sizeof(codes));
            }
            bool rejected = false;
            try {
                auto row_correction_u16 = paraselect::dataset::RowCorrectionAsset::open(path, 3);
                require(row_correction_u16.view().u16[2] == 4);
                require(row_correction_u16.view().scale == 0.5f);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected == !valid);
        }
        bool rejected = false;
        try { paraselect::dataset::RowCorrectionAsset::open(path, 4); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    } catch (...) {
        unlink(path);
        throw;
    }
    unlink(path);
}
