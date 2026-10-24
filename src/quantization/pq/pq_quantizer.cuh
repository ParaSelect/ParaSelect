#ifndef IVF_PQ_QUANTIZER_CUH
#define IVF_PQ_QUANTIZER_CUH

#include "quantization/quantizer.cuh"
#include <cstddef>
#include <vector>

namespace ivf {

/**
 * PQ (CPU): KMeans  + /,host , linear_fp16.
 */
class PQQuantizerCPU : public QuantizerInterfaceCPU {
public:
    /**
     * @param n_subspace  M,dim  M
     * @param n_codes  K( 256), 1
     */
    PQQuantizerCPU(int n_subspace, int n_codes);
    void train(const float* h_data, int n, int dim) override;
    void encode(const float* h_data, int n, int dim, void* h_codes) override;
    void decode(const void* h_codes, int n, int dim, float* h_data) override;
    int code_size_bytes(int n, int dim) const override;

    int get_n_subspace() const { return n_subspace_; }
    int get_n_codes() const { return n_codes_; }
    /**  dim/M, 0 */
    int get_sub_dim() const { return sub_dim_; }
    /** :[M, K, d_sub] */
    const std::vector<float>& get_codebooks() const { return codebooks_; }

private:
    int n_subspace_;
    int n_codes_;
    int sub_dim_;
    /** :M , n_codes * sub_dim, [m][k][d]  */
    std::vector<float> codebooks_;
};

/**
 * PQ (GPU): device,encode/decode  device , linear_fp16.
 */
class PQQuantizerGPU : public QuantizerInterface {
public:
    PQQuantizerGPU(int n_subspace, int n_codes);
    ~PQQuantizerGPU() override;
    void train(const float* h_data, int n, int dim) override;
    void encode(const float* d_data, int n, int dim, void* d_codes) override;
    void decode(const void* d_codes, int n, int dim, float* d_data) override;
    int code_size_bytes(int n, int dim) const override;

    int get_n_subspace() const { return n_subspace_; }
    int get_n_codes() const { return n_codes_; }
    int get_sub_dim() const { return sub_dim_; }

private:
    int n_subspace_;
    int n_codes_;
    int sub_dim_;
    float* d_codebooks_;
};

} // namespace ivf

#endif
