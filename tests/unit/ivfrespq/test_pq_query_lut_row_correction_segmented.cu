#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_seed.cuh"
#include "search/ivf_common/ivf_probe_schedule.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut.cuh"
#include "search/kernels/fine/gpu_fine/pq_adc/pq_query_lut_scan.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int kQueries = 3;
constexpr int kLists = 3;
constexpr int kProbes = 2;
constexpr int kPqM = 16;
constexpr int kGlobalLutStride = paraselect::search::kPqQueryLutGlobalStride;
constexpr int kPoolCapacity = 2048;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        check_cuda(cudaMalloc(&data_, count * sizeof(T)), "allocate test buffer");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

struct Candidate {
    int query;
    int row;
    uint32_t distance_bits;

    auto as_tuple() const { return std::tie(query, row, distance_bits); }
};

std::vector<Candidate> copy_candidates(
    const DeviceBuffer<int>& rows,
    const DeviceBuffer<float>& distances,
    const DeviceBuffer<int>& queries,
    int count) {
    std::vector<int> host_rows(count);
    std::vector<float> host_distances(count);
    std::vector<int> host_queries(count);
    check_cuda(cudaMemcpy(host_rows.data(), rows.get(), count * sizeof(int),
                          cudaMemcpyDeviceToHost), "copy candidate rows");
    check_cuda(cudaMemcpy(host_distances.data(), distances.get(), count * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy candidate distances");
    check_cuda(cudaMemcpy(host_queries.data(), queries.get(), count * sizeof(int),
                          cudaMemcpyDeviceToHost), "copy candidate queries");
    std::vector<Candidate> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i) {
        uint32_t distance_bits = 0;
        std::memcpy(&distance_bits, &host_distances[i], sizeof(distance_bits));
        result.push_back({host_queries[i], host_rows[i], distance_bits});
    }
    std::sort(result.begin(), result.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.as_tuple() < rhs.as_tuple();
    });
    return result;
}

void test_vector_scale_dispatch(int nq, int np, int rows, int probe_begin = 0) {
    constexpr int m=16;
    const int cap=nq*np*rows;
    DeviceBuffer<int> counts(1), query_counts(nq), total(1), status(1), ids(cap), queries(cap);
    DeviceBuffer<long long> offsets(1);
    DeviceBuffer<idx_t> probes(nq*np);
    DeviceBuffer<uint8_t> codes(rows*m), lut(nq*m*256);
    DeviceBuffer<uint16_t> row_correction(rows);
    DeviceBuffer<float> scales(nq*m), unaligned(nq*m+1), lo(nq), coarse(nq*np), thresholds(nq), distances(cap);
    auto upload=[](auto* dst,const auto& src) {
        check_cuda(cudaMemcpy(dst,src.data(),src.size()*sizeof(src[0]),cudaMemcpyHostToDevice),"vector-scale upload");
    };
    upload(counts.get(),std::vector<int>{rows}); upload(offsets.get(),std::vector<long long>{0});
    upload(probes.get(),std::vector<idx_t>(nq*np,0));
    std::vector<uint8_t> hcodes(rows*m), hlut(nq*m*256);
    for(size_t i=0;i<hcodes.size();++i)hcodes[i]=(i*71+19)&255;
    for(size_t i=0;i<hlut.size();++i)hlut[i]=(i*37+i/256)&255;
    std::vector<float> hs(nq*m);for(size_t i=0;i<hs.size();++i)hs[i]=0.00317f*float(1+i%123);
    upload(codes.get(),hcodes);upload(lut.get(),hlut);upload(scales.get(),hs);upload(unaligned.get()+1,hs);
    upload(row_correction.get(),std::vector<uint16_t>(rows,12345));upload(lo.get(),std::vector<float>(nq,-17.31f));
    upload(coarse.get(),std::vector<float>(nq*np,513.97f));upload(thresholds.get(),std::vector<float>(nq,1.e8f));
    paraselect::search::IVFProbeScheduleBuilder builder(1,nq*np,counts.get(),paraselect::search::IVFProbeScheduleStrategy::Dense);
    const auto schedule=builder.build({probes.get(),nq,np},np,nullptr);
    paraselect::search::PQQueryLutView view;
    view.query_codeword_dot_lut_u8=lut.get();view.query_lut_lo_sum=lo.get();view.query_centroid_distances=coarse.get();
    view.query_lut_bias.u16=row_correction.get();view.query_lut_bias.scale=0.731f;view.query_lut_bias.offset=-371.97f;
    const paraselect::search::PQAdcThresholdCandidatePoolView output{
        cap,ids.get(),distances.get(),queries.get(),query_counts.get(),total.get(),status.get(),nullptr};
    std::vector<Candidate> reference;
    for(bool aligned:{false,true}) {
        view.query_lut_scale=aligned?scales.get():unaligned.get()+1;
        check_cuda(cudaMemset(query_counts.get(),0,nq*sizeof(int)),"clear counts");
        check_cuda(cudaMemset(total.get(),0,sizeof(int)),"clear total");
        check_cuda(cudaMemset(status.get(),0,sizeof(int)),"clear status");
        paraselect::search::pq_adc_query_lut_segmented_scan(view, schedule.view,nullptr,codes.get(),
            offsets.get(),counts.get(),thresholds.get(),nq,np,m,paraselect::search::PQAdcListScanRowMode::Serial,
            paraselect::search::PQAdcLutReadMode::Global,nullptr,output,nullptr,
            paraselect::search::PQAdcSharedLutLayout::Compact,probe_begin,np);
        int n=0,s=0;check_cuda(cudaMemcpy(&n,total.get(),4,cudaMemcpyDeviceToHost),"read total");
        check_cuda(cudaMemcpy(&s,status.get(),4,cudaMemcpyDeviceToHost),"read status");
        if(s||n!=nq*(np-probe_begin)*rows)throw std::runtime_error("vector-scale candidate count/status");
        auto actual=copy_candidates(ids,distances,queries,n);
        if(!aligned)reference=actual;
        else for(size_t i=0;i<actual.size();++i)if(actual[i].as_tuple()!=reference[i].as_tuple())
            throw std::runtime_error("vector-scale dispatch bitwise mismatch");
        if (probe_begin == 0) {
            check_cuda(cudaMemset(query_counts.get(),0,nq*sizeof(int)),"clear pair metadata counts");
            check_cuda(cudaMemset(total.get(),0,sizeof(int)),"clear pair metadata total");
            check_cuda(cudaMemset(status.get(),0,sizeof(int)),"clear pair metadata status");
            paraselect::search::pq_adc_query_lut_pair_scan(view, probes.get(),codes.get(),offsets.get(),counts.get(),thresholds.get(),
                nq,1,np,m,output,nullptr,paraselect::search::PQAdcLutReadMode::Global,
                paraselect::search::PQAdcSharedLutLayout::Compact);
            check_cuda(cudaMemcpy(&n,total.get(),4,cudaMemcpyDeviceToHost),"read pair metadata total");
            check_cuda(cudaMemcpy(&s,status.get(),4,cudaMemcpyDeviceToHost),"read pair metadata status");
            if(s || n!=cap)throw std::runtime_error("pair metadata count/status");
            const auto paired=copy_candidates(ids,distances,queries,n);
            for(size_t i=0;i<paired.size();++i)
                if(paired[i].as_tuple()!=actual[i].as_tuple())throw std::runtime_error("pair/list metadata bitwise mismatch");
        }
    }
    std::printf("PASS aligned/unaligned metadata dispatch nq=%d np=%d rows=%d probe_begin=%d\n",nq,np,rows,probe_begin);
}

void test_quantized_pair_and_list(int m_count=20) {
    constexpr int rows=520;
    const std::vector<int> counts{513,4,3};
    const std::vector<long long> offsets{0,513,517};
    const std::vector<idx_t> probes{0,1,0,2,0,2};
    std::vector<uint8_t> codes(rows*m_count), lut(kQueries*m_count*kGlobalLutStride);
    std::vector<uint16_t> row_correction_u16(rows);
    std::vector<float> row_correction(rows), scales(kQueries*m_count), lo(kQueries), inflation(kQueries,0.5f);
    const std::vector<float> coarse{1.f,2.f,1.5f,2.5f,0.5f,3.f};
    const std::vector<float> thresholds{3400.f,3500.f,3600.f};
    for(int row=0;row<rows;++row) {
        row_correction_u16[row]=row; row_correction[row]=3000.f+4.f*row;
        for(int m=0;m<m_count;++m) codes[row*m_count+m]=(row*37+m*11+3)&255;
    }
    for(int q=0;q<kQueries;++q) {
        lo[q]=0.5f*q;
        for(int m=0;m<m_count;++m) {
            scales[q*m_count+m]=float(m+1)/(m_count==64 ? 512.f : 128.f);
            for(int code=0;code<kGlobalLutStride;++code)
                lut[(q*m_count+m)*kGlobalLutStride+code]=(code*13+m*7+q*17)&255;
        }
    }
    DeviceBuffer<int> d_counts(counts.size()); DeviceBuffer<long long> d_offsets(offsets.size());
    DeviceBuffer<idx_t> d_probes(probes.size()); DeviceBuffer<uint8_t> d_codes(codes.size()), d_lut(lut.size());
    DeviceBuffer<uint16_t> d_row_correction_u16(row_correction_u16.size()); DeviceBuffer<float> d_row_correction(row_correction.size()), d_scales(scales.size());
    DeviceBuffer<float> d_lo(lo.size()), d_inflation(inflation.size()), d_coarse(coarse.size()), d_thresholds(thresholds.size());
    auto upload=[](const auto& d,const auto& h) {
        check_cuda(cudaMemcpy(d.get(),h.data(),h.size()*sizeof(h[0]),cudaMemcpyHostToDevice),"upload M20 test data");
    };
    upload(d_counts,counts); upload(d_offsets,offsets); upload(d_probes,probes); upload(d_codes,codes);
    upload(d_lut,lut); upload(d_row_correction_u16,row_correction_u16); upload(d_row_correction,row_correction); upload(d_scales,scales);
    upload(d_lo,lo); upload(d_inflation,inflation); upload(d_coarse,coarse); upload(d_thresholds,thresholds);
    paraselect::search::IVFProbeScheduleBuilder builder(kLists,probes.size(),d_counts.get(),
        paraselect::search::IVFProbeScheduleStrategy::Dense);
    const auto schedule=builder.build({d_probes.get(),kQueries,kProbes},kProbes,nullptr);
    const auto tiles=builder.build_tiles(schedule.view,512,nullptr);
    std::vector<Candidate> expected;
    int expected_counts[kQueries]{};
    for(int q=0;q<kQueries;++q) for(int probe=0;probe<kProbes;++probe) {
        const int list=probes[q*kProbes+probe];
        for(int local=0;local<counts[list];++local) {
            const int row=offsets[list]+local;
            float dot=lo[q];
            for(int m=0;m<m_count;++m)
                dot+=scales[q*m_count+m]*float(lut[(q*m_count+m)*kGlobalLutStride+codes[row*m_count+m]]);
            const float distance=coarse[q*kProbes+probe]+row_correction[row]-2.f*dot;
            if(distance<=thresholds[q]+inflation[q]) {
                uint32_t bits; std::memcpy(&bits,&distance,sizeof(bits));
                expected.push_back({q,row,bits}); ++expected_counts[q];
            }
        }
    }
    std::sort(expected.begin(),expected.end(),[](const Candidate& a,const Candidate& b){return a.as_tuple()<b.as_tuple();});
    if(expected.empty() || expected.size()>=1549) throw std::runtime_error("M20 test must exercise threshold admission");
    DeviceBuffer<int> out_rows(kPoolCapacity), out_queries(kPoolCapacity), query_counts(kQueries), total(1), status(1);
    DeviceBuffer<float> out_distances(kPoolCapacity); DeviceBuffer<unsigned long long> physical(1);
    const paraselect::search::PQAdcThresholdCandidatePoolView output{
        kPoolCapacity,out_rows.get(),out_distances.get(),out_queries.get(),query_counts.get(),total.get(),status.get(),nullptr};
    for(bool quantized_row_correction:{false,true}) {
        paraselect::search::PQQueryLutView view;
        view.query_centroid_distances=d_coarse.get();
        view.query_codeword_dot_lut_u8=d_lut.get(); view.query_lut_scale=d_scales.get();
        view.query_lut_lo_sum=d_lo.get(); view.query_lut_inflation=d_inflation.get();
        if(quantized_row_correction) {view.query_lut_bias.u16=d_row_correction_u16.get(); view.query_lut_bias.scale=4.f; view.query_lut_bias.offset=3000.f;}
        else view.query_lut_bias.fp32=d_row_correction.get();
        check_cuda(cudaMemset(query_counts.get(),0,kQueries*sizeof(int)),"clear M20 query counts");
        check_cuda(cudaMemset(total.get(),0,sizeof(int)),"clear M20 total");
        check_cuda(cudaMemset(status.get(),0,sizeof(int)),"clear M20 status");
        check_cuda(cudaMemset(physical.get(),0,sizeof(unsigned long long)),"clear M20 physical count");
        paraselect::search::pq_adc_query_lut_segmented_scan(view, schedule.view,&tiles.view,d_codes.get(),
            d_offsets.get(),d_counts.get(),d_thresholds.get(),kQueries,kProbes,m_count,
            paraselect::search::PQAdcListScanRowMode::Serial,paraselect::search::PQAdcLutReadMode::Global,
            physical.get(),output,nullptr);
        check_cuda(cudaDeviceSynchronize(),"finish M20 packed code scan");
        int count=0,error=0,actual_counts[kQueries]{}; unsigned long long loaded=0;
        check_cuda(cudaMemcpy(&count,total.get(),sizeof(count),cudaMemcpyDeviceToHost),"copy M20 count");
        check_cuda(cudaMemcpy(&error,status.get(),sizeof(error),cudaMemcpyDeviceToHost),"copy M20 status");
        check_cuda(cudaMemcpy(&loaded,physical.get(),sizeof(loaded),cudaMemcpyDeviceToHost),"copy M20 physical count");
        check_cuda(cudaMemcpy(actual_counts,query_counts.get(),sizeof(actual_counts),cudaMemcpyDeviceToHost),"copy M20 query counts");
        if(error || count!=int(expected.size()) || loaded!=rows || !std::equal(actual_counts,actual_counts+kQueries,expected_counts))
            throw std::runtime_error("M20 scan aggregate mismatch");
        const auto actual=copy_candidates(out_rows,out_distances,out_queries,count);
        for(size_t i=0;i<expected.size();++i)
            if(actual[i].as_tuple()!=expected[i].as_tuple()) throw std::runtime_error("M20 code/ROW_CORRECTION distance mismatch");
        check_cuda(cudaMemset(query_counts.get(),0,kQueries*sizeof(int)),"clear entry counts");
        check_cuda(cudaMemset(total.get(),0,sizeof(int)),"clear entry total");
        check_cuda(cudaMemset(status.get(),0,sizeof(int)),"clear entry status");
        paraselect::search::pq_adc_query_lut_segmented_scan(view, schedule.view,nullptr,d_codes.get(),
            d_offsets.get(),d_counts.get(),d_thresholds.get(),kQueries,kProbes,m_count,
            paraselect::search::PQAdcListScanRowMode::Serial,paraselect::search::PQAdcLutReadMode::Global,
            nullptr,output,nullptr);
        check_cuda(cudaDeviceSynchronize(),"finish entry-mode list scan");
        check_cuda(cudaMemcpy(&count,total.get(),sizeof(count),cudaMemcpyDeviceToHost),"copy entry count");
        check_cuda(cudaMemcpy(&error,status.get(),sizeof(error),cudaMemcpyDeviceToHost),"copy entry status");
        if(error || count!=int(expected.size())) throw std::runtime_error("entry-mode list count mismatch");
        const auto entry_actual=copy_candidates(out_rows,out_distances,out_queries,count);
        for(size_t i=0;i<expected.size();++i)
            if(entry_actual[i].as_tuple()!=expected[i].as_tuple()) throw std::runtime_error("entry-mode list distance mismatch");
        // Pair/list must preserve the same represented distances, including ROW_CORRECTION_U16
        // and the optional quantization compensation. Exercise both LUT stores.
        for (auto read_mode : {paraselect::search::PQAdcLutReadMode::Global,
                              paraselect::search::PQAdcLutReadMode::Shared}) {
            check_cuda(cudaMemset(query_counts.get(),0,kQueries*sizeof(int)),"clear pair counts");
            check_cuda(cudaMemset(total.get(),0,sizeof(int)),"clear pair total");
            check_cuda(cudaMemset(status.get(),0,sizeof(int)),"clear pair status");
            paraselect::search::pq_adc_query_lut_pair_scan(view, d_probes.get(),d_codes.get(),d_offsets.get(),d_counts.get(),
                d_thresholds.get(),kQueries,kLists,kProbes,m_count,output,nullptr,
                read_mode,paraselect::search::PQAdcSharedLutLayout::Compact);
            check_cuda(cudaDeviceSynchronize(),"finish quantized pair scan");
            check_cuda(cudaMemcpy(&count,total.get(),sizeof(count),cudaMemcpyDeviceToHost),"copy pair count");
            check_cuda(cudaMemcpy(&error,status.get(),sizeof(error),cudaMemcpyDeviceToHost),"copy pair status");
            if(error || count!=int(expected.size())) throw std::runtime_error("quantized pair count mismatch");
            const auto paired=copy_candidates(out_rows,out_distances,out_queries,count);
            for(size_t i=0;i<expected.size();++i)
                if(paired[i].as_tuple()!=expected[i].as_tuple()) throw std::runtime_error("quantized pair distance mismatch");
        }
        for (int first_probe = 0; first_probe < kProbes; ++first_probe) {
            check_cuda(cudaMemset(query_counts.get(),0,kQueries*sizeof(int)),"clear M20 prefix counts");
            check_cuda(cudaMemset(total.get(),0,sizeof(int)),"clear M20 prefix total");
            check_cuda(cudaMemset(status.get(),0,sizeof(int)),"clear M20 prefix status");
            paraselect::search::pq_adc_query_lut_segmented_scan(view, schedule.view,&tiles.view,d_codes.get(),d_offsets.get(),d_counts.get(),
                d_thresholds.get(),kQueries,kProbes,m_count,
                paraselect::search::PQAdcListScanRowMode::Serial,
                paraselect::search::PQAdcLutReadMode::Global,physical.get(),output,nullptr,
                paraselect::search::PQAdcSharedLutLayout::Padded,first_probe,first_probe+1);
            check_cuda(cudaDeviceSynchronize(),"finish M20 probe-range scan");
            check_cuda(cudaMemcpy(&count,total.get(),sizeof(count),cudaMemcpyDeviceToHost),"copy M20 range count");
            check_cuda(cudaMemcpy(&error,status.get(),sizeof(error),cudaMemcpyDeviceToHost),"copy M20 range status");
            if (error) throw std::runtime_error("M20 probe-range scan failed");
            const auto selected=copy_candidates(out_rows,out_distances,out_queries,count);
            std::vector<Candidate> matching;
            for (const auto& item : expected) {
                if ((item.row < offsets[1]) == (first_probe == 0)) matching.push_back(item);
            }
            if (selected.size()!=matching.size()) throw std::runtime_error("M20 probe-range count mismatch");
            for (size_t i=0;i<matching.size();++i)
                if (selected[i].as_tuple()!=matching[i].as_tuple())
                    throw std::runtime_error("M20 probe-range candidate mismatch");
        }
    }
}

// Exercise all 32 mask bits, alternating visitors, an empty mask, and the
// unfiltered specialization using identical query/list distances.
void test_probe_mask_boundaries() {
    using namespace paraselect::search;
    constexpr int nq = 32;
    DeviceBuffer<float> lut(nq * kPqM * kGlobalLutStride), coarse(nq * 2), row_correction(1), thresholds(nq);
    DeviceBuffer<uint8_t> codes(kPqM);
    DeviceBuffer<long long> offsets(1);
    DeviceBuffer<int> clusters(1), starts(1), counts(1), list_counts(1), queries(nq), ranks(nq);
    DeviceBuffer<int> rows(nq), out_queries(nq), per_query(nq), total(1), status(1);
    DeviceBuffer<float> distances(nq);
    auto put = [](auto& buffer, const auto& data) {
        check_cuda(cudaMemcpy(buffer.get(), data.data(), data.size() * sizeof(data[0]),
                              cudaMemcpyHostToDevice), "upload mask test");
    };
    put(lut, std::vector<float>(lut.size(), 0.f));
    put(coarse, std::vector<float>(coarse.size(), 0.f));
    put(row_correction, std::vector<float>{1.f}); put(thresholds, std::vector<float>(nq, 2.f));
    put(codes, std::vector<uint8_t>(kPqM, 0)); put(offsets, std::vector<long long>{0});
    put(clusters, std::vector<int>{0}); put(starts, std::vector<int>{0});
    put(counts, std::vector<int>{nq}); put(list_counts, std::vector<int>{1});
    std::vector<int> qs(nq), rs(nq);
    for (int q = 0; q < nq; ++q) qs[q] = q;
    put(queries, qs);
    PQQueryLutView view;
    view.query_codeword_dot_lut = lut.get(); view.query_centroid_distances = coarse.get();
    view.query_lut_bias.fp32 = row_correction.get();
    IVFProbeScheduleView schedule{clusters.get(), starts.get(), counts.get(), queries.get(), ranks.get(), 1};
    PQAdcThresholdCandidatePoolView output{nq, rows.get(), distances.get(), out_queries.get(),
        per_query.get(), total.get(), status.get(), nullptr};
    for (int pattern = 0; pattern < 4; ++pattern) {
        for (int q = 0; q < nq; ++q)
            rs[q] = pattern == 0 ? (q == 31) : pattern == 1 ? q % 2 : 0;
        put(ranks, rs);
        check_cuda(cudaMemset(per_query.get(), 0, nq * sizeof(int)), "clear mask query counts");
        check_cuda(cudaMemset(total.get(), 0, sizeof(int)), "clear mask total");
        check_cuda(cudaMemset(status.get(), 0, sizeof(int)), "clear mask status");
        pq_adc_query_lut_segmented_scan(view, schedule, nullptr, codes.get(), offsets.get(),
            list_counts.get(), thresholds.get(), nq, 2, kPqM, PQAdcListScanRowMode::Serial,
            PQAdcLutReadMode::Global, nullptr, output, nullptr, PQAdcSharedLutLayout::Compact,
            pattern == 3 ? 0 : 1, 2);
        check_cuda(cudaDeviceSynchronize(), "mask scan");
        int count = 0, error = 0;
        check_cuda(cudaMemcpy(&count, total.get(), sizeof(int), cudaMemcpyDeviceToHost), "mask count");
        check_cuda(cudaMemcpy(&error, status.get(), sizeof(int), cudaMemcpyDeviceToHost), "mask status");
        const auto actual = copy_candidates(rows, distances, out_queries, count);
        std::vector<int> expected;
        for (int q = 0; q < nq; ++q) if (pattern == 3 || rs[q] == 1) expected.push_back(q);
        if (error || actual.size() != expected.size()) throw std::runtime_error("mask count mismatch");
        for (size_t i = 0; i < actual.size(); ++i)
            if (actual[i].query != expected[i] || actual[i].row != 0 || actual[i].distance_bits != 0x3f800000u)
                throw std::runtime_error("mask candidate mismatch");
    }
    std::puts("PASS probe mask: bit31, alternating, empty, full");
}

void test_seed_exact_top512() {
    using namespace paraselect::search;
    constexpr int nq=3, nrows=1814;
    for (int k : {256, 512, 1024, 2048})
    for(int mcount : {16,20,64}) for(int seed_count : {2,8,17})
    for (int scratch_queries : {1, 2, nq * 4}) {
        std::vector<uint8_t> code(nrows*mcount), lut(nq*mcount*kGlobalLutStride);
        std::vector<uint16_t> row_correction(nrows);
        std::vector<float> scales(nq*mcount,1.f), zeros(nq,0.f), coarse(nq*seed_count,0.f);
        for (int q=0; q<nq; ++q) for (int probe=0; probe<seed_count; ++probe)
            coarse[q*seed_count+probe] = q*500.f + probe*250.f;
        for(int row=0;row<nrows;++row) {
            row_correction[row]=row%97;
            for(int m=0;m<mcount;++m) code[row*mcount+m]=(row*13+m*17)%256;
        }
        for(size_t i=0;i<lut.size();++i) lut[i]=(i*7+i/256)%11;
        DeviceBuffer<uint8_t> dc(code.size()), dl(lut.size());
        DeviceBuffer<uint16_t> dp(row_correction.size());
        DeviceBuffer<float> ds(scales.size()), dz(nq), dq(coarse.size()), outd(nq*k);
        DeviceBuffer<idx_t> probes(nq*seed_count);
        DeviceBuffer<long long> offsets(2);
        DeviceBuffer<int> counts(2), out(nq*k), status(1);
        DeviceBuffer<uint64_t> scratch(scratch_queries*seed_count*k), merge_scratch(scratch_queries*seed_count*k);
        auto put=[](auto& dst,const auto& src){check_cuda(cudaMemcpy(dst.get(),src.data(),src.size()*sizeof(src[0]),cudaMemcpyHostToDevice),"seed upload");};
        put(dc,code);put(dl,lut);put(dp,row_correction);put(ds,scales);put(dz,zeros);put(dq,coarse);
        std::vector<idx_t> probe_ids(nq*seed_count,-1);
        for(int q=0;q<nq;++q){probe_ids[q*seed_count]=0;probe_ids[q*seed_count+1]=1;}
        put(probes,probe_ids);put(offsets,std::vector<long long>{0,1301});put(counts,std::vector<int>{1301,513});
        PQQueryLutView v;
        v.query_codeword_dot_lut_u8=dl.get();v.query_lut_scale=ds.get();
        v.query_lut_lo_sum=dz.get();v.query_lut_inflation=dz.get();v.query_centroid_distances=dq.get();
        v.query_lut_bias.u16=dp.get();v.query_lut_bias.scale=1.f;v.query_lut_bias.offset=-60.f;
        check_cuda(cudaMemset(status.get(),0,sizeof(int)),"seed clear status");
        pq_query_lut_seed_topk(v,probes.get(),dc.get(),offsets.get(),counts.get(),nq,seed_count,seed_count,mcount,k,true,
            scratch.get(),merge_scratch.get(),scratch.size(),out.get(),outd.get(),status.get(),nullptr,nullptr);
        check_cuda(cudaDeviceSynchronize(),"seed top512 sync");
        std::vector<int> ids(nq*k);std::vector<float> distances(nq*k);int error=0;
        check_cuda(cudaMemcpy(ids.data(),out.get(),ids.size()*sizeof(int),cudaMemcpyDeviceToHost),"seed ids");
        check_cuda(cudaMemcpy(distances.data(),outd.get(),distances.size()*sizeof(float),cudaMemcpyDeviceToHost),"seed distances");
        check_cuda(cudaMemcpy(&error,status.get(),sizeof(int),cudaMemcpyDeviceToHost),"seed status");
        if(error)throw std::runtime_error("seed status nonzero");
        constexpr int bins = 256;
        const bool check_histogram = k == 512 && scratch_queries == 1;
        std::vector<unsigned> actual_histogram(nq*bins);
        if (check_histogram) {
            DeviceBuffer<unsigned> histogram(nq*bins);
            check_cuda(cudaMemset(histogram.get(),0,nq*bins*sizeof(unsigned)),"histogram clear");
            pq_query_lut_seed_histogram(v,probes.get(),dc.get(),offsets.get(),counts.get(),
                nq,seed_count,seed_count,2,mcount,true,histogram.get(),bins,-8.f,0.125f,
                nullptr);
            check_cuda(cudaMemcpy(actual_histogram.data(),histogram.get(),nq*bins*sizeof(unsigned),
                cudaMemcpyDeviceToHost),"histogram download");
        }
        for(int q=0;q<nq;++q){
            std::vector<std::pair<float,int>> expected;
            for(int row=0;row<nrows;++row){
                float dot=0;
                for(int m=0;m<mcount;++m)dot+=lut[(q*mcount+m)*kGlobalLutStride+code[row*mcount+m]];
                expected.emplace_back(coarse[q*seed_count+(row>=1301)] + float(row_correction[row])-60.f-2.f*dot,row);
            }
            if (check_histogram) {
                std::vector<unsigned> expected_histogram(bins,0);
                std::vector<float> edges(bins);
                for (int bin=0; bin<bins; ++bin) edges[bin]=std::exp2(-8.f+(bin+1)*0.125f);
                for (const auto& item : expected) {
                    const int bin=std::min(bins-1,int(std::upper_bound(edges.begin(),edges.end(),item.first)-edges.begin()));
                    ++expected_histogram[bin];
                }
                if (!std::equal(expected_histogram.begin(),expected_histogram.end(),actual_histogram.begin()+q*bins))
                    throw std::runtime_error("seed histogram differs from CPU histogram for query=" + std::to_string(q));
            }
            std::sort(expected.begin(),expected.end());
            for(int i=0;i<k;++i)
                if (i >= int(expected.size()) ? (ids[q*k+i] != -1 || !std::isinf(distances[q*k+i])) :
                    (ids[q*k+i]!=expected[i].second || distances[q*k+i]!=expected[i].first))
                    throw std::runtime_error("seed exact top512 differs from exhaustive CPU sort");
        }
    }
    std::puts("PASS exact seed k=256/512/1024/2048 M16/M20/M64, query chunk tails, ties, insufficient seed; histogram CPU oracle");
}

// Same public dispatcher and represented data; unaligned scales select the
// existing generic kernel without adding a production A/B switch.
void benchmark_pair_correction() {
    using namespace paraselect::search;
    constexpr int lists=65536, rows_per_list=512, rows=lists*rows_per_list;
    constexpr int max_queries=10000, max_probes=512, capacity=1<<25;
    std::printf("m,queries,probes,u16,specialized,repeat,scan_ms,candidates\n");
    for (int m : {16,20}) {
        DeviceBuffer<uint8_t> codes(size_t(rows)*m), lut(size_t(max_queries)*m*kGlobalLutStride);
        DeviceBuffer<uint16_t> correction_u16(rows);
        DeviceBuffer<float> correction(rows), scales(size_t(max_queries)*m), unaligned(size_t(max_queries)*m+1);
        DeviceBuffer<float> lo(max_queries), coarse(size_t(max_queries)*max_probes), thresholds(max_queries);
        DeviceBuffer<long long> offsets(lists);
        DeviceBuffer<int> counts(lists), ids(capacity), queries(capacity), query_counts(max_queries), total(1), status(1);
        DeviceBuffer<float> distances(capacity);
        DeviceBuffer<idx_t> probes(size_t(max_queries)*max_probes);
        auto upload=[](auto* dst,const auto& src) {
            check_cuda(cudaMemcpy(dst,src.data(),src.size()*sizeof(src[0]),cudaMemcpyHostToDevice),"upload micro input");
        };
        uint32_t state=1234567;
        const auto random=[&]() { state^=state<<13; state^=state>>17; state^=state<<5; return state; };
        {
            std::vector<uint8_t> host(size_t(rows)*m);
            for(auto& x:host)x=uint8_t(random());
            upload(codes.get(),host);
        }
        {
            std::vector<float> host(rows);
            for(auto& x:host)x=float(random()&65535);
            upload(correction.get(),host);
            upload(correction_u16.get(),std::vector<uint16_t>(host.begin(),host.end()));
        }
        {
            std::vector<uint8_t> host(size_t(max_queries)*m*kGlobalLutStride);
            for(auto& x:host)x=uint8_t(random());
            upload(lut.get(),host);
        }
        std::vector<float> host_scales(size_t(max_queries)*m);
        for(auto& x:host_scales)x=float(1+(random()%7))/64.f;
        upload(scales.get(),host_scales); upload(unaligned.get()+1,host_scales);
        std::vector<long long> host_offsets(lists);
        for(int i=0;i<lists;++i)host_offsets[i]=size_t(i)*rows_per_list;
        upload(offsets.get(),host_offsets);
        upload(counts.get(),std::vector<int>(lists,rows_per_list));
        check_cuda(cudaMemset(lo.get(),0,max_queries*sizeof(float)),"zero lo");
        check_cuda(cudaMemset(coarse.get(),0,size_t(max_queries)*max_probes*sizeof(float)),"zero coarse");
        check_cuda(cudaMemset(thresholds.get(),0,max_queries*sizeof(float)),"zero thresholds");
        PQQueryLutView view{};
        view.query_codeword_dot_lut_u8=lut.get(); view.query_lut_lo_sum=lo.get();
        view.query_centroid_distances=coarse.get(); view.query_lut_bias.fp32=correction.get();
        const PQAdcThresholdCandidatePoolView output{capacity,ids.get(),distances.get(),queries.get(),
            query_counts.get(),total.get(),status.get(),nullptr};
        cudaEvent_t begin,end;
        check_cuda(cudaEventCreate(&begin),"create begin"); check_cuda(cudaEventCreate(&end),"create end");
        for(int nq : {8,32,256,2048,10000}) for(int np : {32,512}) {
            std::vector<idx_t> host_probes(size_t(nq)*np);
            for(auto& x:host_probes)x=random()%lists;
            upload(probes.get(),host_probes);
            int reference=-1;
            for(int repeat=0;repeat<7;++repeat) for(int arm=0;arm<4;++arm) {
                const int config=(arm+repeat)%4;
                const int specialized=config%2;
                const int u16=config/2;
                view.query_lut_bias = u16
                    ? QueryLutBias{nullptr, correction_u16.get(), 1.0f, 0.0f, rows}
                    : QueryLutBias{correction.get(), nullptr, 0.0f, 0.0f, rows};
                view.query_lut_scale=specialized?scales.get():unaligned.get()+1;
                check_cuda(cudaMemset(query_counts.get(),0,nq*sizeof(int)),"zero query counts");
                check_cuda(cudaMemset(total.get(),0,sizeof(int)),"zero total");
                check_cuda(cudaMemset(status.get(),0,sizeof(int)),"zero status");
                check_cuda(cudaEventRecord(begin),"record begin");
                pq_adc_query_lut_pair_scan(view,probes.get(),codes.get(),offsets.get(),counts.get(),
                    thresholds.get(),nq,lists,np,m,output,nullptr,PQAdcLutReadMode::Global,PQAdcSharedLutLayout::Compact);
                check_cuda(cudaEventRecord(end),"record end"); check_cuda(cudaEventSynchronize(end),"finish scan");
                float ms;check_cuda(cudaEventElapsedTime(&ms,begin,end),"elapsed scan");
                int n,error;check_cuda(cudaMemcpy(&n,total.get(),4,cudaMemcpyDeviceToHost),"read total");
                check_cuda(cudaMemcpy(&error,status.get(),4,cudaMemcpyDeviceToHost),"read status");
                if(reference<0)reference=n;
                if(error || n!=reference)throw std::runtime_error("micro candidate count/status mismatch");
                std::printf("%d,%d,%d,%d,%d,%d,%.6f,%d\n",m,nq,np,u16,specialized,repeat,ms,n);
            }
        }
        cudaEventDestroy(begin);cudaEventDestroy(end);
    }
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--pair-correction-micro") {
            benchmark_pair_correction(); return 0;
        }
        test_vector_scale_dispatch(256,1,19);
        test_vector_scale_dispatch(256,63,19);
        test_vector_scale_dispatch(256,64,19);
        test_vector_scale_dispatch(8,8,513);
        test_vector_scale_dispatch(8,8,513,1);
        test_probe_mask_boundaries();
        test_seed_exact_top512();
        if(argc==2 && std::string(argv[1])=="--m20-only") {
            test_quantized_pair_and_list(); test_quantized_pair_and_list(16); test_quantized_pair_and_list(64);
            std::puts("PASS pair/list/entry/probe-range scans M16/M20/M64 against CPU oracle");
            return 0;
        }
        const std::vector<int> list_counts = {513, 4, 3};
        const std::vector<long long> list_offsets = {0, 513, 517};
        const std::vector<idx_t> probes = {0, 1, 0, 2, 0, 2};
        constexpr int kRows = 520;
        constexpr int kExpectedCandidates = 1549;

        std::vector<uint8_t> codes(kRows * kPqM);
        for (size_t i = 0; i < codes.size(); ++i) {
            codes[i] = static_cast<uint8_t>((i * 17 + 3) & 255);
        }
        std::vector<float> row_correction(kRows);
        for (int row = 0; row < kRows; ++row) row_correction[row] = 0.125f * row;
        std::vector<float> query_lut(kQueries * kPqM * kGlobalLutStride);
        for (int q = 0; q < kQueries; ++q) {
            for (int m = 0; m < kPqM; ++m) {
                for (int code = 0; code < kGlobalLutStride; ++code) {
                    query_lut[((size_t)q * kPqM + m) * kGlobalLutStride + code] =
                        0.01f * q + 0.001f * m + 0.0001f * code;
                }
            }
        }
        const std::vector<float> coarse_distances = {
            1.0f, 2.0f, 1.5f, 2.5f, 0.5f, 3.0f};
        const std::vector<float> thresholds(kQueries, 1.0e9f);

        DeviceBuffer<int> d_list_counts(list_counts.size());
        DeviceBuffer<long long> d_list_offsets(list_offsets.size());
        DeviceBuffer<idx_t> d_probes(probes.size());
        DeviceBuffer<uint8_t> d_codes(codes.size());
        DeviceBuffer<float> d_row_correction(row_correction.size());
        DeviceBuffer<float> d_query_lut(query_lut.size());
        DeviceBuffer<float> d_coarse_distances(coarse_distances.size());
        DeviceBuffer<float> d_thresholds(thresholds.size());
        const std::vector<float> finite_thresholds(kQueries, 30.0f);
        DeviceBuffer<float> d_finite_thresholds(finite_thresholds.size());

        check_cuda(cudaMemcpy(d_list_counts.get(), list_counts.data(),
                              list_counts.size() * sizeof(int), cudaMemcpyHostToDevice),
                   "copy list counts");
        check_cuda(cudaMemcpy(d_list_offsets.get(), list_offsets.data(),
                              list_offsets.size() * sizeof(long long), cudaMemcpyHostToDevice),
                   "copy list offsets");
        check_cuda(cudaMemcpy(d_probes.get(), probes.data(),
                              probes.size() * sizeof(idx_t), cudaMemcpyHostToDevice),
                   "copy probes");
        check_cuda(cudaMemcpy(d_codes.get(), codes.data(), codes.size(), cudaMemcpyHostToDevice),
                   "copy codes");
        check_cuda(cudaMemcpy(d_row_correction.get(), row_correction.data(),
                              row_correction.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "copy row correction");
        check_cuda(cudaMemcpy(d_query_lut.get(), query_lut.data(),
                              query_lut.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "copy query LUT");
        check_cuda(cudaMemcpy(d_coarse_distances.get(), coarse_distances.data(),
                              coarse_distances.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "copy coarse distances");
        check_cuda(cudaMemcpy(d_thresholds.get(), thresholds.data(),
                              thresholds.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "copy thresholds");
        check_cuda(cudaMemcpy(d_finite_thresholds.get(), finite_thresholds.data(),
                              finite_thresholds.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "copy finite thresholds");

        paraselect::search::IVFProbeScheduleBuilder schedule_builder(
            kLists, static_cast<int>(probes.size()), d_list_counts.get(),
            paraselect::search::IVFProbeScheduleStrategy::Dense);
        const auto schedule = schedule_builder.build(
            {d_probes.get(), kQueries, kProbes}, kProbes, nullptr);
        const auto tile_schedule =
            schedule_builder.build_tiles(schedule.view, 512, nullptr);
        if (tile_schedule.view.work_count != 4) {
            throw std::runtime_error("fixed-tile schedule work count mismatch");
        }
        const paraselect::search::PQQueryLutView query_lut_row_correction{
            d_query_lut.get(), d_coarse_distances.get(), {d_row_correction.get()}};

        DeviceBuffer<int> pair_rows(kPoolCapacity);
        DeviceBuffer<float> pair_distances(kPoolCapacity);
        DeviceBuffer<int> pair_queries(kPoolCapacity);
        DeviceBuffer<int> pair_query_counts(kQueries);
        DeviceBuffer<int> pair_total(1);
        DeviceBuffer<int> pair_status(1);
        check_cuda(cudaMemset(pair_query_counts.get(), 0, kQueries * sizeof(int)),
                   "clear pair counts");
        check_cuda(cudaMemset(pair_total.get(), 0, sizeof(int)), "clear pair total");
        check_cuda(cudaMemset(pair_status.get(), 0, sizeof(int)), "clear pair status");
        const paraselect::search::PQAdcThresholdCandidatePoolView pair_output{
            kPoolCapacity, pair_rows.get(), pair_distances.get(), pair_queries.get(),
            pair_query_counts.get(), pair_total.get(), pair_status.get(), nullptr};
        paraselect::search::pq_adc_query_lut_pair_scan(query_lut_row_correction, d_probes.get(), d_codes.get(), d_list_offsets.get(),
            d_list_counts.get(), d_thresholds.get(), kQueries, kLists, kProbes, kPqM,
            pair_output, nullptr);

        DeviceBuffer<int> segmented_rows(kPoolCapacity);
        DeviceBuffer<float> segmented_distances(kPoolCapacity);
        DeviceBuffer<int> segmented_queries(kPoolCapacity);
        DeviceBuffer<int> segmented_query_counts(kQueries);
        DeviceBuffer<int> segmented_total(1);
        DeviceBuffer<int> segmented_status(1);
        DeviceBuffer<unsigned long long> physical_rows(1);
        check_cuda(cudaMemset(segmented_query_counts.get(), 0, kQueries * sizeof(int)),
                   "clear segmented counts");
        check_cuda(cudaMemset(segmented_total.get(), 0, sizeof(int)),
                   "clear segmented total");
        check_cuda(cudaMemset(segmented_status.get(), 0, sizeof(int)),
                   "clear segmented status");
        check_cuda(cudaMemset(physical_rows.get(), 0, sizeof(unsigned long long)),
                   "clear physical rows");
        const paraselect::search::PQAdcThresholdCandidatePoolView segmented_output{
            kPoolCapacity, segmented_rows.get(), segmented_distances.get(),
            segmented_queries.get(), segmented_query_counts.get(), segmented_total.get(),
            segmented_status.get(), nullptr};
        paraselect::search::pq_adc_query_lut_segmented_scan(query_lut_row_correction, schedule.view, nullptr, d_codes.get(), d_list_offsets.get(),
            d_list_counts.get(), d_thresholds.get(), kQueries, kProbes, kPqM,
            paraselect::search::PQAdcListScanRowMode::Serial,
            paraselect::search::PQAdcLutReadMode::Shared,
            physical_rows.get(), segmented_output, nullptr);
        check_cuda(cudaDeviceSynchronize(), "finish segmented scan test");

        int pair_count = 0;
        int segmented_count = 0;
        int pair_error = 0;
        int segmented_error = 0;
        unsigned long long loaded_rows = 0;
        check_cuda(cudaMemcpy(&pair_count, pair_total.get(), sizeof(int), cudaMemcpyDeviceToHost),
                   "copy pair total");
        check_cuda(cudaMemcpy(&segmented_count, segmented_total.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy segmented total");
        check_cuda(cudaMemcpy(&pair_error, pair_status.get(), sizeof(int), cudaMemcpyDeviceToHost),
                   "copy pair status");
        check_cuda(cudaMemcpy(&segmented_error, segmented_status.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy segmented status");
        check_cuda(cudaMemcpy(&loaded_rows, physical_rows.get(), sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost), "copy physical rows");
        if (pair_error != 0 || segmented_error != 0 ||
            pair_count != kExpectedCandidates ||
            segmented_count != pair_count) {
            throw std::runtime_error("segmented scan candidate count mismatch");
        }
        if (loaded_rows != kRows) {
            throw std::runtime_error("segmented scan did not reuse list rows across queries");
        }
        const auto pair = copy_candidates(
            pair_rows, pair_distances, pair_queries, pair_count);
        check_cuda(cudaMemset(pair_query_counts.get(), 0, kQueries * sizeof(int)),
                   "clear global pair counts");
        check_cuda(cudaMemset(pair_total.get(), 0, sizeof(int)), "clear global pair total");
        check_cuda(cudaMemset(pair_status.get(), 0, sizeof(int)), "clear global pair status");
        paraselect::search::pq_adc_query_lut_pair_scan(query_lut_row_correction, d_probes.get(), d_codes.get(), d_list_offsets.get(),
            d_list_counts.get(), d_thresholds.get(), kQueries, kLists, kProbes, kPqM,
            pair_output, nullptr, paraselect::search::PQAdcLutReadMode::Global);
        check_cuda(cudaDeviceSynchronize(), "finish global pair scan test");
        check_cuda(cudaMemcpy(&pair_count, pair_total.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy global pair total");
        check_cuda(cudaMemcpy(&pair_error, pair_status.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy global pair status");
        if (pair_error != 0 || pair_count != kExpectedCandidates) {
            throw std::runtime_error("global pair scan aggregate mismatch");
        }
        const auto global_pair = copy_candidates(
            pair_rows, pair_distances, pair_queries, pair_count);
        if (global_pair.size() != pair.size()) {
            throw std::runtime_error("global pair scan output size mismatch");
        }
        for (size_t i = 0; i < pair.size(); ++i) {
            if (pair[i].as_tuple() != global_pair[i].as_tuple()) {
                throw std::runtime_error("global pair LUT reads changed candidate triples");
            }
        }
        const auto segmented = copy_candidates(
            segmented_rows, segmented_distances, segmented_queries, segmented_count);
        for (const Candidate& candidate : pair) {
            const int cluster = candidate.row < list_offsets[1]
                ? 0
                : candidate.row < list_offsets[2] ? 1 : 2;
            int probe = -1;
            for (int p = 0; p < kProbes; ++p) {
                if (probes[candidate.query * kProbes + p] == cluster) {
                    probe = p;
                    break;
                }
            }
            if (probe < 0) {
                throw std::runtime_error("candidate row is outside its query probe set");
            }
            float query_dot_reconstruction = 0.0f;
            for (int m = 0; m < kPqM; ++m) {
                const int code = codes[candidate.row * kPqM + m];
                query_dot_reconstruction += query_lut[
                    ((size_t)candidate.query * kPqM + m) * kGlobalLutStride + code];
            }
            const float expected_distance =
                coarse_distances[candidate.query * kProbes + probe] +
                row_correction[candidate.row] - 2.0f * query_dot_reconstruction;
            uint32_t expected_bits = 0;
            std::memcpy(&expected_bits, &expected_distance, sizeof(expected_bits));
            if (candidate.distance_bits != expected_bits) {
                throw std::runtime_error("query-LUT candidate distance mismatch");
            }
        }
        if (pair.size() != segmented.size()) {
            throw std::runtime_error("segmented scan output size mismatch");
        }
        for (size_t i = 0; i < pair.size(); ++i) {
            if (pair[i].as_tuple() != segmented[i].as_tuple()) {
                throw std::runtime_error("segmented scan changed candidate triples");
            }
        }

        DeviceBuffer<int> tiled_rows(kPoolCapacity);
        DeviceBuffer<float> tiled_distances(kPoolCapacity);
        DeviceBuffer<int> tiled_queries(kPoolCapacity);
        DeviceBuffer<int> tiled_query_counts(kQueries);
        DeviceBuffer<int> tiled_total(1);
        DeviceBuffer<int> tiled_status(1);
        check_cuda(cudaMemset(tiled_query_counts.get(), 0, kQueries * sizeof(int)),
                   "clear tiled counts");
        check_cuda(cudaMemset(tiled_total.get(), 0, sizeof(int)), "clear tiled total");
        check_cuda(cudaMemset(tiled_status.get(), 0, sizeof(int)), "clear tiled status");
        check_cuda(cudaMemset(physical_rows.get(), 0, sizeof(unsigned long long)),
                   "clear tiled physical rows");
        const paraselect::search::PQAdcThresholdCandidatePoolView tiled_output{
            kPoolCapacity, tiled_rows.get(), tiled_distances.get(), tiled_queries.get(),
            tiled_query_counts.get(), tiled_total.get(), tiled_status.get(), nullptr};
        paraselect::search::pq_adc_query_lut_segmented_scan(query_lut_row_correction, schedule.view, &tile_schedule.view, d_codes.get(),
            d_list_offsets.get(), d_list_counts.get(), d_thresholds.get(), kQueries,
            kProbes, kPqM, paraselect::search::PQAdcListScanRowMode::Serial,
            paraselect::search::PQAdcLutReadMode::Shared,
            physical_rows.get(), tiled_output, nullptr);
        check_cuda(cudaDeviceSynchronize(), "finish fixed-tile segmented scan test");
        int tiled_count = 0;
        int tiled_error = 0;
        check_cuda(cudaMemcpy(&tiled_count, tiled_total.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy tiled total");
        check_cuda(cudaMemcpy(&tiled_error, tiled_status.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy tiled status");
        check_cuda(cudaMemcpy(&loaded_rows, physical_rows.get(), sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost), "copy tiled physical rows");
        if (tiled_error != 0 || tiled_count != pair_count || loaded_rows != kRows) {
            throw std::runtime_error("fixed-tile segmented scan aggregate mismatch");
        }
        const auto tiled = copy_candidates(
            tiled_rows, tiled_distances, tiled_queries, tiled_count);
        if (tiled.size() != pair.size()) {
            throw std::runtime_error("fixed-tile segmented scan output size mismatch");
        }
        for (size_t i = 0; i < pair.size(); ++i) {
            if (pair[i].as_tuple() != tiled[i].as_tuple()) {
                throw std::runtime_error("fixed-tile segmented scan changed candidate triples");
            }
        }

        check_cuda(cudaMemset(tiled_query_counts.get(), 0, kQueries * sizeof(int)),
                   "clear global LUT counts");
        check_cuda(cudaMemset(tiled_total.get(), 0, sizeof(int)),
                   "clear global LUT total");
        check_cuda(cudaMemset(tiled_status.get(), 0, sizeof(int)),
                   "clear global LUT status");
        check_cuda(cudaMemset(physical_rows.get(), 0, sizeof(unsigned long long)),
                   "clear global LUT physical rows");
        paraselect::search::pq_adc_query_lut_segmented_scan(query_lut_row_correction, schedule.view, &tile_schedule.view, d_codes.get(),
            d_list_offsets.get(), d_list_counts.get(), d_thresholds.get(), kQueries,
            kProbes, kPqM, paraselect::search::PQAdcListScanRowMode::Serial,
            paraselect::search::PQAdcLutReadMode::Global,
            physical_rows.get(), tiled_output, nullptr);
        check_cuda(cudaDeviceSynchronize(), "finish global LUT scan test");
        check_cuda(cudaMemcpy(&tiled_count, tiled_total.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy global LUT total");
        check_cuda(cudaMemcpy(&tiled_error, tiled_status.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy global LUT status");
        check_cuda(cudaMemcpy(&loaded_rows, physical_rows.get(), sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost), "copy global LUT physical rows");
        if (tiled_error != 0 || tiled_count != pair_count || loaded_rows != kRows) {
            throw std::runtime_error("global LUT scan aggregate mismatch");
        }
        const auto global_lut = copy_candidates(
            tiled_rows, tiled_distances, tiled_queries, tiled_count);
        for (size_t i = 0; i < pair.size(); ++i) {
            if (pair[i].as_tuple() != global_lut[i].as_tuple()) {
                throw std::runtime_error("global LUT reads changed candidate triples");
            }
        }

        std::vector<uint16_t> row_correction_u16(kRows);
        for (int row = 0; row < kRows; ++row) {
            row_correction_u16[row] = static_cast<uint16_t>(row);
        }
        DeviceBuffer<uint16_t> d_row_correction_u16(row_correction_u16.size());
        check_cuda(cudaMemcpy(
                       d_row_correction_u16.get(), row_correction_u16.data(),
                       row_correction_u16.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice),
                   "copy ROW_CORRECTION_U16 row correction");
        auto row_correction_u16_lut_pc = query_lut_row_correction;
        row_correction_u16_lut_pc.query_lut_bias.fp32 = nullptr;
        row_correction_u16_lut_pc.query_lut_bias.u16 = d_row_correction_u16.get();
        row_correction_u16_lut_pc.query_lut_bias.scale = 0.125f;
        row_correction_u16_lut_pc.query_lut_bias.offset = 0.0f;
        check_cuda(cudaMemset(tiled_query_counts.get(), 0, kQueries * sizeof(int)),
                   "clear ROW_CORRECTION_U16 counts");
        check_cuda(cudaMemset(tiled_total.get(), 0, sizeof(int)),
                   "clear ROW_CORRECTION_U16 total");
        check_cuda(cudaMemset(tiled_status.get(), 0, sizeof(int)),
                   "clear ROW_CORRECTION_U16 status");
        check_cuda(cudaMemset(physical_rows.get(), 0, sizeof(unsigned long long)),
                   "clear ROW_CORRECTION_U16 physical rows");
        paraselect::search::pq_adc_query_lut_segmented_scan(row_correction_u16_lut_pc, schedule.view, &tile_schedule.view, d_codes.get(),
            d_list_offsets.get(), d_list_counts.get(), d_thresholds.get(), kQueries,
            kProbes, kPqM, paraselect::search::PQAdcListScanRowMode::Serial,
            paraselect::search::PQAdcLutReadMode::Shared,
            physical_rows.get(), tiled_output, nullptr,
            paraselect::search::PQAdcSharedLutLayout::Compact);
        check_cuda(cudaDeviceSynchronize(), "finish ROW_CORRECTION_U16 scan");
        check_cuda(cudaMemcpy(&tiled_count, tiled_total.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy ROW_CORRECTION_U16 total");
        const auto u16_candidates = copy_candidates(
            tiled_rows, tiled_distances, tiled_queries, tiled_count);
        if (u16_candidates.size() != pair.size()) {
            throw std::runtime_error("ROW_CORRECTION_U16 changed candidate count");
        }
        for (size_t i = 0; i < pair.size(); ++i) {
            if (pair[i].as_tuple() != u16_candidates[i].as_tuple()) {
                throw std::runtime_error("ROW_CORRECTION_U16 changed exact candidate triples");
            }
        }

        std::vector<uint8_t> quantized_lut(
            kQueries * kPqM * kGlobalLutStride);
        std::vector<float> quantized_scale(kQueries * kPqM);
        std::vector<float> quantized_lo_sum(kQueries);
        std::vector<float> quantized_inflation(kQueries, 0.0f);
        for (int q = 0; q < kQueries; ++q) {
            quantized_lo_sum[q] = 0.25f * q;
            for (int m = 0; m < kPqM; ++m) {
                quantized_scale[q * kPqM + m] = 0.0005f * (m + 1);
                for (int code = 0; code < kGlobalLutStride; ++code) {
                    quantized_lut[((size_t)q * kPqM + m) *
                                      kGlobalLutStride + code] =
                        static_cast<uint8_t>((code + 7 * m + 13 * q) & 255);
                }
            }
        }
        DeviceBuffer<uint8_t> d_quantized_lut(quantized_lut.size());
        DeviceBuffer<float> d_quantized_scale(quantized_scale.size());
        DeviceBuffer<float> d_quantized_lo_sum(quantized_lo_sum.size());
        DeviceBuffer<float> d_quantized_inflation(quantized_inflation.size());
        check_cuda(cudaMemcpy(d_quantized_lut.get(), quantized_lut.data(),
                              quantized_lut.size(), cudaMemcpyHostToDevice),
                   "copy quantized LUT");
        check_cuda(cudaMemcpy(d_quantized_scale.get(), quantized_scale.data(),
                              quantized_scale.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy quantized scales");
        check_cuda(cudaMemcpy(d_quantized_lo_sum.get(), quantized_lo_sum.data(),
                              quantized_lo_sum.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy quantized low sums");
        check_cuda(cudaMemcpy(d_quantized_inflation.get(), quantized_inflation.data(),
                              quantized_inflation.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy quantized inflation");
        paraselect::search::PQQueryLutView quantized_lut_pc;
        quantized_lut_pc.query_centroid_distances = d_coarse_distances.get();
        quantized_lut_pc.query_lut_bias.fp32 = d_row_correction.get();
        quantized_lut_pc.query_codeword_dot_lut_u8 = d_quantized_lut.get();
        quantized_lut_pc.query_lut_scale = d_quantized_scale.get();
        quantized_lut_pc.query_lut_lo_sum = d_quantized_lo_sum.get();
        quantized_lut_pc.query_lut_inflation = d_quantized_inflation.get();
        const auto run_quantized = [&](paraselect::search::PQAdcLutReadMode mode) {
            check_cuda(cudaMemset(tiled_query_counts.get(), 0, kQueries * sizeof(int)),
                       "clear quantized counts");
            check_cuda(cudaMemset(tiled_total.get(), 0, sizeof(int)),
                       "clear quantized total");
            check_cuda(cudaMemset(tiled_status.get(), 0, sizeof(int)),
                       "clear quantized status");
            check_cuda(cudaMemset(physical_rows.get(), 0, sizeof(unsigned long long)),
                       "clear quantized physical rows");
            paraselect::search::pq_adc_query_lut_segmented_scan(quantized_lut_pc, schedule.view, &tile_schedule.view, d_codes.get(),
                d_list_offsets.get(), d_list_counts.get(), d_thresholds.get(), kQueries,
                kProbes, kPqM, paraselect::search::PQAdcListScanRowMode::Serial,
                mode, physical_rows.get(), tiled_output, nullptr,
                paraselect::search::PQAdcSharedLutLayout::Compact);
            check_cuda(cudaDeviceSynchronize(), "finish quantized LUT scan");
            int count = 0;
            int status = 0;
            unsigned long long physical = 0;
            check_cuda(cudaMemcpy(&count, tiled_total.get(), sizeof(int),
                                  cudaMemcpyDeviceToHost), "copy quantized total");
            check_cuda(cudaMemcpy(&status, tiled_status.get(), sizeof(int),
                                  cudaMemcpyDeviceToHost), "copy quantized status");
            check_cuda(cudaMemcpy(&physical, physical_rows.get(),
                                  sizeof(unsigned long long), cudaMemcpyDeviceToHost),
                       "copy quantized physical rows");
            if (status != 0 || physical != kRows) {
                throw std::runtime_error("quantized LUT scan aggregate mismatch");
            }
            return copy_candidates(
                tiled_rows, tiled_distances, tiled_queries, count);
        };
        const auto quantized_global = run_quantized(
            paraselect::search::PQAdcLutReadMode::Global);
        const auto quantized_shared = run_quantized(
            paraselect::search::PQAdcLutReadMode::Shared);
        if (quantized_global.size() != quantized_shared.size()) {
            throw std::runtime_error("shared quantized LUT changed candidate count");
        }
        for (size_t i = 0; i < quantized_global.size(); ++i) {
            float global_distance = 0.0f, shared_distance = 0.0f;
            std::memcpy(&global_distance, &quantized_global[i].distance_bits, sizeof(float));
            std::memcpy(&shared_distance, &quantized_shared[i].distance_bits, sizeof(float));
            if (quantized_global[i].query != quantized_shared[i].query ||
                quantized_global[i].row != quantized_shared[i].row ||
                std::fabs(global_distance - shared_distance) > 1e-5f) {
                throw std::runtime_error("shared quantized LUT changed candidate triples");
            }
        }

        auto quantized_row_correction_u16 = quantized_lut_pc;
        quantized_row_correction_u16.query_lut_bias.fp32 = nullptr;
        quantized_row_correction_u16.query_lut_bias.u16 = d_row_correction_u16.get();
        quantized_row_correction_u16.query_lut_bias.scale = 0.125f;
        quantized_row_correction_u16.query_lut_bias.offset = 0.0f;
        check_cuda(cudaMemset(tiled_query_counts.get(), 0, kQueries * sizeof(int)),
                   "clear query-row tile counts");
        check_cuda(cudaMemset(tiled_total.get(), 0, sizeof(int)),
                   "clear query-row tile total");
        check_cuda(cudaMemset(tiled_status.get(), 0, sizeof(int)),
                   "clear query-row tile status");
        check_cuda(cudaMemset(physical_rows.get(), 0, sizeof(unsigned long long)),
                   "clear query-row tile physical rows");
        paraselect::search::pq_adc_query_lut_segmented_scan(quantized_row_correction_u16, schedule.view, nullptr, d_codes.get(),
            d_list_offsets.get(), d_list_counts.get(), d_thresholds.get(), kQueries,
            kProbes, kPqM,
            paraselect::search::PQAdcListScanRowMode::QueryRowTile4,
            paraselect::search::PQAdcLutReadMode::Shared,
            physical_rows.get(), tiled_output, nullptr,
            paraselect::search::PQAdcSharedLutLayout::Compact);
        check_cuda(cudaDeviceSynchronize(), "finish query-row tile scan");
        int query_row_count = 0;
        int query_row_status = 0;
        unsigned long long query_row_physical = 0;
        check_cuda(cudaMemcpy(&query_row_count, tiled_total.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy query-row total");
        check_cuda(cudaMemcpy(&query_row_status, tiled_status.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy query-row status");
        check_cuda(cudaMemcpy(&query_row_physical, physical_rows.get(),
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost),
                   "copy query-row physical rows");
        if (query_row_status != 0 || query_row_physical != kRows) {
            throw std::runtime_error(
                "query-row tile aggregate mismatch: status=" +
                std::to_string(query_row_status) + " physical=" +
                std::to_string(query_row_physical) + " candidates=" +
                std::to_string(query_row_count));
        }
        const auto query_row = copy_candidates(
            tiled_rows, tiled_distances, tiled_queries, query_row_count);
        if (query_row.size() != quantized_shared.size()) {
            throw std::runtime_error("query-row tile changed candidate count");
        }
        for (size_t i = 0; i < query_row.size(); ++i) {
            if (query_row[i].as_tuple() != quantized_shared[i].as_tuple()) {
                throw std::runtime_error("query-row tile changed candidate triples");
            }
        }

        check_cuda(cudaMemset(tiled_query_counts.get(), 0, kQueries * sizeof(int)),
                   "clear interleaved counts");
        check_cuda(cudaMemset(tiled_total.get(), 0, sizeof(int)),
                   "clear interleaved total");
        check_cuda(cudaMemset(tiled_status.get(), 0, sizeof(int)),
                   "clear interleaved status");
        check_cuda(cudaMemset(physical_rows.get(), 0, sizeof(unsigned long long)),
                   "clear interleaved physical rows");
        paraselect::search::pq_adc_query_lut_segmented_scan(query_lut_row_correction, schedule.view, &tile_schedule.view, d_codes.get(),
            d_list_offsets.get(), d_list_counts.get(), d_thresholds.get(), kQueries,
            kProbes, kPqM, paraselect::search::PQAdcListScanRowMode::Interleaved,
            paraselect::search::PQAdcLutReadMode::Shared,
            physical_rows.get(), tiled_output, nullptr);
        check_cuda(cudaDeviceSynchronize(), "finish interleaved row scan test");
        check_cuda(cudaMemcpy(&tiled_count, tiled_total.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy interleaved total");
        check_cuda(cudaMemcpy(&tiled_error, tiled_status.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy interleaved status");
        check_cuda(cudaMemcpy(&loaded_rows, physical_rows.get(), sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost), "copy interleaved physical rows");
        if (tiled_error != 0 || tiled_count != pair_count || loaded_rows != kRows) {
            throw std::runtime_error("interleaved row scan aggregate mismatch");
        }
        const auto interleaved = copy_candidates(
            tiled_rows, tiled_distances, tiled_queries, tiled_count);
        for (size_t i = 0; i < pair.size(); ++i) {
            if (pair[i].as_tuple() != interleaved[i].as_tuple()) {
                throw std::runtime_error("interleaved row scan changed candidate triples");
            }
        }

        const auto run_overflow = [&](int mode) {
            constexpr int kOverflowCapacity = 8;
            DeviceBuffer<int> rows(kOverflowCapacity);
            DeviceBuffer<float> distances(kOverflowCapacity);
            DeviceBuffer<int> queries(kOverflowCapacity);
            DeviceBuffer<int> query_counts(kQueries);
            DeviceBuffer<int> total(1);
            DeviceBuffer<int> status(1);
            check_cuda(cudaMemset(query_counts.get(), 0, kQueries * sizeof(int)),
                       "clear overflow query counts");
            check_cuda(cudaMemset(total.get(), 0, sizeof(int)), "clear overflow total");
            check_cuda(cudaMemset(status.get(), 0, sizeof(int)), "clear overflow status");
            const paraselect::search::PQAdcThresholdCandidatePoolView output{
                kOverflowCapacity, rows.get(), distances.get(), queries.get(),
                query_counts.get(), total.get(), status.get(), nullptr};
            if (mode != 0) {
                check_cuda(cudaMemset(physical_rows.get(), 0, sizeof(unsigned long long)),
                           "clear overflow physical rows");
                paraselect::search::pq_adc_query_lut_segmented_scan(query_lut_row_correction, schedule.view,
                    mode >= 2 ? &tile_schedule.view : nullptr, d_codes.get(),
                    d_list_offsets.get(), d_list_counts.get(), d_thresholds.get(),
                    kQueries, kProbes, kPqM,
                    mode == 3
                        ? paraselect::search::PQAdcListScanRowMode::Interleaved
                        : paraselect::search::PQAdcListScanRowMode::Serial,
                    paraselect::search::PQAdcLutReadMode::Shared,
                    physical_rows.get(), output, nullptr);
            } else {
                paraselect::search::pq_adc_query_lut_pair_scan(query_lut_row_correction, d_probes.get(), d_codes.get(), d_list_offsets.get(),
                    d_list_counts.get(), d_thresholds.get(), kQueries, kLists, kProbes,
                    kPqM, output, nullptr);
            }
            check_cuda(cudaDeviceSynchronize(), "finish overflow scan");
            int host_total = 0;
            int host_status = 0;
            std::vector<int> host_query_counts(kQueries);
            check_cuda(cudaMemcpy(&host_total, total.get(), sizeof(int), cudaMemcpyDeviceToHost),
                       "copy overflow total");
            check_cuda(cudaMemcpy(&host_status, status.get(), sizeof(int),
                                  cudaMemcpyDeviceToHost), "copy overflow status");
            check_cuda(cudaMemcpy(host_query_counts.data(), query_counts.get(),
                                  kQueries * sizeof(int), cudaMemcpyDeviceToHost),
                       "copy overflow query counts");
            return std::make_tuple(
                host_total,
                host_status,
                host_query_counts[0] + host_query_counts[1] + host_query_counts[2]);
        };
        const auto pair_overflow = run_overflow(0);
        const auto segmented_overflow = run_overflow(1);
        const auto tiled_overflow = run_overflow(2);
        const auto interleaved_overflow = run_overflow(3);
        if (std::get<0>(pair_overflow) != kExpectedCandidates ||
            std::get<1>(pair_overflow) != 1 || std::get<2>(pair_overflow) != 8 ||
            pair_overflow != segmented_overflow || pair_overflow != tiled_overflow ||
            pair_overflow != interleaved_overflow) {
            throw std::runtime_error("segmented scan changed global-pool overflow semantics");
        }

        std::vector<idx_t> invalid_probes = probes;
        invalid_probes[0] = kLists;
        check_cuda(cudaMemcpy(d_probes.get(), invalid_probes.data(),
                              invalid_probes.size() * sizeof(idx_t), cudaMemcpyHostToDevice),
                   "copy invalid probes");
        bool invalid_rejected = false;
        try {
            (void)schedule_builder.build(
                {d_probes.get(), kQueries, kProbes}, kProbes, nullptr);
        } catch (const std::out_of_range&) {
            invalid_rejected = true;
        }
        if (!invalid_rejected) {
            throw std::runtime_error("segmented schedule accepted an invalid list");
        }
        test_quantized_pair_and_list();
        test_quantized_pair_and_list(16);
        test_quantized_pair_and_list(64);
        std::puts("PASS pair/list/entry/probe-range scans M16/M20/M64 against CPU oracle");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[test_pq_query_lut_row_correction_segmented] %s\n", error.what());
        return 1;
    }
}
