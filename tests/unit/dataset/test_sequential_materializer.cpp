#include "utils/types.h"
#include "dataset/assignment_materializer.cuh"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>

namespace {
struct Fixture {
    std::filesystem::path root;
    Fixture() {
        char pattern[]="/tmp/paraselect-materializer-XXXXXX";
        char* directory=::mkdtemp(pattern);
        if(!directory) throw std::runtime_error("mkdtemp failed");
        root=directory;
        std::filesystem::create_directory(root/"workspace");
        ::setenv("PARASELECT_TMPDIR",(root/"workspace").c_str(),1);
    }
    ~Fixture() { std::error_code error;std::filesystem::remove_all(root,error); }
};

template<class Input> void verify(Fixture& fixture,const char* name,const char* env,const char* extension) {
    using Output=std::conditional_t<std::is_same_v<Input,float>,float,uint8_t>;
    using namespace paraselect::dataset;
    constexpr uint32_t rows=65539;
    auto spec=describe_dataset(name,"1M");spec.n_vectors=rows;
    const uint32_t dim=static_cast<uint32_t>(spec.dim);
    const auto path=fixture.root/(std::string(name)+extension);
    std::vector<Input> input(size_t(rows)*dim);
    for(uint32_t row=0;row<rows;++row) for(uint32_t d=0;d<dim;++d) {
        const size_t index=size_t(row)*dim+d;
        if constexpr(std::is_same_v<Input,float>) input[index]=float(index)/8.f;
        else {
            const int value=int((row*17+d*31+(row>>8)*13)%256);
            input[index]=static_cast<Input>(std::is_same_v<Input,int8_t> ? value-128 : value);
        }
    }
    {
        std::ofstream file(path,std::ios::binary);
        file.write(reinterpret_cast<const char*>(&rows),sizeof(rows));
        file.write(reinterpret_cast<const char*>(&dim),sizeof(dim));
        file.write(reinterpret_cast<const char*>(input.data()),input.size()*sizeof(Input));
        if(!file) throw std::runtime_error("fixture write failed");
    }
    ::setenv(env,path.c_str(),1);
    std::vector<idx_t> permutation(rows);
    for(uint32_t row=0;row<rows;++row) permutation[row]=(uint64_t(row)*37)%rows;
    const DatasetSource source(spec);
    const auto result=detail::materialize_reordered_sequential_impl<Output>(
        source,permutation.data(),rows,dim,spec.dtype);
    paraselect::ownership::BufferHandle output(result.data,size_t(rows)*dim*sizeof(Output),result.owner);
    for(uint32_t row=0;row<rows;++row) for(uint32_t d=0;d<dim;++d) {
        const Input value=input[size_t(permutation[row])*dim+d];
        const Output expected=std::is_same_v<Input,int8_t>
            ? static_cast<Output>(int(value)+128) : static_cast<Output>(value);
        if(result.data[size_t(row)*dim+d]!=expected)
            throw std::runtime_error(std::string(name)+" source-to-row mismatch");
    }
    std::vector<int> row_order(rows);
    for(uint32_t row=0;row<rows;++row)row_order[row]=rows-1-row;
    const auto sorted=detail::materialize_reordered_sequential_impl<Output>(
        source,permutation.data(),rows,dim,spec.dtype,row_order.data());
    paraselect::ownership::BufferHandle sorted_output(sorted.data,size_t(rows)*dim*sizeof(Output),sorted.owner);
    for(uint32_t row=0;row<rows;++row)for(uint32_t d=0;d<dim;++d)
        if(sorted.data[size_t(row)*dim+d]!=result.data[size_t(row_order[row])*dim+d])
            throw std::runtime_error("scalar materialization composition mismatch");
    for(bool duplicate:{false,true}) {
        auto invalid_order=row_order;
        invalid_order[1]=duplicate ? invalid_order[0] : int(rows);
        bool order_rejected=false;
        try {
            (void)detail::materialize_reordered_sequential_impl<Output>(
                source,permutation.data(),rows,dim,spec.dtype,invalid_order.data());
        } catch(const std::invalid_argument&) { order_rejected=true; }
        if(!order_rejected)throw std::runtime_error("invalid scalar order accepted");

        auto invalid=permutation;invalid[1]=duplicate ? invalid[0] : rows;
        bool rejected=false;
        try {
            (void)detail::materialize_reordered_sequential_impl<Output>(
                source,invalid.data(),rows,dim,spec.dtype);
        } catch(const std::invalid_argument&) { rejected=true; }
        if(!rejected) throw std::runtime_error("invalid permutation accepted");
    }
    if(!std::filesystem::is_empty(fixture.root/"workspace"))
        throw std::runtime_error("temporary inverse mapping leaked");
}
} // namespace

int main() try {
    Fixture fixture;
    ::setenv("PARASELECT_REORDER_PREFAULT","invalid",1);
    bool rejected=false;
    try { verify<float>(fixture,"deep","PARASELECT_DEEP_BASE",".fbin"); }
    catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("invalid prefault option accepted");
    for(const char* prefault:{"0","1"}) {
        ::setenv("PARASELECT_REORDER_PREFAULT",prefault,1);
        verify<float>(fixture,"deep","PARASELECT_DEEP_BASE",".fbin");
        verify<uint8_t>(fixture,"sift","PARASELECT_SIFT_BASE",".u8bin");
        verify<int8_t>(fixture,"spacev","PARASELECT_SPACEV_BASE",".i8bin");
    }
    std::cout<<"PASS FP32/U8/I8 reorder across 65536-row read boundary, invalid permutations and cleanup\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;
}
