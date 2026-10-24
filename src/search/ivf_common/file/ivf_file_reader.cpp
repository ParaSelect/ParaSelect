#include "search/ivf_common/file/ivf_file_reader.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <liburing.h>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace paraselect::search {
namespace {

constexpr size_t kNoSlot = std::numeric_limits<size_t>::max();
constexpr size_t kPrefillChunkBytes = 128ULL * 1024ULL * 1024ULL;

std::shared_ptr<void> allocate_aligned(size_t alignment, size_t bytes) {
    void* pointer = nullptr;
    if (::posix_memalign(&pointer, alignment, bytes) != 0 || !pointer) {
        throw std::bad_alloc();
    }
    return std::shared_ptr<void>(pointer, [](void* value) { std::free(value); });
}

void pread_exact(int fd, void* output, size_t bytes, off_t offset) {
    uint8_t* destination = static_cast<uint8_t*>(output);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t received = ::pread(
            fd, destination + done, bytes - done, offset + static_cast<off_t>(done));
        if (received < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("file rerank read failed: ") + std::strerror(errno));
        }
        if (received == 0) throw std::runtime_error("file rerank encountered a short read");
        done += static_cast<size_t>(received);
    }
}

}  // namespace

struct IVFFileReader::Impl {
    struct IoRing {
        IoRing(int fd, int queue_depth) {
            const int initialized = io_uring_queue_init(
                static_cast<unsigned>(queue_depth), &value, IORING_SETUP_IOPOLL);
            if (initialized < 0) {
                throw std::runtime_error(
                    std::string("file rerank io_uring initialization failed: ") +
                    std::strerror(-initialized));
            }
            initialized_ = true;
            const int registered = io_uring_register_files(&value, &fd, 1);
            if (registered < 0) {
                io_uring_queue_exit(&value);
                initialized_ = false;
                throw std::runtime_error(
                    std::string("file rerank io_uring file registration failed: ") +
                    std::strerror(-registered));
            }
            registered_ = true;
        }

        ~IoRing() {
            if (registered_) io_uring_unregister_files(&value);
            if (initialized_) io_uring_queue_exit(&value);
        }

        IoRing(const IoRing&) = delete;
        IoRing& operator=(const IoRing&) = delete;

        io_uring value{};

    private:
        bool initialized_ = false;
        bool registered_ = false;
    };

    struct Slot {
        uint64_t block = 0;
        size_t previous = kNoSlot;
        size_t next = kNoSlot;
        uint32_t pins = 0;
        bool used = false;
        bool ready = false;
    };

    explicit Impl(
        const IVFFileStorageView& requested_storage,
        const IVFFileReadConfig& requested_config)
        : storage(requested_storage), config(requested_config) {
        block_size = config.read_block_bytes;
        capacity_bytes = config.cache_capacity_bytes;
        if (storage.fd < 0 || storage.block_size == 0 || storage.row_stride == 0 ||
            block_size == 0 || block_size < storage.block_size ||
            block_size % storage.block_size != 0 ||
            block_size % storage.row_stride != 0 ||
            (capacity_bytes != 0 &&
             (capacity_bytes < block_size || capacity_bytes % block_size != 0))) {
            throw std::invalid_argument("file rerank read configuration is not block aligned");
        }
        if (config.ring_count <= 0 || config.ring_count > 64 ||
            config.queue_depth_per_ring <= 0 || config.queue_depth_per_ring > 4096) {
            throw std::invalid_argument("file rerank reader has an invalid ring configuration");
        }
        cache_enabled = capacity_bytes != 0;
        if (cache_enabled) {
            slot_count = capacity_bytes / block_size;
            slots.resize(slot_count);
            void* mapping = ::mmap(
                nullptr, capacity_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (mapping == MAP_FAILED) {
                throw std::runtime_error(
                    std::string("file rerank reader mmap failed: ") + std::strerror(errno));
            }
            data = static_cast<uint8_t*>(mapping);
            (void)::madvise(data, capacity_bytes, MADV_RANDOM);
            lookup.reserve(slot_count);
        }
        try {
            rings.reserve(config.ring_count);
            for (int ring = 0; ring < config.ring_count; ++ring) {
                rings.push_back(std::make_unique<IoRing>(
                    storage.fd, config.queue_depth_per_ring));
                validate_io(*rings.back());
            }
            if (cache_enabled) prefill();
        } catch (...) {
            if (data) {
                ::munmap(data, capacity_bytes);
                data = nullptr;
            }
            throw;
        }
    }

    void validate_io(IoRing& ring) {
        const auto validation_buffer = allocate_aligned(
            storage.block_size, storage.block_size);
        io_uring_sqe* sqe = io_uring_get_sqe(&ring.value);
        if (!sqe) throw std::runtime_error("file rerank IOPOLL validation queue is full");
        io_uring_prep_read(
            sqe,
            0,
            validation_buffer.get(),
            static_cast<unsigned>(storage.block_size),
            static_cast<off_t>(storage.payload_offset));
        sqe->flags |= IOSQE_FIXED_FILE;
        const int submitted = io_uring_submit_and_wait(&ring.value, 1);
        if (submitted < 0) {
            throw std::runtime_error(
                std::string("file rerank IOPOLL validation submit failed: ") +
                std::strerror(-submitted));
        }
        if (submitted != 1) {
            throw std::runtime_error("file rerank IOPOLL validation submitted no request");
        }
        io_uring_cqe* cqe = nullptr;
        const int wait_result = io_uring_wait_cqe(&ring.value, &cqe);
        if (wait_result < 0) {
            throw std::runtime_error(
                std::string("file rerank IOPOLL validation completion failed: ") +
                std::strerror(-wait_result));
        }
        const int result = cqe->res;
        io_uring_cqe_seen(&ring.value, cqe);
        if (result == -EOPNOTSUPP) {
            throw std::runtime_error(
                "file rerank IOPOLL is unsupported by the backing block device; "
                "configure a polling hardware queue before running the file backend");
        }
        if (result < 0) {
            throw std::runtime_error(
                std::string("file rerank IOPOLL validation read failed: ") +
                std::strerror(-result));
        }
        if (result != static_cast<int>(storage.block_size)) {
            throw std::runtime_error("file rerank IOPOLL validation encountered a short read");
        }
    }

    void prefill() {
        const auto begin = std::chrono::steady_clock::now();
        const size_t payload_bytes = static_cast<size_t>(storage.count) * storage.row_stride;
        const size_t bytes = std::min(capacity_bytes, payload_bytes);
        const size_t direct_bytes =
            (bytes + storage.block_size - 1) / storage.block_size * storage.block_size;
        for (size_t offset = 0; offset < direct_bytes; offset += kPrefillChunkBytes) {
            const size_t chunk = std::min(kPrefillChunkBytes, direct_bytes - offset);
            pread_exact(
                storage.fd,
                data + offset,
                chunk,
                static_cast<off_t>(storage.payload_offset + offset));
        }
        const size_t blocks = (bytes + block_size - 1) / block_size;
        for (size_t block = 0; block < blocks; ++block) {
            Slot& slot = slots[block];
            slot.block = block;
            slot.used = true;
            slot.ready = true;
            lookup.emplace(block, block);
            link_front(block);
        }
        next_unused = blocks;
        counters.prefill_bytes = direct_bytes;
        counters.prefill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    }

    ~Impl() {
        if (data) ::munmap(data, capacity_bytes);
    }

    void unlink(size_t slot) {
        Slot& value = slots[slot];
        if (value.previous != kNoSlot) slots[value.previous].next = value.next;
        if (value.next != kNoSlot) slots[value.next].previous = value.previous;
        if (head == slot) head = value.next;
        if (tail == slot) tail = value.previous;
        value.previous = value.next = kNoSlot;
    }

    void link_front(size_t slot) {
        Slot& value = slots[slot];
        value.previous = kNoSlot;
        value.next = head;
        if (head != kNoSlot) slots[head].previous = slot;
        head = slot;
        if (tail == kNoSlot) tail = slot;
    }

    void touch(size_t slot) {
        if (head == slot) return;
        unlink(slot);
        link_front(slot);
    }

    size_t allocate_slot() {
        if (!free_slots.empty()) {
            const size_t slot = free_slots.back();
            free_slots.pop_back();
            return slot;
        }
        if (next_unused < slot_count) return next_unused++;
        size_t slot = tail;
        while (slot != kNoSlot && slots[slot].pins != 0) slot = slots[slot].previous;
        if (slot == kNoSlot) {
            throw std::runtime_error("file rerank reader has no evictable block");
        }
        lookup.erase(slots[slot].block);
        unlink(slot);
        return slot;
    }

    IVFFileStorageView storage;
    IVFFileReadConfig config;
    size_t block_size = 0;
    size_t capacity_bytes = 0;
    bool cache_enabled = false;
    size_t slot_count = 0;
    uint8_t* data = nullptr;
    std::vector<Slot> slots;
    std::vector<size_t> free_slots;
    std::unordered_map<uint64_t, size_t> lookup;
    size_t next_unused = 0;
    size_t head = kNoSlot;
    size_t tail = kNoSlot;
    Stats counters{0, 0, 0, 0, 0, 0.0, capacity_bytes};
    mutable std::mutex mutex;
    std::vector<std::unique_ptr<IoRing>> rings;
};

IVFFileReader::Block::Block(
    IVFFileReader* owner, size_t slot, uint8_t* data, bool reused)
    : owner_(owner), slot_(slot), data_(data), reused_(reused) {}

IVFFileReader::Block::Block(
    uint8_t* data,
    bool reused,
    std::shared_ptr<void> owned_data)
    : data_(data), reused_(reused), owned_data_(std::move(owned_data)) {}

IVFFileReader::Block::~Block() { release(); }

IVFFileReader::Block::Block(Block&& other) noexcept { *this = std::move(other); }

IVFFileReader::Block& IVFFileReader::Block::operator=(Block&& other) noexcept {
    if (this == &other) return *this;
    release();
    owner_ = other.owner_;
    slot_ = other.slot_;
    data_ = other.data_;
    reused_ = other.reused_;
    owned_data_ = std::move(other.owned_data_);
    other.owner_ = nullptr;
    other.data_ = nullptr;
    return *this;
}

void IVFFileReader::Block::release() noexcept {
    if (!owner_) return;
    owner_->release(slot_);
    owner_ = nullptr;
    data_ = nullptr;
    owned_data_.reset();
}

IVFFileReader::IVFFileReader(
    const IVFFileStorageView& storage,
    const IVFFileReadConfig& config)
    : impl_(std::make_unique<Impl>(storage, config)) {}

IVFFileReader::~IVFFileReader() = default;

bool IVFFileReader::contains(uint64_t block) const {
    if (!impl_->cache_enabled) return false;
    std::lock_guard<std::mutex> guard(impl_->mutex);
    const auto found = impl_->lookup.find(block);
    return found != impl_->lookup.end() && impl_->slots[found->second].ready;
}

std::vector<uint8_t> IVFFileReader::contains_many(
    const std::vector<uint64_t>& blocks) const {
    std::vector<uint8_t> cached(blocks.size(), 0);
    if (!impl_->cache_enabled) return cached;
    std::lock_guard<std::mutex> guard(impl_->mutex);
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto found = impl_->lookup.find(blocks[i]);
        cached[i] = static_cast<uint8_t>(
            found != impl_->lookup.end() && impl_->slots[found->second].ready);
    }
    return cached;
}

std::vector<IVFFileReader::Block> IVFFileReader::read(
    const std::vector<Request>& requests) {
    struct Miss {
        size_t request = 0;
        size_t slot = 0;
    };

    std::vector<Block> blocks;
    blocks.reserve(requests.size());
    std::vector<Miss> misses;
    misses.reserve(requests.size());
    if (!impl_->cache_enabled) {
        uint64_t bytes_read = 0;
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            const Request& request = requests[request_index];
            if (request.bytes == 0 || request.bytes > impl_->block_size) {
                throw std::invalid_argument("file rerank request has invalid byte count");
            }
            auto owned = allocate_aligned(impl_->storage.block_size, impl_->block_size);
            blocks.push_back(Block(
                static_cast<uint8_t*>(owned.get()), false, std::move(owned)));
            misses.push_back({request_index, kNoSlot});
            bytes_read += request.bytes;
        }
        {
            std::lock_guard<std::mutex> guard(impl_->mutex);
            impl_->counters.misses += requests.size();
            impl_->counters.bytes_read += bytes_read;
        }
    } else {
        std::lock_guard<std::mutex> guard(impl_->mutex);
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            const Request& request = requests[request_index];
            if (request.bytes == 0 || request.bytes > impl_->block_size) {
                throw std::invalid_argument("file rerank request has invalid byte count");
            }
            const auto found = impl_->lookup.find(request.block);
            if (found != impl_->lookup.end()) {
                Impl::Slot& slot = impl_->slots[found->second];
                if (!slot.ready) {
                    throw std::invalid_argument("file rerank batch contains a duplicate block");
                }
                ++slot.pins;
                impl_->touch(found->second);
                ++impl_->counters.hits;
                impl_->counters.bytes_reused += request.bytes;
                blocks.push_back(Block(
                    this,
                    found->second,
                    impl_->data + found->second * impl_->block_size,
                    true));
                continue;
            }

            const size_t slot_index = impl_->allocate_slot();
            Impl::Slot& slot = impl_->slots[slot_index];
            slot.block = request.block;
            slot.pins = 1;
            slot.used = true;
            slot.ready = false;
            impl_->lookup.emplace(request.block, slot_index);
            impl_->link_front(slot_index);
            ++impl_->counters.misses;
            impl_->counters.bytes_read += request.bytes;
            blocks.push_back(Block(
                this,
                slot_index,
                impl_->data + slot_index * impl_->block_size,
                false));
            misses.push_back({request_index, slot_index});
        }
    }

    const size_t ring_count = impl_->rings.size();
    const size_t queue_depth = static_cast<size_t>(impl_->config.queue_depth_per_ring);
    std::vector<std::vector<size_t>> assigned(ring_count);
    for (size_t miss = 0; miss < misses.size(); ++miss) {
        assigned[miss % ring_count].push_back(miss);
    }
    std::vector<std::vector<size_t>> completed(ring_count);
    std::vector<std::exception_ptr> errors(ring_count);
    std::vector<std::thread> workers;
    workers.reserve(ring_count);

    for (size_t ring_index = 0; ring_index < ring_count; ++ring_index) {
        if (assigned[ring_index].empty()) continue;
        workers.emplace_back([&, ring_index]() {
            try {
                io_uring& ring = impl_->rings[ring_index]->value;
                const auto& ring_misses = assigned[ring_index];
                auto& ring_completed = completed[ring_index];
                ring_completed.reserve(ring_misses.size());
                size_t next = 0;
                size_t in_flight = 0;

                auto submit_available = [&]() {
                    size_t prepared = 0;
                    while (next < ring_misses.size() &&
                           in_flight + prepared < queue_depth) {
                        const size_t miss_index = ring_misses[next++];
                        const Miss& miss = misses[miss_index];
                        const Request& request = requests[miss.request];
                        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                        if (!sqe) {
                            throw std::runtime_error(
                                "file rerank io_uring submission queue is full");
                        }
                        io_uring_prep_read(
                            sqe,
                            0,
                            blocks[miss.request].data(),
                            static_cast<unsigned>(request.bytes),
                            static_cast<off_t>(impl_->storage.payload_offset +
                                request.block * impl_->block_size));
                        sqe->flags |= IOSQE_FIXED_FILE;
                        io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(miss_index));
                        ++prepared;
                    }
                    if (prepared == 0) return;
                    const int submitted = io_uring_submit(&ring);
                    if (submitted < 0) {
                        throw std::runtime_error(
                            std::string("file rerank io_uring submit failed: ") +
                            std::strerror(-submitted));
                    }
                    if (submitted != static_cast<int>(prepared)) {
                        throw std::runtime_error(
                            "file rerank io_uring submitted an incomplete window");
                    }
                    in_flight += prepared;
                };

                auto consume_completion = [&](io_uring_cqe* cqe) {
                    --in_flight;
                    const size_t miss_index = reinterpret_cast<size_t>(io_uring_cqe_get_data(cqe));
                    const int result = cqe->res;
                    io_uring_cqe_seen(&ring, cqe);
                    if (miss_index >= misses.size()) {
                        throw std::runtime_error(
                            "file rerank received an unknown completion");
                    }
                    const Miss& miss = misses[miss_index];
                    const int expected = static_cast<int>(requests[miss.request].bytes);
                    if (result < 0) {
                        throw std::runtime_error(
                            std::string("file rerank io_uring read failed: ") +
                            std::strerror(-result));
                    }
                    if (result != expected) {
                        throw std::runtime_error(
                            "file rerank io_uring encountered a short read");
                    }
                    ring_completed.push_back(miss.slot);
                };

                submit_available();
                while (in_flight > 0) {
                    io_uring_cqe* cqe = nullptr;
                    const int waited = io_uring_wait_cqe(&ring, &cqe);
                    if (waited < 0) {
                        throw std::runtime_error(
                            std::string("file rerank io_uring completion failed: ") +
                            std::strerror(-waited));
                    }
                    consume_completion(cqe);
                    while (in_flight > 0) {
                        cqe = nullptr;
                        const int peeked = io_uring_peek_cqe(&ring, &cqe);
                        if (peeked == -EAGAIN || !cqe) break;
                        if (peeked < 0) {
                            throw std::runtime_error(
                                std::string("file rerank io_uring completion peek failed: ") +
                                std::strerror(-peeked));
                        }
                        consume_completion(cqe);
                    }
                    submit_available();
                }
            } catch (...) {
                errors[ring_index] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers) worker.join();

    std::exception_ptr read_error;
    {
        std::lock_guard<std::mutex> guard(impl_->mutex);
        if (impl_->cache_enabled) {
            for (const auto& ring_completed : completed) {
                for (size_t slot : ring_completed) impl_->slots[slot].ready = true;
            }
        }
        for (const auto& error : errors) {
            if (error && !read_error) read_error = error;
        }
    }

    if (read_error && impl_->cache_enabled) {
        std::lock_guard<std::mutex> guard(impl_->mutex);
        for (const Miss& miss : misses) {
            Impl::Slot& slot = impl_->slots[miss.slot];
            if (!slot.ready) {
                impl_->lookup.erase(slot.block);
                impl_->unlink(miss.slot);
                slot.pins = 0;
                slot.used = false;
                impl_->free_slots.push_back(miss.slot);
            }
        }
    }
    if (read_error) std::rethrow_exception(read_error);

    return blocks;
}

IVFFileReader::Stats IVFFileReader::stats() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->counters;
}

size_t IVFFileReader::capacity_blocks() const noexcept {
    return impl_->slot_count;
}

size_t IVFFileReader::block_size() const noexcept {
    return impl_->block_size;
}

int IVFFileReader::ring_count() const noexcept {
    return impl_->config.ring_count;
}

int IVFFileReader::queue_depth_per_ring() const noexcept {
    return impl_->config.queue_depth_per_ring;
}

void IVFFileReader::release(size_t slot_index) noexcept {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    Impl::Slot& slot = impl_->slots[slot_index];
    if (slot.pins > 0) --slot.pins;
}

std::shared_ptr<IVFFileReader> make_ivf_file_reader(
    const IVFFileStorageView& storage,
    const IVFFileReadConfig& config) {
    return std::make_shared<IVFFileReader>(storage, config);
}

}  // namespace paraselect::search
