/**
 * safetensors.hpp — native safetensors reader (no transformers, no torch).
 *
 * Reads a model.safetensors file: the JSON header + the raw bytes of each
 * tensor. Supports the dtypes used by the Winnex stack:
 *   - F16 / BF16 (raw half/bfloat16 bytes)
 *   - F32
 *   - I32 (int32, used for GPTQ-packed qweight/qzeros/g_idx)
 *
 * This is the structural read the X-factor and the forward pass need —
 * it does NOT load the model into memory or run anything.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_SAFETENSORS_HPP
#define WINNEX_NANO_SAFETENSORS_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace winnex_nano {

enum class TensorDtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I32 = 3,
    U8 = 4,
    Unknown = 99,
};

// A loaded tensor: shape, dtype, and the raw bytes (offset into the file data).
struct SafeTensor {
    std::string name;
    TensorDtype dtype = TensorDtype::Unknown;
    std::vector<int64_t> shape;
    std::vector<int64_t> data_offsets;  // [begin, end) into the shared data blob
    // Which shard's payload data_offsets index into. 0 for a single file.
    int shard = 0;
};

// Immutable view over a .safetensors file.
//
// TWO OWNERSHIP MODES, the caller picks:
//
//   Safetensors(path)            copies the payload into an owned vector.
//                                Simple, safe, and wrong at scale: Qwen2.5-7B
//                                is 14.2 GB resident before any weight is used.
//
//   Safetensors::view(ptr, len)  takes a POINTER to memory the caller owns and
//                                never copies. This is the MadhavaL2 contract —
//                                "the caller owns the data pointer; the engine
//                                never copies the corpus" — applied to weights.
//
// The view mode is what makes mmap possible: the Python side opens an
// np.memmap over the payload and passes its address; the kernel pages it in on
// access and evicts clean pages under pressure, so RSS tracks the working set
// (a few layers) instead of the file size.
//
// The payload in view mode MUST outlive this object. That is the same contract
// MadhavaL2::build() has with its raw_base.
class Safetensors {
public:
    // Copies the payload into memory (the historical behaviour).
    explicit Safetensors(const std::string& path);
    Safetensors() = default;      // empty; used internally by view()

    // Non-owning view. `data` points at the payload bytes (i.e. the file
    // content AFTER the 8-byte header length + header JSON); `len` is the
    // payload length. `header_json` is the header text. Neither is copied.
    static Safetensors view(const uint8_t* data, size_t len,
                            const std::string& header_json);

    // Reads ONLY the 8-byte header length + header JSON from `path` — a few KB.
    // Use this to learn the payload offset before mapping it on the caller side.
    static std::string read_header(const std::string& path, uint64_t* payload_offset);

    // Adds another shard to this view. `data`/`len` are that shard's payload
    // bytes, borrowed exactly like view(). Tensor names must not collide.
    //
    // Why: hub checkpoints ship as model-00001-of-0000N.safetensors and the
    // index (model.safetensors.index.json) maps each tensor to its shard.
    // Refusing to open those meant every caller had to concatenate 15 GB of
    // files before the engine would run — the Kaggle notebook among them.
    void add_shard(const uint8_t* data, size_t len, const std::string& header_json);

    // Payload base of one shard (0 when there is only the primary payload).
    const uint8_t* shard_data(int shard) const {
        return shard == 0 ? data_ptr_ : shards_[(size_t)shard - 1].ptr;
    }
    int shard_count() const { return 1 + (int)shards_.size(); }

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const SafeTensor& get(const SafeTensor& t) const { return t; }
    const SafeTensor& get(const std::string& name) const;
    const std::map<std::string, SafeTensor>& tensors() const { return tensors_; }
    size_t count() const { return tensors_.size(); }

    // Base pointer of the payload (owned or borrowed — check owns_data()).
    const uint8_t* data() const { return data_ptr_; }
    size_t data_size() const { return data_size_; }
    bool owns_data() const { return owned_.size() > 0; }

    // Convenience: reinterpret the tensor bytes as float (converts f16/bf16/f32).
    // For I32 (GPTQ-packed) the bytes are returned as-is (use as int32).
    // Returns a new vector (the caller owns it).
    std::vector<float> to_float(const SafeTensor& t) const;
    std::vector<int32_t> to_int32(const SafeTensor& t) const;

    // Row-bounded read: converts ONLY rows [row_start, row_start+n_rows) of a
    // 2D tensor, leaving the full tensor unmaterialized on the Python side.
    std::vector<float> read_rows(const SafeTensor& t, int row_start, int n_rows) const;

private:
    void parse_header(const std::string& header, int shard = 0);
    std::vector<uint8_t> owned_;      // empty in view mode
    const uint8_t* data_ptr_ = nullptr;
    size_t data_size_ = 0;
    struct ShardRef { const uint8_t* ptr; size_t len; };
    std::vector<ShardRef> shards_;    // shards 1..N-1 (0 is data_ptr_)
    std::map<std::string, SafeTensor> tensors_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_SAFETENSORS_HPP
