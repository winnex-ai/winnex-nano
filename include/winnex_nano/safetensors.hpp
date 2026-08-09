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
};

class Safetensors {
public:
    // Loads a .safetensors file: parses the header, keeps the data blob.
    // The blob stays in memory (mmap-friendly: the file is read once).
    explicit Safetensors(const std::string& path);

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const SafeTensor& get(const std::string& name) const;
    const std::vector<uint8_t>& data() const { return data_; }
    const std::map<std::string, SafeTensor>& tensors() const { return tensors_; }
    size_t count() const { return tensors_.size(); }

    // Convenience: reinterpret the tensor bytes as float (converts f16/bf16/f32).
    // For I32 (GPTQ-packed) the bytes are returned as-is (use as int32).
    // Returns a new vector (the caller owns it).
    std::vector<float> to_float(const SafeTensor& t) const;
    std::vector<int32_t> to_int32(const SafeTensor& t) const;

private:
    std::vector<uint8_t> data_;
    std::map<std::string, SafeTensor> tensors_;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_SAFETENSORS_HPP
