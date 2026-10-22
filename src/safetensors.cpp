// safetensors.cpp — native safetensors reader.
#include "winnex_nano/safetensors.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace winnex_nano {

namespace {

// Converts a dtype string from the safetensors header to TensorDtype.
TensorDtype parse_dtype(const std::string& s) {
    if (s == "F32") return TensorDtype::F32;
    if (s == "F16") return TensorDtype::F16;
    if (s == "BF16") return TensorDtype::BF16;
    if (s == "I32") return TensorDtype::I32;
    if (s == "U8") return TensorDtype::U8;
    return TensorDtype::Unknown;
}

// Minimal JSON parse for the safetensors header. We only need: object of
// tensors, each {dtype, shape, data_offsets}. Handles nested arrays/ints.
// This is a tiny purpose-built parser (the header format is fixed).
struct Json {
    std::string s;
    size_t p = 0;
    explicit Json(const std::string& src) : s(src) {}

    void ws() { while (p < s.size() && (s[p]==' '||s[p]=='\n'||s[p]=='\t'||s[p]=='\r')) p++; }
    bool more() { ws(); return p < s.size(); }
    char peek() { ws(); return p < s.size() ? s[p] : '\0'; }
    char next() { ws(); return p < s.size() ? s[p++] : '\0'; }

    // Parses a JSON value into a generic variant-ish via callbacks.
    void parse_object() {
        next();  // '{'
        while (more() && peek() != '}') {
            std::string key = parse_string();
            next();  // ':'
            parse_value(key);
            if (peek() == ',') next();
        }
        next();  // '}'
    }
    virtual void on_key_string(const std::string&, const std::string&) {}
    virtual void on_key_int(const std::string&, int64_t) {}
    virtual void on_key_array(const std::string&, const std::vector<int64_t>&) {}
    std::string parse_string() {
        next();  // '"'
        std::string out;
        while (p < s.size() && s[p] != '"') {
            if (s[p] == '\\' && p + 1 < s.size()) { out += s[p+1]; p += 2; }
            else out += s[p++];
        }
        next();  // '"'
        return out;
    }
    void parse_value(const std::string& key) {
        char c = peek();
        if (c == '{') parse_object();
        else if (c == '"') on_key_string(key, parse_string());
        else if (c == '[') {
            std::vector<int64_t> arr;
            next();  // '['
            while (more() && peek() != ']') {
                if (peek() == ',') { next(); continue; }
                arr.push_back(parse_int());
            }
            next();  // ']'
            on_key_array(key, arr);
        }
        else on_key_int(key, parse_int());
    }
    int64_t parse_int() {
        ws(); bool neg = false;
        if (peek() == '-') { neg = true; next(); }
        int64_t v = 0;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') { v = v*10 + (s[p]-'0'); p++; }
        return neg ? -v : v;
    }
};

} // namespace

Safetensors::Safetensors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Safetensors: cannot open " + path);

    // Header length (uint64 little-endian).
    uint64_t header_len = 0;
    f.read((char*)&header_len, 8);
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // Data blob (the rest of the file) — COPIED into owned_.
    //
    // This is the safe mode and the wrong one at scale: 14.2 GB for Qwen2.5-7B
    // before a single weight is used. Callers that can mmap should use
    // Safetensors::view() instead, which borrows the pointer and never copies
    // (the MadhavaL2 contract).
    f.seekg(0, std::ios::end);
    std::streampos end = f.tellg();
    size_t data_size = (size_t)end - 8 - (size_t)header_len;
    owned_.resize(data_size);
    f.seekg(8 + (std::streampos)header_len, std::ios::beg);
    f.read((char*)owned_.data(), data_size);

    data_ptr_ = owned_.data();
    data_size_ = data_size;
    parse_header(header);
}

// The header parse, factored out so the copy mode and the view mode share it.
void Safetensors::add_shard(const uint8_t* data, size_t len, const std::string& header_json) {
    shards_.push_back({data, len});
    parse_header(header_json, (int)shards_.size());   // shard 0 is the primary
}

void Safetensors::parse_header(const std::string& header, int shard) {
    size_t p = 0;
    while (p < header.size() && header[p] != '{') p++;
    p++;
    while (p < header.size() && header[p] != '}') {
        while (p < header.size() && header[p] != '"') p++;
        if (p >= header.size()) break;
        p++;
        std::string name;
        while (p < header.size() && header[p] != '"') { name += header[p]; p++; }
        p++;
        while (p < header.size() && header[p] != ':') p++;
        p++;
        if (name == "__metadata__") {
            int depth = 0;
            while (p < header.size()) {
                if (header[p] == '{') depth++;
                else if (header[p] == '}') { depth--; if (depth == 0) { p++; break; } }
                p++;
            }
            continue;
        }
        SafeTensor t;
        t.name = name;
        int depth = 0;
        std::string obj;
        while (p < header.size()) {
            if (header[p] == '{') depth++;
            else if (header[p] == '}') { depth--; if (depth == 0) { p++; break; } }
            obj += header[p];
            p++;
        }
        auto find_field = [&](const std::string& field) -> std::string {
            size_t pos = obj.find("\"" + field + "\"");
            if (pos == std::string::npos) return "";
            pos = obj.find(':', pos);
            if (pos == std::string::npos) return "";
            pos++;
            while (pos < obj.size() && (obj[pos]==' '||obj[pos]=='\n'||obj[pos]=='\t')) pos++;
            if (obj[pos] == '"') {
                pos++;
                std::string v;
                while (pos < obj.size() && obj[pos] != '"') { v += obj[pos]; pos++; }
                return v;
            }
            if (obj[pos] == '[') {
                std::string v;
                while (pos < obj.size() && obj[pos] != ']') { v += obj[pos]; pos++; }
                v += ']';
                return v;
            }
            return "";
        };
        auto parse_array = [&](const std::string& str) -> std::vector<int64_t> {
            std::vector<int64_t> out;
            size_t q = str.find('[');
            if (q == std::string::npos) return out;
            q++;
            while (q < str.size()) {
                while (q < str.size() && (str[q]<'0'||str[q]>'9') && str[q] != '-') q++;
                if (q >= str.size()) break;
                bool neg = false;
                if (str[q] == '-') { neg = true; q++; }
                int64_t v = 0;
                while (q < str.size() && str[q] >= '0' && str[q] <= '9') { v = v*10 + (str[q]-'0'); q++; }
                out.push_back(neg ? -v : v);
                while (q < str.size() && str[q] != ',' && str[q] != ']') q++;
                if (q < str.size() && str[q] == ']') break;
                q++;
            }
            return out;
        };
        t.dtype = parse_dtype(find_field("dtype"));
        t.shape = parse_array(find_field("shape"));
        auto offs = parse_array(find_field("data_offsets"));
        if (offs.size() >= 2) t.data_offsets = {offs[0], offs[1]};
        t.shard = shard;
        // First shard to declare a name wins. Checkpoints sharded by the hub
        // never duplicate a tensor, but a silently overwritten entry would be
        // a very confusing failure to debug, so this keeps it deterministic.
        if (tensors_.count(name) == 0) tensors_[name] = t;
        if (p < header.size() && header[p] == ',') p++;
    }
}

// Non-owning view: the caller keeps the payload alive (MadhavaL2's contract).
Safetensors Safetensors::view(const uint8_t* data, size_t len,
                              const std::string& header_json) {
    Safetensors s;
    s.data_ptr_ = data;
    s.data_size_ = len;
    s.parse_header(header_json);
    return s;
}

// Reads ONLY the 8-byte length + header JSON (a few KB), leaving the payload
// on disk so the caller can mmap it.
std::string Safetensors::read_header(const std::string& path, uint64_t* payload_offset) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Safetensors: cannot open " + path);
    uint64_t header_len = 0;
    f.read((char*)&header_len, 8);
    if (header_len == 0 || header_len > (1u << 28))
        throw std::runtime_error("Safetensors: implausible header length");
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    if (payload_offset) *payload_offset = 8 + header_len;
    return header;
}

const SafeTensor& Safetensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Safetensors: tensor not found: " + name);
    }
    return it->second;
}

namespace {
inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline float bf16_to_float(uint16_t b) {
    uint32_t f = ((uint32_t)b) << 16;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}
} // namespace

std::vector<float> Safetensors::to_float(const SafeTensor& t) const {
    size_t n = 1;
    for (auto s : t.shape) n *= (size_t)s;
    std::vector<float> out(n);
    size_t bytes_per = (t.dtype == TensorDtype::F32) ? 4 : 2;
    const uint8_t* base = shard_data(t.shard) + (size_t)t.data_offsets[0];
    if (t.dtype == TensorDtype::F32) {
        std::memcpy(out.data(), base, n * 4);
    } else if (t.dtype == TensorDtype::F16) {
        for (size_t i = 0; i < n; ++i) {
            uint16_t h; std::memcpy(&h, base + i*2, 2);
            out[i] = half_to_float(h);
        }
    } else if (t.dtype == TensorDtype::BF16) {
        for (size_t i = 0; i < n; ++i) {
            uint16_t b; std::memcpy(&b, base + i*2, 2);
            out[i] = bf16_to_float(b);
        }
    } else {
        throw std::runtime_error("Safetensors: to_float on non-float tensor " + t.name);
    }
    return out;
}

std::vector<int32_t> Safetensors::to_int32(const SafeTensor& t) const {
    size_t n = 1;
    for (auto s : t.shape) n *= (size_t)s;
    std::vector<int32_t> out(n);
    const uint8_t* base = shard_data(t.shard) + (size_t)t.data_offsets[0];
    if (t.dtype == TensorDtype::I32) {
        std::memcpy(out.data(), base, n * 4);
    } else {
        throw std::runtime_error("Safetensors: to_int32 on non-I32 tensor " + t.name);
    }
    return out;
}


// Row-bounded read of a 2D tensor. Converts ONLY the requested rows.
//
// Why this exists: materialising embed_tokens for Qwen2.5-7B is
// 152.064 x 3.584 = 2.2 GB in float32. A benchmark that reads ~110 token
// embeddings should not pay that. This converts row by row from the source
// dtype, so peak memory is (n_rows x cols x 4) instead of the full matrix.
std::vector<float> Safetensors::read_rows(const SafeTensor& t, int row_start, int n_rows) const {
    if (t.shape.size() != 2)
        throw std::runtime_error("read_rows: tensor must be 2D");
    const int rows = (int)t.shape[0];
    const int cols = (int)t.shape[1];
    if (row_start < 0 || n_rows < 0 || row_start + n_rows > rows)
        throw std::runtime_error("read_rows: out of range");

    std::vector<float> out((size_t)n_rows * cols);
    const uint8_t* base = shard_data(t.shard) + (size_t)t.data_offsets[0];
    const size_t row_bytes = (t.dtype == TensorDtype::F32) ? (size_t)cols * 4
                                                           : (size_t)cols * 2;
    for (int r = 0; r < n_rows; ++r) {
        const uint8_t* src = base + (size_t)(row_start + r) * row_bytes;
        float* dst = out.data() + (size_t)r * cols;
        if (t.dtype == TensorDtype::F32) {
            std::memcpy(dst, src, (size_t)cols * 4);
        } else if (t.dtype == TensorDtype::F16) {
            for (int c = 0; c < cols; ++c) {
                uint16_t h; std::memcpy(&h, src + (size_t)c * 2, 2);
                dst[c] = half_to_float(h);
            }
        } else if (t.dtype == TensorDtype::BF16) {
            for (int c = 0; c < cols; ++c) {
                uint16_t b; std::memcpy(&b, src + (size_t)c * 2, 2);
                dst[c] = bf16_to_float(b);
            }
        } else {
            throw std::runtime_error("read_rows: unsupported dtype");
        }
    }
    return out;
}

} // namespace winnex_nano
