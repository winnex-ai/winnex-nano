// tokenizer_artifact.cpp — persistência do índice do EngineTokenizer por modelo.
#include "winnex_nano/tokenizer_artifact.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace winnex_nano {

namespace {

constexpr const char* kMagic = "WNNXTOK1";     // 8 bytes, cabeçalho
constexpr const char* kEngine = "MADHAVA_INDEX_V1";
constexpr size_t kSampleRows = 8;              // linhas amostradas por ponta

// sha256 mínimo (domínio público, sem dependência externa). O fingerprint
// precisa ser estável entre máquinas e versões, não criptograficamente forte
// contra adversário — ele distingue modelos, não protege segredo.
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t len = 0;
    uint8_t buf[64];
    size_t buflen = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
            0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
            0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
            0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
            0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
            0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
            0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
            0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
            0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t)p[i*4]<<24 | (uint32_t)p[i*4+1]<<16 |
                   (uint32_t)p[i*4+2]<<8 | (uint32_t)p[i*4+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15]>>3);
            uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2]>>10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    void update(const void* data, size_t n) {
        const uint8_t* p = (const uint8_t*)data;
        len += n;
        while (n) {
            const size_t take = std::min(n, sizeof(buf) - buflen);
            std::memcpy(buf + buflen, p, take);
            buflen += take; p += take; n -= take;
            if (buflen == 64) { block(buf); buflen = 0; }
        }
    }
    std::string finish() {
        const uint64_t bits = len * 8;
        uint8_t pad = 0x80; update(&pad, 1);
        uint8_t z = 0; while (buflen != 56) update(&z, 1);
        uint8_t lb[8];
        for (int i = 0; i < 8; ++i) lb[i] = (uint8_t)(bits >> (56 - 8*i));
        update(lb, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) snprintf(out + i*8, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

void write_u32(std::ostream& o, uint32_t v) { o.write((const char*)&v, 4); }
void write_str(std::ostream& o, const std::string& s) {
    write_u32(o, (uint32_t)s.size());
    o.write(s.data(), (std::streamsize)s.size());
}
uint32_t read_u32(std::istream& i) {
    uint32_t v = 0; i.read((char*)&v, 4); return v;
}
std::string read_str(std::istream& i, size_t cap = 1 << 20) {
    uint32_t n = read_u32(i);
    if (n > cap) throw std::invalid_argument("tokenizer_artifact: string corrompida");
    std::string s(n, '\0');
    if (n) i.read(&s[0], (std::streamsize)n);
    return s;
}

} // namespace

// ---------------------------------------------------------------------------
std::vector<std::string> known_embed_tensor_names() {
    // Ordem de tentativa. São as famílias densas que o ForwardEngine aceita
    // (qwen2/qwen3/llama/mistral) mais o gpt2. NÃO é uma lista fechada: quem
    // tem modelo fora dessas passa o nome explícito.
    return {
        "model.embed_tokens.weight",        // qwen2/qwen3/llama/mistral
        "transformer.wte.weight",           // gpt2
        "model.decoder.embed_tokens.weight",// variantes encoder-decoder
        "embed_tokens.weight",              // checkpoints achatados
    };
}

std::string resolve_embed_tensor_name(
    const std::string& explicit_name,
    const std::function<bool(const std::string&)>& has) {
    if (!explicit_name.empty()) {
        if (!has(explicit_name))
            throw std::invalid_argument(
                "embed tensor '" + explicit_name + "' nao existe no checkpoint");
        return explicit_name;
    }
    for (const auto& n : known_embed_tensor_names())
        if (has(n)) return n;
    throw std::invalid_argument(
        "nenhum tensor de embedding conhecido no checkpoint. Modelos fora de "
        "qwen/llama/mistral/gpt2 exigem o nome explicito — um default "
        "silencioso aqui produziria ids validos e sem sentido");
}

bool ModelIdentity::compatible_with(const ModelIdentity& o) const {
    return vocab == o.vocab && dim == o.dim &&
           tensor_name == o.tensor_name && fingerprint == o.fingerprint;
}

std::string ModelIdentity::mismatch(const ModelIdentity& o) const {
    std::ostringstream s;
    if (vocab != o.vocab) s << "vocab " << vocab << " vs " << o.vocab << "; ";
    if (dim != o.dim) s << "dim " << dim << " vs " << o.dim << "; ";
    if (tensor_name != o.tensor_name)
        s << "tensor '" << tensor_name << "' vs '" << o.tensor_name << "'; ";
    if (fingerprint != o.fingerprint)
        s << "fingerprint " << fingerprint.substr(0, 12) << " vs "
          << o.fingerprint.substr(0, 12) << "; ";
    return s.str();
}

// ---------------------------------------------------------------------------
std::string TokenizerArtifact::fingerprint(const float* emb, int vocab, int dim) {
    if (!emb || vocab <= 0 || dim <= 0)
        throw std::invalid_argument("fingerprint: argumentos invalidos");
    Sha256 h;
    // Amostra as primeiras e as últimas linhas. Não hasheia o arquivo inteiro:
    // 545 MB por chamada seria caro, e o que o guard precisa é distinguir dois
    // modelos, não detectar um bit trocado.
    const size_t n = std::min((size_t)vocab, kSampleRows);
    for (size_t i = 0; i < n; ++i)
        h.update(emb + i * dim, sizeof(float) * dim);
    for (size_t i = 0; i < n; ++i) {
        const size_t row = (size_t)vocab - 1 - i;
        h.update(emb + row * dim, sizeof(float) * dim);
    }
    uint32_t meta[2] = {(uint32_t)vocab, (uint32_t)dim};
    h.update(meta, sizeof(meta));
    return h.finish();
}

std::string TokenizerArtifact::default_path(const std::string& dir,
                                            const std::string& model_name) {
    if (model_name.empty())
        throw std::invalid_argument("default_path: model_name vazio");
    // Sanitiza: nada de barra, ".." ou espaço. O nome do modelo vem do
    // chamador e vira caminho de arquivo — sem isso, um nome com "../" escreve
    // fora do diretório de cache.
    std::string safe;
    safe.reserve(model_name.size());
    for (char c : model_name) {
        const unsigned char u = (unsigned char)c;
        if (std::isalnum(u) || c == '-' || c == '_' || c == '.') {
            safe += c;
        } else {
            safe += '_';
        }
    }
    if (safe == "." || safe == ".." || safe.empty())
        throw std::invalid_argument("default_path: model_name invalido: '" +
                                    model_name + "'");
    std::string sep = dir.empty() ? "" : (dir.back() == '/' ? "" : "/");
    return dir + sep + safe + ".wnnxtok";
}

ModelIdentity TokenizerArtifact::peek(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("peek: nao abriu " + path);
    char magic[9] = {0};
    in.read(magic, 8);
    if (in.gcount() != 8 || std::strncmp(magic, kMagic, 8) != 0)
        throw std::invalid_argument("peek: " + path + " nao e' um artefato WNNXTOK1");
    ModelIdentity id;
    id.engine      = read_str(in);
    id.model_name  = read_str(in);
    id.tensor_name = read_str(in);
    id.vocab       = (int)read_u32(in);
    id.dim         = (int)read_u32(in);
    id.fingerprint = read_str(in);
    return id;
}

bool TokenizerArtifact::save(const std::string& path, const ModelIdentity& id,
                             const std::string& madhava_index_path) const {
    if (id.engine.empty() || id.tensor_name.empty() || id.fingerprint.empty())
        throw std::invalid_argument("save: identidade incompleta");

    std::ifstream idx(madhava_index_path, std::ios::binary);
    if (!idx) throw std::invalid_argument(
        "save: indice Madhava nao encontrado: " + madhava_index_path);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(kMagic, 8);
    write_str(out, id.engine);
    write_str(out, id.model_name);
    write_str(out, id.tensor_name);
    write_u32(out, (uint32_t)id.vocab);
    write_u32(out, (uint32_t)id.dim);
    write_str(out, id.fingerprint);
    // Índice em seguida, cru. O TAMANHO precisa ir no cabeçalho: sem ele o
    // extract não sabe onde o corpo começa nem termina, e um `<< rdbuf()` sem
    // âncora lê do começo do arquivo, corrompendo tudo em silêncio.
    std::ostringstream tmp;
    tmp << idx.rdbuf();
    const std::string body = tmp.str();
    write_u32(out, (uint32_t)body.size());
    out.write(body.data(), (std::streamsize)body.size());
    return out.good();
}

std::string TokenizerArtifact::extract_index_to(const std::string& artifact_path,
                                                const std::string& tmp_path) const {
    std::ifstream in(artifact_path, std::ios::binary);
    if (!in) throw std::invalid_argument("extract: nao abriu " + artifact_path);
    char magic[9] = {0};
    in.read(magic, 8);
    if (in.gcount() != 8 || std::strncmp(magic, kMagic, 8) != 0)
        throw std::invalid_argument("extract: artefato invalido");
    read_str(in); read_str(in); read_str(in);   // engine, model, tensor
    read_u32(in); read_u32(in);                 // vocab, dim
    read_str(in);                               // fingerprint
    const uint32_t body_len = read_u32(in);

    std::vector<char> body(body_len);
    if (body_len && !in.read(body.data(), (std::streamsize)body_len))
        throw std::invalid_argument("extract: corpo truncado em " + artifact_path);

    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("extract: nao escreveu " + tmp_path);
    if (body_len) out.write(body.data(), (std::streamsize)body_len);
    out.close();
    return tmp_path;
}

// ---------------------------------------------------------------------------
// Verificação de pareamento. Existe porque não há outra forma de pegar o erro
// mais caro do multimodelo: índice do modelo A, embeddings do modelo B. O
// formato do arquivo é válido nos dois casos; só a identidade distingue.
void verify_artifact_matches(const ModelIdentity& artifact_id,
                             const float* embed_tokens, int vocab, int dim,
                             const std::string& tensor_name) {
    ModelIdentity got;
    got.tensor_name = tensor_name;
    got.vocab = vocab;
    got.dim = dim;
    if (vocab != artifact_id.vocab || dim != artifact_id.dim)
        throw std::invalid_argument(
            "artefato pertence a outro modelo: " + artifact_id.mismatch(got));
    if (!tensor_name.empty() && tensor_name != artifact_id.tensor_name)
        throw std::invalid_argument(
            "artefato pertence a outro tensor: " + artifact_id.mismatch(got));
    got.fingerprint = TokenizerArtifact::fingerprint(embed_tokens, vocab, dim);
    if (got.fingerprint != artifact_id.fingerprint)
        throw std::invalid_argument(
            "o embed_tokens NAO e' o mesmo sobre o qual o artefato foi "
            "construido (fingerprint " + got.fingerprint.substr(0, 12) +
            " vs " + artifact_id.fingerprint.substr(0, 12) + "). Carregar "
            "este indice e buscar com este corpus devolveria ids validos e "
            "errados. Reconstrua o artefato ou use o do modelo certo.");
}

} // namespace winnex_nano
