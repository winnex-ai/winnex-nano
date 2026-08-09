/**
 * x_factor.hpp — the X factor: manifold basis of a model's embedding space.
 *
 * For ANY model, the X factor is the orthonormal basis of the manifold where
 * its token embeddings live — computed deterministically (no training) from
 * the model's embed_tokens via eigendecomposition of the covariance:
 *
 *   1. Ē = E − mean(E)                     (center)
 *   2. C = ĒᵀĒ / V                         (D×D covariance)
 *   3. C = U Λ Uᵀ                           (eigh; U orthonormal, Λ descending)
 *   4. r = min{r : cumsum(Λ₁..r)/sum(Λ) ≥ τ}   (effective rank, τ=0.95)
 *   5. X = U[:, :r]                         (the X factor: D×r manifold basis)
 *   6. P = X·Xᵀ                             (orthogonal projector onto the manifold)
 *
 * To embed text into the model's space WITHOUT training:
 *   7. ψ_c ∈ ℝ^d                            (spectral token of a character)
 *   8. ψ̃_c = expand(ψ_c) ∈ ℝ^D              (zero-pad + L2 normalize)
 *   9. h_c = P·ψ̃_c ∈ ℝ^D                    (project onto the manifold)
 *   10. h₁..hₙ → model layers → logits      (forward pass)
 *
 * The projector is idempotent (P² = P) and symmetric (Pᵀ = P), so it is the
 * orthogonal projection onto the model's embedding manifold.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_X_FACTOR_HPP
#define WINNEX_NANO_X_FACTOR_HPP

#include <cstdint>
#include <vector>

namespace winnex_nano {

/**
 * XFactor — the deterministic manifold basis of a model's embedding space.
 *
 * Computed once per model from embed_tokens. Thread-safe after construction.
 */
class XFactor {
public:
    // Computes the X factor from the model's embed_tokens (row-major, V×D).
    // Throws on invalid input.
    XFactor(const float* embed_tokens, int vocab, int dim, double variance_tau = 0.95);

    // Dimension of the input embedding space.
    int dim() const { return dim_; }

    // Effective rank r (number of principal directions kept).
    int rank() const { return (int)X_.size() / dim_; }

    // The X factor basis (D×r, column-major). Access for inspection.
    const float* basis() const { return X_.data(); }

    // The orthogonal projector P = X·Xᵀ (D×D, row-major).
    // Precomputed for fast projection.
    const float* projector() const { return P_.data(); }

    // Projects a D-dim vector onto the manifold: h' = P·h.
    // in/out are length dim(). in and out may alias.
    void project(const float* in, float* out) const;

    // Projects a batch of n vectors (n×D) onto the manifold.
    void project_batch(const float* in, float* out, int n) const;

    // Variance fraction captured by the effective rank (e.g. 0.95).
    double variance_captured() const { return variance_captured_; }

private:
    int dim_ = 0;
    double variance_tau_ = 0.95;
    double variance_captured_ = 0.0;
    std::vector<float> X_;  // D×r (row-major: [D][r])
    std::vector<float> P_;  // D×D (row-major: the projector)
};

/**
 * expand_spectral — expands a d-dim spectral token to D-dim with zero-pad +
 * L2 normalization (step 8 of the X-factor math).
 */
void expand_spectral(const float* psi, int d, float* out, int D);

} // namespace winnex_nano

#endif // WINNEX_NANO_X_FACTOR_HPP
