/**
 * weight_balancer.hpp — PsiQRH-derived multimodel weight blending.
 *
 *   W' = Σᵢ αᵢ · R(qᵢ) · Wᵢ      with Σᵢ αᵢ = 1, 0 ≤ αᵢ ≤ 1
 *
 * αᵢ are operator-controlled blend weights (cost/benefit choice), and R(qᵢ)
 * is an optional unit-quaternion rotation applied per tensor. The rotation is
 * a "control knob" that lets the operator phase-align weight matrices before
 * blending, avoiding the destructive interference of plain linear
 * interpolation of independently-trained models.
 *
 * The engine is MODEL-AGNOSTIC: it blends any tensors by name. Two (or more)
 * models of the SAME architecture provide a name → Tensor map each; this
 * service produces a single blended map consumed by the forward pass.
 *
 * BSL 1.1 | pay@winnex.ai | (c) Winnex Brasil Soluções Empresariais LTDA-ME
 */
#ifndef WINNEX_NANO_WEIGHT_BALANCER_HPP
#define WINNEX_NANO_WEIGHT_BALANCER_HPP

#include <map>
#include <string>
#include <vector>

#include "winnex_nano/spectral_tokenizer.hpp"  // for Quat, quat_mul

namespace winnex_nano {

// A model's tensor map: name -> raw float weights (already dequantized/f32).
// For the balancer we operate on f32 (CPU path) or float16/bf16 (GPU path);
// this version uses float for simplicity and correctness.
using WeightMap = std::map<std::string, std::vector<float>>;

struct BlendWeight {
    double alpha = 0.0;   // contribution of this model (Σ α = 1)
    double theta = 0.0;   // quaternion rotation angle (radians), 0 = none
    double omega = 0.0;
    double phi = 0.0;
};

class WeightBalancer {
public:
    // Blends N model weight maps into one. Only tensor names present in ALL
    // models are blended (the intersection); tensors missing from some models
    // are taken from the first model that has them.
    WeightMap blend(const std::vector<WeightMap>& models,
                    const std::vector<BlendWeight>& weights) const;

    // Same but with per-tensor rotation applied (uses quaternion R(qᵢ)).
    // If theta==0 for all, this reduces to the plain weighted sum.
    WeightMap blend_with_rotation(const std::vector<WeightMap>& models,
                                  const std::vector<BlendWeight>& weights) const;

private:
    // Applies a unit quaternion rotation to a weight vector, treating each
    // consecutive group of 4 values as a quaternion [w,x,y,z].
    void _rotate_tensor(std::vector<float>& t, const Quat& q) const;
};

} // namespace winnex_nano

#endif // WINNEX_NANO_WEIGHT_BALANCER_HPP
