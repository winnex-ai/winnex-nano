// weight_balancer.cpp — multimodel weight blending (Winnex-derived).
#include "winnex_nano/weight_balancer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace winnex_nano {

void WeightBalancer::_rotate_tensor(std::vector<float>& t, const Quat& q) const {
    // Rotate each consecutive group of 4 floats as a quaternion:
    //   t' = q * t * q†  (q unit → q† = conj)
    // The rotation is embarrassingly parallel → OpenMP over the blocks.
    const float nq2 = q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z;
    if (nq2 < 1e-12f) {
        // Not a valid (unit) quaternion — leave the tensor unchanged.
        return;
    }
    // Normalize so q is exactly unit (validates the rotation is an isometry).
    const float inv = 1.0f / std::sqrt(nq2);
    Quat qn{q.w * inv, q.x * inv, q.y * inv, q.z * inv};
    Quat qi = quat_conj(qn);

    const size_t n4 = t.size() / 4;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(n4); ++i) {
        const size_t k = static_cast<size_t>(i);
        Quat v{t[k*4], t[k*4+1], t[k*4+2], t[k*4+3]};
        Quat r = quat_mul(quat_mul(qn, v), qi);
        t[k*4]   = r.w;
        t[k*4+1] = r.x;
        t[k*4+2] = r.y;
        t[k*4+3] = r.z;
    }
}

WeightMap WeightBalancer::blend(const std::vector<WeightMap>& models,
                                const std::vector<BlendWeight>& weights) const {
    if (models.size() != weights.size() || models.empty()) {
        throw std::runtime_error("WeightBalancer: models and weights must match and be non-empty");
    }

    // Validate the blend weights: each α ∈ [0,1] and Σα = 1 (within tolerance).
    double alpha_sum = 0.0;
    for (const auto& w : weights) {
        if (w.alpha < 0.0 || w.alpha > 1.0) {
            throw std::runtime_error("WeightBalancer: alpha must be in [0,1]");
        }
        alpha_sum += w.alpha;
    }
    if (std::abs(alpha_sum - 1.0) > 1e-6) {
        throw std::runtime_error("WeightBalancer: sum of alphas must equal 1 (got " +
                                 std::to_string(alpha_sum) + ")");
    }

    // Intersection of tensor names across all models.
    std::vector<std::string> names;
    if (!models.empty()) {
        for (const auto& kv : models[0]) names.push_back(kv.first);
        for (size_t m = 1; m < models.size(); ++m) {
            names.erase(
                std::remove_if(names.begin(), names.end(),
                    [&](const std::string& n) { return models[m].count(n) == 0; }),
                names.end());
        }
    }

    WeightMap out;
    for (const auto& name : names) {
        const size_t n = models[0].at(name).size();
        std::vector<float> acc(n, 0.0f);
        for (size_t m = 0; m < models.size(); ++m) {
            const auto& w = models[m].at(name);
            if (w.size() != n) {
                throw std::runtime_error("WeightBalancer: tensor size mismatch for " + name);
            }
            float a = static_cast<float>(weights[m].alpha);
            for (size_t i = 0; i < n; ++i) acc[i] += a * w[i];
        }
        out[name] = std::move(acc);
    }
    return out;
}

WeightMap WeightBalancer::blend_with_rotation(
    const std::vector<WeightMap>& models,
    const std::vector<BlendWeight>& weights) const {
    if (models.size() != weights.size() || models.empty()) {
        throw std::runtime_error("WeightBalancer: models and weights must match and be non-empty");
    }

    // Validate the blend weights (α ∈ [0,1], Σα = 1).
    double alpha_sum = 0.0;
    for (const auto& w : weights) {
        if (w.alpha < 0.0 || w.alpha > 1.0) {
            throw std::runtime_error("WeightBalancer: alpha must be in [0,1]");
        }
        alpha_sum += w.alpha;
    }
    if (std::abs(alpha_sum - 1.0) > 1e-6) {
        throw std::runtime_error("WeightBalancer: sum of alphas must equal 1 (got " +
                                 std::to_string(alpha_sum) + ")");
    }

    std::vector<std::string> names;
    for (const auto& kv : models[0]) names.push_back(kv.first);
    for (size_t m = 1; m < models.size(); ++m) {
        names.erase(
            std::remove_if(names.begin(), names.end(),
                [&](const std::string& n) { return models[m].count(n) == 0; }),
            names.end());
    }

    WeightMap out;
    for (const auto& name : names) {
        const size_t n = models[0].at(name).size();
        std::vector<float> acc(n, 0.0f);
        for (size_t m = 0; m < models.size(); ++m) {
            auto w = models[m].at(name);
            const BlendWeight& bw = weights[m];
            if (bw.theta != 0.0) {
                Quat q = unit_quaternion((float)bw.theta, (float)bw.omega, (float)bw.phi);
                _rotate_tensor(w, q);
            }
            float a = static_cast<float>(bw.alpha);
            for (size_t i = 0; i < n; ++i) acc[i] += a * w[i];
        }
        out[name] = std::move(acc);
    }
    return out;
}

} // namespace winnex_nano
