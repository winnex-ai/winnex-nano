// weight_balancer.cpp — multimodel weight blending (PsiQRH-derived).
#include "winnex_nano/weight_balancer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace winnex_nano {

void WeightBalancer::_rotate_tensor(std::vector<float>& t, const Quat& q) const {
    // Rotate each consecutive group of 4 floats as a quaternion:
    //   t' = q * t * q†  (q unit → q† = conj)
    Quat qi = quat_conj(q);
    const size_t n4 = t.size() / 4;
    for (size_t i = 0; i < n4; ++i) {
        Quat v{t[i*4], t[i*4+1], t[i*4+2], t[i*4+3]};
        Quat r = quat_mul(quat_mul(q, v), qi);
        t[i*4]   = r.w;
        t[i*4+1] = r.x;
        t[i*4+2] = r.y;
        t[i*4+3] = r.z;
    }
}

WeightMap WeightBalancer::blend(const std::vector<WeightMap>& models,
                                const std::vector<BlendWeight>& weights) const {
    if (models.size() != weights.size() || models.empty()) {
        throw std::runtime_error("WeightBalancer: models and weights must match and be non-empty");
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
