#include "infini_train/include/optimizer.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params) : params_(params) {}

void Optimizer::ZeroGrad(bool set_to_none) {
    for (auto param : params_) { param->ZeroGrad(set_to_none); }
}

namespace optimizers {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate)
    : Optimizer(params), learning_rate_(learning_rate) {}

void SGD::Step() {
    for (auto param : params_) {
        if (!param->grad()) {
            LOG(INFO) << "Skipping param with null grad.";
            continue;
        }
        auto device = param->GetDevice();
        core::DeviceGuard guard(device);
        auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AccumulateGrad"});
        kernel.Call<void>(param->grad(), -learning_rate_, param);
    }
}

auto SGD::StateDict() const -> std::unordered_map<std::string, std::shared_ptr<Tensor>> {
    return {};
}

auto SGD::LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) -> void {
    // no state to load for SGD
}

} // namespace optimizers

/**
 * @brief Implementation of Adam optimizer
 */
namespace optimizers {

Adam::Adam(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate, float beta1, float beta2, float eps)
    : Optimizer(params), t_(0), learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {

    for (const auto &param : params_) {
        m_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        v_.emplace_back(std::make_shared<Tensor>(param->Dims(), param->Dtype(), param->GetDevice()));
        DispatchFunc<INFINI_ALL_TYPES>(
            param->Dtype(),
            [this]<typename T>() {
                m_.back()->Fill<T>(0);
                v_.back()->Fill<T>(0);
            },
            "CUDA Adam");
    }
}

void Adam::Step() {
    ++t_;

    for (size_t i = 0; i < params_.size(); ++i) {
        auto &param = params_[i];
        const auto &grad = param->grad();
        if (!grad) {
            LOG(INFO) << "Skipping param with null grad.";
            continue;
        }
        auto &m = m_[i];
        auto &v = v_[i];

        auto device = param->GetDevice();
        core::DeviceGuard guard(device);
        auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AdamAccumulateGrad"});
        kernel.Call<void>(grad, param, m, v, learning_rate_, beta1_, beta2_, eps_, t_);
    }
}

auto Adam::StateDict() const -> std::unordered_map<std::string, std::shared_ptr<Tensor>> {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;

    for (size_t i = 0; i < m_.size(); ++i) {
        state_dict["__opt_m_" + std::to_string(i)] = m_[i];
        state_dict["__opt_v_" + std::to_string(i)] = v_[i];
    }

    // Store t_ as a 1-element INT64 tensor on CPU
    auto t_tensor = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT64, Device());
    *static_cast<int64_t *>(t_tensor->DataPtr()) = t_;
    state_dict["__opt_t__"] = t_tensor;

    return state_dict;
}

auto Adam::LoadStateDict(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) -> void {
    // Restore t_
    auto t_it = state_dict.find("__opt_t__");
    if (t_it != state_dict.end()) {
        Tensor t_cpu = t_it->second->To(Device());
        auto *impl = core::GetDeviceGuardImpl(t_it->second->GetDevice().type());
        impl->SynchronizeDevice(t_it->second->GetDevice());
        t_ = *static_cast<const int64_t *>(t_cpu.DataPtr());
    }

    // Restore m_ and v_ (CopyFrom handles H2D/D2H as needed)
    for (size_t i = 0; i < m_.size(); ++i) {
        auto m_it = state_dict.find("__opt_m_" + std::to_string(i));
        if (m_it != state_dict.end()) {
            m_[i]->CopyFrom(*m_it->second);
        }

        auto v_it = state_dict.find("__opt_v_" + std::to_string(i));
        if (v_it != state_dict.end()) {
            v_[i]->CopyFrom(*v_it->second);
        }
    }
}

} // namespace optimizers

} // namespace infini_train
