#pragma once

#include <memory>
#include <string>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

class CheckpointManager {
public:
    using TensorMap = std::unordered_map<std::string, std::shared_ptr<Tensor>>;

    /**
     * @brief Saves a checkpoint containing the model's parameters and the optimizer's state.
     * @param path The file path to save the checkpoint to.
     * @param model The model to save.
     * @param optimizer The optimizer to save.
     * @param step The current training step (optional, can be used for naming or logging purposes).
     */
    static void Save(const std::string &path, const nn::Module &model, const Optimizer &optimizer, int64_t step);

    /**
     * @brief Loads a checkpoint from a file, restoring the model's parameters and the optimizer's state.
     * @param path The file path to load the checkpoint from.
     * @param model The model to load the parameters into.
     * @param optimizer The optimizer to load the state into (optional, can be nullptr if not needed).
     * @return The training step at which the checkpoint was saved (if available), or -1 if not available.
     */
    static int64_t Load(const std::string &path, nn::Module &model, Optimizer *optimizer = nullptr);

private:
    static auto WriteTensors(std::ofstream &, const TensorMap &) -> void;
    static auto ReadTensors(std::ifstream &) -> TensorMap;
};

} // namespace infini_train