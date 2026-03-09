#include "infini_train/include/checkpoint.h"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

namespace {
constexpr char kMagic[] = "INFINICKPT"; // 10 bytes, no null written to file
constexpr uint32_t kVersion = 1;
constexpr char kStepKey[] = "__ckpt_step__";
} // namespace

auto CheckpointManager::WriteTensors(std::ofstream &file, const TensorMap &tensors) -> void {
    uint64_t num_tensors = static_cast<uint64_t>(tensors.size());
    file.write(reinterpret_cast<const char *>(&num_tensors), sizeof(num_tensors));

    for (const auto &[key, tensor] : tensors) {
        // Bring to CPU; synchronize source device before and after the async copy.
        auto device = tensor->GetDevice();
        auto *impl = core::GetDeviceGuardImpl(device.type());
        impl->SynchronizeDevice(device);
        Tensor cpu_tensor = tensor->To(Device()); // async D2H (no-op if already CPU)
        impl->SynchronizeDevice(device);          // wait for copy to complete

        // key
        uint32_t key_len = static_cast<uint32_t>(key.size());
        file.write(reinterpret_cast<const char *>(&key_len), sizeof(key_len));
        file.write(key.data(), static_cast<std::streamsize>(key_len));

        // dtype
        auto dtype_int = static_cast<int8_t>(cpu_tensor.Dtype());
        file.write(reinterpret_cast<const char *>(&dtype_int), sizeof(dtype_int));

        // shape
        uint32_t ndim = static_cast<uint32_t>(cpu_tensor.Dims().size());
        file.write(reinterpret_cast<const char *>(&ndim), sizeof(ndim));
        for (int64_t d : cpu_tensor.Dims()) { file.write(reinterpret_cast<const char *>(&d), sizeof(d)); }

        // raw data
        uint64_t data_size = cpu_tensor.SizeInBytes();
        file.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char *>(cpu_tensor.DataPtr()), static_cast<std::streamsize>(data_size));
    }
}

auto CheckpointManager::ReadTensors(std::ifstream &file) -> TensorMap {
    TensorMap tensors;

    uint64_t num_tensors = 0;
    file.read(reinterpret_cast<char *>(&num_tensors), sizeof(num_tensors));

    for (uint64_t i = 0; i < num_tensors; ++i) {
        // key
        uint32_t key_len = 0;
        file.read(reinterpret_cast<char *>(&key_len), sizeof(key_len));
        std::string key(key_len, '\0');
        file.read(key.data(), static_cast<std::streamsize>(key_len));

        // dtype
        int8_t dtype_int = 0;
        file.read(reinterpret_cast<char *>(&dtype_int), sizeof(dtype_int));
        DataType dtype = static_cast<DataType>(dtype_int);

        // shape
        uint32_t ndim = 0;
        file.read(reinterpret_cast<char *>(&ndim), sizeof(ndim));
        std::vector<int64_t> dims(ndim);
        for (uint32_t j = 0; j < ndim; ++j) { file.read(reinterpret_cast<char *>(&dims[j]), sizeof(int64_t)); }

        // raw data — allocate directly on CPU and read into it
        uint64_t data_size = 0;
        file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));
        auto tensor = std::make_shared<Tensor>(dims, dtype, Device());
        file.read(reinterpret_cast<char *>(tensor->DataPtr()), static_cast<std::streamsize>(data_size));

        tensors.emplace(std::move(key), std::move(tensor));
    }

    CHECK(file.good()) << "CheckpointManager::ReadTensors: read error while deserializing tensors";
    return tensors;
}

auto CheckpointManager::Save(const std::string &path, const nn::Module &model, const Optimizer &optimizer, int64_t step)
    -> void {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << "CheckpointManager::Save: failed to open file for writing: " << path;

    // Header: magic + version
    file.write(kMagic, static_cast<std::streamsize>(std::strlen(kMagic)));
    file.write(reinterpret_cast<const char *>(&kVersion), sizeof(kVersion));

    // Collect all tensors into one flat map
    TensorMap all_tensors;

    for (const auto &[k, v] : model.StateDict()) { all_tensors.emplace(k, v); }
    for (const auto &[k, v] : optimizer.StateDict()) { all_tensors.emplace(k, v); }

    // Training step as a 1-element INT64 tensor on CPU
    auto step_tensor = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT64, Device());
    *static_cast<int64_t *>(step_tensor->DataPtr()) = step;
    all_tensors.emplace(kStepKey, step_tensor);

    WriteTensors(file, all_tensors);

    CHECK(file.good()) << "CheckpointManager::Save: write error on file: " << path;
    LOG(INFO) << "CheckpointManager::Save: saved checkpoint (step=" << step << ") -> " << path;
}

auto CheckpointManager::Load(const std::string &path, nn::Module &model, Optimizer *optimizer) -> int64_t {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << "CheckpointManager::Load: failed to open file: " << path;

    // Verify magic
    char magic_buf[sizeof(kMagic)] = {};
    file.read(magic_buf, static_cast<std::streamsize>(std::strlen(kMagic)));
    CHECK(std::strncmp(magic_buf, kMagic, std::strlen(kMagic)) == 0)
        << "CheckpointManager::Load: invalid checkpoint file (bad magic): " << path;

    // Verify version
    uint32_t version = 0;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    CHECK_EQ(version, kVersion) << "CheckpointManager::Load: unsupported checkpoint version " << version;

    TensorMap all_tensors = ReadTensors(file);

    // Partition tensors by prefix
    TensorMap model_state;
    TensorMap optimizer_state;
    int64_t step = -1;

    for (const auto &[key, tensor] : all_tensors) {
        if (key == kStepKey) {
            step = *static_cast<const int64_t *>(tensor->DataPtr());
        } else if (key.starts_with("__opt_")) {
            optimizer_state.emplace(key, tensor);
        } else {
            model_state.emplace(key, tensor);
        }
    }

    // Restore model (CopyFrom inside handles H2D if model is on GPU)
    model.LoadStateDict(model_state);

    // Restore optimizer state
    if (optimizer != nullptr && !optimizer_state.empty()) {
        optimizer->LoadStateDict(optimizer_state);
    }

    LOG(INFO) << "CheckpointManager::Load: loaded checkpoint (step=" << step << ") <- " << path;
    return step;
}

} // namespace infini_train