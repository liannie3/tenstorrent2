#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rag {

// Dimensions fixed by the TinyLlama and CLIP representations used in the paper.
constexpr std::uint32_t kTextDim = 2048;
constexpr std::uint32_t kVisualTokens = 101;
constexpr std::uint32_t kVisualDim = 1024;
constexpr std::size_t kVisualFeatureCount =
    static_cast<std::size_t>(kVisualTokens) * kVisualDim;

struct Payload {
  std::string description;
  std::vector<float> visual_features;
};

// Loads search keys eagerly and reads the winning entry's larger payload on demand.
class Bank {
 public:
  explicit Bank(const std::string& path);

  std::uint32_t size() const { return count_; }
  const float* keys() const { return keys_.data(); }
  Payload load(std::uint32_t id) const;

 private:
  std::string path_;
  std::uint32_t count_ = 0;
  std::vector<float> keys_;
  std::vector<std::uint64_t> offsets_;
};

}  // namespace rag
