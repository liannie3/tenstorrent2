#pragma once

#ifdef RAG_HAS_NVTX
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif
#endif

namespace rag::profile {

class ScopedRange {
 public:
  explicit ScopedRange(const char* name) noexcept {
#ifdef RAG_HAS_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ~ScopedRange() {
#ifdef RAG_HAS_NVTX
    nvtxRangePop();
#endif
  }

  ScopedRange(const ScopedRange&) = delete;
  ScopedRange& operator=(const ScopedRange&) = delete;
};

}  // namespace rag::profile

#define RAG_PROFILE_JOIN_IMPL(left, right) left##right
#define RAG_PROFILE_JOIN(left, right) RAG_PROFILE_JOIN_IMPL(left, right)
#define RAG_PROFILE_RANGE(name) \
  ::rag::profile::ScopedRange RAG_PROFILE_JOIN(rag_profile_range_, __LINE__)(name)
