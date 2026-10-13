#pragma once

#include "rag/search.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace rag {

// Caller-visible result of the standalone retrieval service. An accepted
// response owns its payload; a rejected response deliberately has no payload.
struct RetrievalResponse {
  Match match;
  std::optional<Payload> payload;
};

RetrievalResponse retrieve_payload(const Bank& bank, const Match& match);

std::vector<RetrievalResponse> retrieve_payloads(
    const Bank& bank,
    const std::vector<Match>& matches);

std::size_t payload_bytes(const RetrievalResponse& response) noexcept;

}  // namespace rag
