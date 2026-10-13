#include "rag/retrieval.h"

#include <stdexcept>

namespace rag {

RetrievalResponse retrieve_payload(const Bank& bank, const Match& match) {
  RetrievalResponse response{match, std::nullopt};
  if (!match.accepted) {
    return response;
  }

  response.payload = bank.load(match.id);
  if (response.payload->visual_features.size() != kVisualFeatureCount) {
    throw std::runtime_error("Retrieved payload has an invalid shape");
  }
  return response;
}

std::vector<RetrievalResponse> retrieve_payloads(
    const Bank& bank,
    const std::vector<Match>& matches) {
  std::vector<RetrievalResponse> responses;
  responses.reserve(matches.size());
  for (const Match& match : matches) {
    responses.push_back(retrieve_payload(bank, match));
  }
  return responses;
}

std::size_t payload_bytes(const RetrievalResponse& response) noexcept {
  if (!response.payload) {
    return 0;
  }
  return response.payload->description.size() +
      response.payload->visual_features.size() * sizeof(float);
}

}  // namespace rag
