#include "rag/bank.h"
#include "rag/search.h"
#include "rag/tt_metal_contract.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace tt::constants;
using namespace tt::tt_metal;

std::uint32_t round_up(
    const std::uint32_t value, const std::uint32_t alignment) {
  return ((value + alignment - 1U) / alignment) * alignment;
}

std::uint32_t runtime_address(const DeviceAddr address) {
  if (address > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("device address does not fit a kernel runtime argument");
  }
  return static_cast<std::uint32_t>(address);
}

std::uint32_t float_bits(const float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

rag::tt_metal::CompactResult run_device_search(
    const rag::Bank& bank,
    const std::vector<float>& normalized_query,
    const float threshold,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
  constexpr std::uint32_t m = TILE_HEIGHT;
  constexpr std::uint32_t k = rag::kTextDim;
  const std::uint32_t n = round_up(bank.size(), TILE_WIDTH);
  const std::uint32_t mt = m / TILE_HEIGHT;
  const std::uint32_t kt = k / TILE_WIDTH;
  const std::uint32_t global_nt = n / TILE_WIDTH;

  std::vector<bfloat16> query_matrix(static_cast<std::size_t>(m) * k, 0.0F);
  for (std::uint32_t column = 0; column < k; ++column) {
    query_matrix[column] = bfloat16(normalized_query[column]);
  }

  // B is KxN: each column is one normalized bank key. Padding columns are
  // zero but are excluded from host-side selection by bank.size().
  std::vector<bfloat16> bank_matrix(static_cast<std::size_t>(k) * n, 0.0F);
  for (std::uint32_t entry = 0; entry < bank.size(); ++entry) {
    const float* key = bank.keys() + static_cast<std::size_t>(entry) * k;
    for (std::uint32_t dimension = 0; dimension < k; ++dimension) {
      bank_matrix[static_cast<std::size_t>(dimension) * n + entry] =
          bfloat16(key[dimension]);
    }
  }

  query_matrix = tilize_nfaces(query_matrix, m, k);
  bank_matrix = tilize_nfaces(bank_matrix, k, n);
  auto& queue = mesh_device->mesh_command_queue();
  distributed::MeshWorkload scoring_workload;
  distributed::MeshWorkload reduction_workload;
  const distributed::MeshCoordinateRange device_range(mesh_device->shape());
  Program scoring_program;
  Program reduction_program;
  const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
  const auto available_workers =
      static_cast<std::uint32_t>(grid.x * grid.y);
  const auto tile_ranges = rag::tt_metal::partition_entries(
      global_nt, available_workers);
  const auto active_cores = static_cast<std::uint32_t>(tile_ranges.size());
  const std::uint32_t tile_bytes =
      sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
  constexpr std::uint32_t candidate_page_bytes =
      sizeof(rag::tt_metal::DeviceCandidatePage);
  constexpr std::uint32_t result_page_bytes = 32U;
  constexpr std::uint32_t result_page_words =
      result_page_bytes / sizeof(std::uint32_t);
  const distributed::DeviceLocalBufferConfig tile_local_config{
      .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
  const distributed::DeviceLocalBufferConfig candidate_local_config{
      .page_size = candidate_page_bytes, .buffer_type = BufferType::DRAM};
  const distributed::ReplicatedBufferConfig query_config{
      .size = sizeof(bfloat16) * query_matrix.size()};
  const distributed::ReplicatedBufferConfig bank_config{
      .size = sizeof(bfloat16) * bank_matrix.size()};
  const distributed::ReplicatedBufferConfig candidate_config{
      .size = candidate_page_bytes * active_cores};
  const distributed::ReplicatedBufferConfig result_config{
      .size = result_page_bytes};
  auto query_buffer = distributed::MeshBuffer::create(
      query_config,
      tile_local_config,
      mesh_device.get());
  auto bank_buffer = distributed::MeshBuffer::create(
      bank_config,
      tile_local_config,
      mesh_device.get());
  auto candidate_buffer = distributed::MeshBuffer::create(
      candidate_config,
      candidate_local_config,
      mesh_device.get());
  auto result_buffer = distributed::MeshBuffer::create(
      result_config,
      candidate_local_config,
      mesh_device.get());

  constexpr tt::DataFormat format = tt::DataFormat::Float16_b;
  constexpr std::uint32_t query_cb = tt::CBIndex::c_0;
  constexpr std::uint32_t bank_cb = tt::CBIndex::c_1;
  constexpr std::uint32_t candidate_cb = tt::CBIndex::c_2;
  constexpr std::uint32_t scores_cb = tt::CBIndex::c_16;
  const std::string kernel_root = RAG_TT_KERNEL_ROOT;
  for (std::uint32_t worker = 0; worker < active_cores; ++worker) {
    const CoreCoord core{worker % grid.x, worker / grid.x};
    const rag::tt_metal::WorkRange tile_range = tile_ranges[worker];
    const std::uint32_t local_nt = tile_range.end - tile_range.begin;
    CreateCircularBuffer(
        scoring_program,
        core,
        CircularBufferConfig(2 * tile_bytes, {{query_cb, format}})
            .set_page_size(query_cb, tile_bytes));
    CreateCircularBuffer(
        scoring_program,
        core,
        CircularBufferConfig(2 * tile_bytes, {{bank_cb, format}})
            .set_page_size(bank_cb, tile_bytes));
    CreateCircularBuffer(
        scoring_program,
        core,
        CircularBufferConfig(2 * tile_bytes, {{scores_cb, format}})
            .set_page_size(scores_cb, tile_bytes));
    CreateCircularBuffer(
        scoring_program,
        core,
        CircularBufferConfig(
            candidate_page_bytes,
            {{candidate_cb, tt::DataFormat::UInt32}})
            .set_page_size(candidate_cb, candidate_page_bytes));

    std::vector<std::uint32_t> reader_compile_args;
    TensorAccessorArgs(*query_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(*bank_buffer).append_to(reader_compile_args);
    const auto reader = CreateKernel(
        scoring_program,
        kernel_root + "/reader_scores.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    std::vector<std::uint32_t> writer_compile_args;
    TensorAccessorArgs(*candidate_buffer).append_to(writer_compile_args);
    const auto writer = CreateKernel(
        scoring_program,
        kernel_root + "/writer_scores.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});
    CreateKernel(
        scoring_program,
        kernel_root + "/compute_scores.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {mt, kt, local_nt}});

    SetRuntimeArgs(
        scoring_program,
        reader,
        core,
        {runtime_address(query_buffer->address()),
         runtime_address(bank_buffer->address()), mt, kt, global_nt,
         tile_range.begin, local_nt});
    SetRuntimeArgs(
        scoring_program,
        writer,
        core,
        {runtime_address(candidate_buffer->address()), worker,
         tile_range.begin, local_nt, bank.size()});
  }

  std::cout << "TT-Metal score grid: " << grid.x << 'x' << grid.y
            << ", active cores=" << active_cores
            << ", score tiles=" << global_nt
            << ", result D2H bytes="
            << result_page_bytes << " (12-byte logical result)\n";

  const CoreCoord reduction_core{0, 0};
  CreateCircularBuffer(
      reduction_program,
      reduction_core,
      CircularBufferConfig(
          candidate_page_bytes,
          {{query_cb, tt::DataFormat::UInt32}})
          .set_page_size(query_cb, candidate_page_bytes));
  CreateCircularBuffer(
      reduction_program,
      reduction_core,
      CircularBufferConfig(
          result_page_bytes,
          {{bank_cb, tt::DataFormat::UInt32}})
          .set_page_size(bank_cb, result_page_bytes));
  std::vector<std::uint32_t> reduction_compile_args;
  TensorAccessorArgs(*candidate_buffer).append_to(reduction_compile_args);
  TensorAccessorArgs(*result_buffer).append_to(reduction_compile_args);
  const auto reducer = CreateKernel(
      reduction_program,
      kernel_root + "/reduce_candidates.cpp",
      reduction_core,
      DataMovementConfig{
          .processor = DataMovementProcessor::RISCV_0,
          .noc = NOC::RISCV_0_default,
          .compile_args = reduction_compile_args});
  SetRuntimeArgs(
      reduction_program,
      reducer,
      reduction_core,
      {runtime_address(candidate_buffer->address()),
       runtime_address(result_buffer->address()), active_cores,
       float_bits(threshold), bank.size()});

  distributed::EnqueueWriteMeshBuffer(queue, query_buffer, query_matrix, false);
  distributed::EnqueueWriteMeshBuffer(queue, bank_buffer, bank_matrix, false);
  scoring_workload.add_program(device_range, std::move(scoring_program));
  reduction_workload.add_program(device_range, std::move(reduction_program));
  distributed::EnqueueMeshWorkload(queue, scoring_workload, false);
  distributed::EnqueueMeshWorkload(queue, reduction_workload, false);
  std::vector<std::uint32_t> result_words(result_page_words, 0U);
  distributed::EnqueueReadMeshBuffer(
      queue, result_words, result_buffer, true);
  if (result_words[3] != 0U) {
    throw std::runtime_error(
        "device global reduction reported error " +
        std::to_string(result_words[3]));
  }
  for (std::size_t index = 4; index < result_words.size(); ++index) {
    if (result_words[index] != 0U) {
      throw std::runtime_error(
          "device global reduction returned nonzero reserved data");
    }
  }
  rag::tt_metal::CompactResult result{
      result_words[0], result_words[1], 0.0F};
  std::memcpy(&result.score, &result_words[2], sizeof(result.score));
  return result;
}

}  // namespace

namespace rag {

struct TtMetalSearcher::Impl {
  const Bank& bank;
  float threshold;
  std::shared_ptr<distributed::MeshDevice> device;

  Impl(const Bank& source_bank, const float configured_threshold)
      : bank(source_bank),
        threshold(configured_threshold),
        device(distributed::MeshDevice::create_unit_mesh(0)) {
    if (!std::isfinite(threshold) || threshold < -1.0F || threshold > 1.0F) {
      throw std::invalid_argument("threshold must be finite and in [-1, 1]");
    }
  }

  ~Impl() {
    if (device) {
      device->close();
    }
  }

  Match execute(const std::vector<float>& query) {
    const auto normalized = normalize_query(query);
    const auto compact =
        run_device_search(bank, normalized, threshold, device);
    return tt_metal::decode_result(compact, bank.size());
  }
};

TtMetalSearcher::TtMetalSearcher(const Bank& bank, const float threshold)
    : impl_(std::make_unique<Impl>(bank, threshold)) {}

TtMetalSearcher::~TtMetalSearcher() = default;

Match TtMetalSearcher::search(const std::vector<float>& query) {
  return impl_->execute(query);
}

TimedSearchResult TtMetalSearcher::search_batch_timed(
    const std::vector<float>& queries,
    const std::size_t query_count,
    const float threshold) {
  if (query_count == 0 || queries.size() != query_count * kTextDim) {
    throw std::invalid_argument("TT-Metal query batch has an invalid shape");
  }
  if (threshold != impl_->threshold) {
    throw std::invalid_argument(
        "TT-Metal threshold must match the value configured at initialization");
  }

  TimedSearchResult result;
  result.matches.reserve(query_count);
  for (std::size_t query = 0; query < query_count; ++query) {
    const auto begin = queries.begin() + query * kTextDim;
    result.matches.push_back(impl_->execute(
        std::vector<float>(begin, begin + kTextDim)));
  }
  result.timing.h2d_bytes =
      query_count * tt::constants::TILE_HEIGHT * kTextDim *
      sizeof(tt::tt_metal::bfloat16);
  result.timing.d2h_bytes = query_count * 32U;
  return result;
}

}  // namespace rag
