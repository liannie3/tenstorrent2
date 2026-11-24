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
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace tt::constants;
using namespace tt::tt_metal;

std::vector<float> load_query(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("unable to open query: " + path);
  }
  const auto bytes = input.tellg();
  constexpr std::streamoff query_bytes = rag::kTextDim * sizeof(float);
  if (bytes < query_bytes || bytes % query_bytes != 0) {
    throw std::runtime_error(
        "query corpus must contain complete 2048-float queries");
  }
  input.seekg(0);
  std::vector<float> query(rag::kTextDim);
  input.read(
      reinterpret_cast<char*>(query.data()),
      static_cast<std::streamsize>(query_bytes));
  if (!input) {
    throw std::runtime_error("failed to read query");
  }
  return query;
}

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

std::vector<float> run_scores(
    const rag::Bank& bank,
    const std::vector<float>& normalized_query,
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
  std::vector<bfloat16> output(static_cast<std::size_t>(m) * n, 0.0F);

  auto& queue = mesh_device->mesh_command_queue();
  distributed::MeshWorkload workload;
  const distributed::MeshCoordinateRange device_range(mesh_device->shape());
  Program program;
  const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
  const std::uint32_t available_workers = grid.x * grid.y;
  const auto tile_ranges = rag::tt_metal::partition_entries(
      global_nt, available_workers);
  const auto active_cores = static_cast<std::uint32_t>(tile_ranges.size());
  const std::uint32_t tile_bytes =
      sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
  const distributed::DeviceLocalBufferConfig local_config{
      .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
  const distributed::ReplicatedBufferConfig query_config{
      .size = sizeof(bfloat16) * query_matrix.size()};
  const distributed::ReplicatedBufferConfig bank_config{
      .size = sizeof(bfloat16) * bank_matrix.size()};
  const distributed::ReplicatedBufferConfig output_config{
      .size = sizeof(bfloat16) * output.size()};
  auto query_buffer = distributed::MeshBuffer::create(
      query_config,
      local_config,
      mesh_device.get());
  auto bank_buffer = distributed::MeshBuffer::create(
      bank_config,
      local_config,
      mesh_device.get());
  auto output_buffer = distributed::MeshBuffer::create(
      output_config,
      local_config,
      mesh_device.get());

  constexpr tt::DataFormat format = tt::DataFormat::Float16_b;
  constexpr std::uint32_t query_cb = tt::CBIndex::c_0;
  constexpr std::uint32_t bank_cb = tt::CBIndex::c_1;
  constexpr std::uint32_t scores_cb = tt::CBIndex::c_16;
  const std::string kernel_root = RAG_TT_KERNEL_ROOT;
  for (std::uint32_t worker = 0; worker < active_cores; ++worker) {
    const CoreCoord core{worker % grid.x, worker / grid.x};
    const rag::tt_metal::WorkRange tile_range = tile_ranges[worker];
    const std::uint32_t local_nt = tile_range.end - tile_range.begin;
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(2 * tile_bytes, {{query_cb, format}})
            .set_page_size(query_cb, tile_bytes));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(2 * tile_bytes, {{bank_cb, format}})
            .set_page_size(bank_cb, tile_bytes));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(2 * tile_bytes, {{scores_cb, format}})
            .set_page_size(scores_cb, tile_bytes));

    std::vector<std::uint32_t> reader_compile_args;
    TensorAccessorArgs(*query_buffer).append_to(reader_compile_args);
    TensorAccessorArgs(*bank_buffer).append_to(reader_compile_args);
    const auto reader = CreateKernel(
        program,
        kernel_root + "/reader_scores.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    std::vector<std::uint32_t> writer_compile_args;
    TensorAccessorArgs(*output_buffer).append_to(writer_compile_args);
    const auto writer = CreateKernel(
        program,
        kernel_root + "/writer_scores.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});
    CreateKernel(
        program,
        kernel_root + "/compute_scores.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {mt, kt, local_nt}});

    SetRuntimeArgs(
        program,
        reader,
        core,
        {runtime_address(query_buffer->address()),
         runtime_address(bank_buffer->address()), mt, kt, global_nt,
         tile_range.begin, local_nt});
    SetRuntimeArgs(
        program,
        writer,
        core,
        {runtime_address(output_buffer->address()), mt, global_nt,
         tile_range.begin, local_nt});
  }

  std::cout << "TT-Metal score grid: " << grid.x << 'x' << grid.y
            << ", active cores=" << active_cores
            << ", score tiles=" << global_nt << '\n';

  distributed::EnqueueWriteMeshBuffer(queue, query_buffer, query_matrix, false);
  distributed::EnqueueWriteMeshBuffer(queue, bank_buffer, bank_matrix, false);
  workload.add_program(device_range, std::move(program));
  distributed::EnqueueMeshWorkload(queue, workload, false);
  distributed::EnqueueReadMeshBuffer(queue, output, output_buffer, true);
  output = untilize_nfaces(output, m, n);

  std::vector<float> scores(bank.size());
  for (std::uint32_t entry = 0; entry < bank.size(); ++entry) {
    scores[entry] = static_cast<float>(output[entry]);
  }
  return scores;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: rag_tt_metal_smoke BANK QUERY [THRESHOLD]\n";
    return 2;
  }
  try {
    const float threshold = argc == 4 ? std::stof(argv[3]) : 0.7F;
    const rag::Bank bank(argv[1]);
    const std::vector<float> query = load_query(argv[2]);
    const std::vector<float> normalized = rag::normalize_query(query);
    const rag::Match expected = rag::search_cpu(bank, query, threshold);

    auto device = distributed::MeshDevice::create_unit_mesh(0);
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const std::uint32_t score_tiles =
        round_up(bank.size(), TILE_WIDTH) / TILE_WIDTH;
    const std::uint32_t active_workers =
        std::min(score_tiles, grid.x * grid.y);
    const std::vector<float> scores = run_scores(bank, normalized, device);
    const auto compact = rag::tt_metal::compact_result_from_scores(
        scores, active_workers, threshold);
    const rag::Match actual = rag::tt_metal::decode_result(
        compact, bank.size());
    const bool closed = device->close();

    std::cout << "expected: accepted=" << expected.accepted
              << " id=" << expected.id << " score=" << expected.score << '\n';
    std::cout << "metalium: accepted=" << actual.accepted
              << " id=" << actual.id << " score=" << actual.score << '\n';
    if (!closed || actual.id != expected.id ||
        actual.accepted != expected.accepted) {
      std::cerr << "multicore TT-Metal RAG scoring smoke: FAIL\n";
      return 1;
    }
    std::cout << "multicore TT-Metal RAG scoring smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rag_tt_metal_smoke: " << error.what() << '\n';
    return 1;
  }
}
