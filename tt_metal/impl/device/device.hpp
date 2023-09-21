/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt {

namespace tt_metal {

// Fwd declares
enum class BufferType;
class Buffer;
class Program;

using on_close_device_callback = std::function<void ()>;

// A physical PCIexpress Tenstorrent device
class Device {
   public:
    static size_t detect_num_available_devices();
    friend void tt_gdb(Device* device, int chip_id, const vector<CoreCoord> cores, vector<string> ops);
    Device () = delete;
    Device(int device_id, const std::vector<uint32_t>& l1_bank_remap = {});

    ~Device();

    // TODO: Add copy/move semantics
    Device(const Device &other) { }
    Device& operator=(const Device &other) { return *this; }

    Device(Device &&other) { }
    Device& operator=(Device &&other) { return *this; }

    tt::ARCH arch() const;

    int id() const { return id_; }

    tt_cluster *cluster() const;  // Need to access cluster in llrt APIs

    bool is_initialized() const { return this->initialized_; }

    int num_dram_channels() const;

    uint32_t l1_size() const;
    uint32_t dram_bank_size() const;

    CoreCoord logical_grid_size() const;

    CoreCoord compute_with_storage_grid_size() const;

    CoreCoord worker_core_from_logical_core(const CoreCoord &logical_core) const;

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores);

    uint32_t num_banks(const BufferType &buffer_type) const;

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord core_from_dram_channel(uint32_t dram_channel) const;

    i32 l1_bank_offset_from_bank_id(uint32_t bank_id) const;

    i32 dram_bank_offset_from_bank_id(uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    std::vector<uint32_t> bank_ids_from_dram_channel(uint32_t dram_channel) const;

    std::vector<uint32_t> bank_ids_from_logical_core(const CoreCoord &logical_core) const;
    bool cluster_is_initialized() const;

    allocator::Statistics get_memory_allocation_statistics(const BufferType &buffer_type) const;

    void dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const;

    // Set of logical storage only core coordinates
    const std::set<CoreCoord> &storage_only_cores() const { return this->storage_only_cores_; }

    // Set of logical dispatch core coordinates
    const std::set<CoreCoord> &dispatch_cores() const { return this->dispatch_cores_; }
    void deallocate_buffers();

   private:
    void check_allocator_is_initialized() const;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize(const std::vector<uint32_t>& l1_bank_remap = {});
    void initialize_cluster();
    void initialize_allocator(const std::vector<uint32_t>& l1_bank_remap = {});
    void initialize_dispatch_and_banking_information();
    void clear_l1_state();
    // Puts device into reset
    bool close();
    friend bool CloseDevice(Device *device);

    // TODO: Uplift usage of friends. Buffer and Program just need access to allocator
    friend class Buffer;
    friend class Program;

#ifdef TT_METAL_VERSIM_DISABLED
    static constexpr TargetDevice target_type_ = TargetDevice::Silicon;
#else
    static constexpr TargetDevice target_type_ = TargetDevice::Versim;
#endif
    static constexpr MemoryAllocator allocator_scheme_ = MemoryAllocator::L1_BANKING;
    int id_;
    std::unique_ptr<Allocator> allocator_ = nullptr;
    bool initialized_ = false;

    std::set<CoreCoord> storage_only_cores_;
    std::set<CoreCoord> dispatch_cores_;
};

}  // namespace tt_metal

}  // namespace tt
