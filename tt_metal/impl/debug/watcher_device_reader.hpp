// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::watcher {

#define GET_WATCHER_DEV_ADDR_FOR_CORE(dev, core, sub_type)              \
    (dev->get_dev_addr(core, HalMemAddrType::WATCHER) + offsetof(watcher_msg_t, sub_type))

constexpr uint64_t DEBUG_SANITIZE_NOC_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_NOC_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_16 = 0xbada;
constexpr uint8_t DEBUG_SANITIZE_NOC_SENTINEL_OK_8 = 0xda;

// Struct containing relevant info for stack usage
typedef struct {
    CoreCoord core;
    uint16_t stack_usage;
    uint16_t kernel_id;
} stack_usage_info_t;

class WatcherDeviceReader {
    public:
     WatcherDeviceReader(
         FILE *f,
         Device *device,
         vector<string> &kernel_names,
         void (* set_watcher_exception_message)(const string &)) :
         f(f),
         device(device),
         kernel_names(kernel_names),
         set_watcher_exception_message(set_watcher_exception_message) {}
     void Dump(FILE *file = nullptr);

    private:
    // Functions for dumping each watcher feature to the log
    void DumpCore(CoreDescriptor logical_core, bool is_active_eth_core);
    void DumpL1Status(CoreCoord core, const launch_msg_t *launch_msg);
    void DumpNocSanitizeStatus(CoreCoord core, const string &core_str, const mailboxes_t *mbox_data, int noc);
    void DumpAssertStatus(CoreCoord core, const string &core_str, const mailboxes_t *mbox_data);
    void DumpPauseStatus(CoreCoord core, const string &core_str,const mailboxes_t *mbox_data);
    void DumpRingBuffer(CoreCoord core, const mailboxes_t *mbox_data, bool to_stdout);
    void DumpRunState(CoreCoord core, const launch_msg_t *launch_msg, uint32_t state);
    void DumpLaunchMessage(CoreCoord core, const mailboxes_t *mbox_data);
    void DumpWaypoints(CoreCoord core, const mailboxes_t *mbox_data, bool to_stdout);
    void DumpSyncRegs(CoreCoord core);
    void DumpStackUsage(CoreCoord core, const mailboxes_t *mbox_data);
    void ValidateKernelIDs(CoreCoord core, const launch_msg_t *launch);

    // Helper functions
    void LogRunningKernels(const launch_msg_t *launch_msg);
    string GetKernelName(CoreCoord core, const launch_msg_t *launch_msg, uint32_t type);

    FILE *f;
    Device *device;
    vector<string> &kernel_names;
    void (* set_watcher_exception_message)(const string &);

    // Information that needs to be kept around on a per-dump basis
    std::set<std::pair<CoreCoord, riscv_id_t>> paused_cores;
    std::map<riscv_id_t, stack_usage_info_t> highest_stack_usage;
    std::map<int, bool> used_kernel_names;
};

}  // namespace tt::watcher