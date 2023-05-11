#include <chrono>
#include <fstream>
#include <errno.h>
#include <random>
#include <algorithm>

#include "llrt/llrt.hpp"
#include "utils.hpp"
#include "common/logger.hpp"
#include "test_libs/tiles.hpp"

using RAMData = std::vector<uint32_t>;

//! dram sweep with callback for execution
void sweep_dram(
    tt_cluster *cluster,
    std::function<void(const tt_target_dram&, const unsigned int& block)> callback) {

    int sub_channel = 0;
    for (const auto& chip_id : cluster->get_all_chips()) {
        const auto& sdesc = cluster->get_soc_desc(chip_id);
        for (unsigned int channel = 0; channel < sdesc.get_num_dram_channels(); channel++) {
            for (unsigned int block = 0; block < sdesc.get_num_dram_blocks_per_channel(); block++) {
                tt_target_dram dram_core({static_cast<int>(chip_id), channel, sub_channel});
                callback(dram_core, block);
            }
        }
    }
}

bool dram_rdwr_check(tt_cluster *cluster, unsigned start_address, std::size_t data_size) {
    RAMData actual_vec;
    std::size_t vec_size = data_size / sizeof(uint32_t);
    RAMData expected_vec = tt::tiles_test::create_random_vec<RAMData>(vec_size, tt::tiles_test::get_seed_from_systime());

    // Sweep and write to all dram with expected vector
    sweep_dram(
        cluster,
        [cluster, &expected_vec, start_address]
        (const tt_target_dram& dram_core, const unsigned int& block){
            int chip_id, channel, sub_channel;
            std::tie(chip_id, channel, sub_channel) = dram_core;
            log_debug(tt::LogTest, "Writing to chip_id={} channel={} sub_channel={} start_address={}",
                chip_id,
                channel,
                sub_channel,
                start_address);
            cluster->write_dram_vec(
                expected_vec,
                dram_core,
                start_address); // write to address
            log_debug(tt::LogTest, "Done writing to chip_id={} channel={} sub_channel={} start_address={}",
                chip_id,
                channel,
                sub_channel,
                start_address);
        });

    bool all_are_equal = true;
    sweep_dram(
        cluster,
        [cluster, &expected_vec, &actual_vec, start_address, data_size, &all_are_equal]
        (const tt_target_dram& dram_core, const unsigned int& block){
            int chip_id, channel, sub_channel;
            std::tie(chip_id, channel, sub_channel) = dram_core;
            log_debug(tt::LogTest, "Reading from chip_id={} channel={} sub_channel={} start_address={}",
                chip_id,
                channel,
                sub_channel,
                start_address);
            cluster->read_dram_vec(actual_vec, dram_core, start_address, data_size); // read size is in bytes
            log_debug(tt::LogTest, "Done reading from chip_id={} channel={} sub_channel={} start_address={}",
                chip_id,
                channel,
                sub_channel,
                start_address);
            log_debug(tt::LogVerif, "expected vec size = {}", expected_vec.size());
            log_debug(tt::LogVerif, "actual vec size   = {}", actual_vec.size());
            bool are_equal = actual_vec == expected_vec;

            all_are_equal &= are_equal;
            if (are_equal){
                log_debug(tt::LogTest, "chip_id {} channel {} sub_channel {} has passed",
                    chip_id,
                    channel,
                    sub_channel);
            }
            else {
                log_error(tt::LogTest, "chip_id {} channel {} sub_channel {} has not passed",
                    chip_id,
                    channel,
                    sub_channel);
            }
            std::fill(actual_vec.begin(), actual_vec.end(), 0);
        });
    return all_are_equal;
}

int main(int argc, char** argv)
{
    bool pass = true;

    const std::string output_dir = ".";

    std::vector<std::string> input_args(argv, argv + argc);
    bool short_mode = false;
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        std::tie(short_mode, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--short");
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    const std::string sdesc_file = get_soc_description_file(arch, target_type);


    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);

        //cluster->start_device({.init_device = false}); // works on 2/3 machines
        cluster->start_device(default_params); // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);

        std::size_t chunk_size =  1024 * 1024 * 1024;
        unsigned total_chunks = 1024 * 1024 * 1024 / chunk_size;
        if (short_mode) {
            chunk_size =  1024;
            total_chunks = 20 * 1024 / chunk_size;
        }

        for (int chunk_num = 0; chunk_num < total_chunks; chunk_num++) {
            int start_address = chunk_size * chunk_num;
            log_debug(tt::LogTest, "Testing chunk #{}/{}", chunk_num + 1, total_chunks);
            TT_ASSERT(dram_rdwr_check(cluster, start_address, chunk_size));
        }

        cluster->close_device();
        delete cluster;

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
