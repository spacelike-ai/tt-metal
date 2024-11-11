// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/linear_op.hpp"

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/tensor_initializers.hpp"

void compare_tensors(const ttnn::Tensor& t1, const ttnn::Tensor& t2, float eps) {
    ASSERT_EQ(t1.get_shape(), t2.get_shape());
    auto t1_vec = ttml::core::to_vector(t1);
    auto t2_vec = ttml::core::to_vector(t2);
    ASSERT_EQ(t1_vec.size(), t2_vec.size());
    bool all_equals = true;
    for (size_t i = 0; i < t1_vec.size() && all_equals; i++) {
        if (std::abs(t1_vec[i] - t2_vec[i]) > eps) {
            all_equals = false;
            EXPECT_NEAR(t1_vec[i], t2_vec[i], eps);
        }
    }
    EXPECT_TRUE(all_equals);
}

bool compare_tensors_for_broken(const ttnn::Tensor& t1, const ttnn::Tensor& t2, float eps) {
    if (t1.get_shape() != t2.get_shape()) {
        return false;
    }

    auto t1_vec = ttml::core::to_vector(t1);
    auto t2_vec = ttml::core::to_vector(t2);
    bool all_equals = true;
    for (size_t i = 0; i < t1_vec.size() && all_equals; i++) {
        if (std::abs(t1_vec[i] - t2_vec[i]) > eps) {
            all_equals = false;
        }
    }
    return all_equals;
}

TEST(LinearOpTest, TTNNBackwardGoodShape) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor = ttml::autograd::create_tensor();
    ttml::init::uniform_init(tensor, ttml::core::create_shape({64, 1, 256, 64}), ttml::init::UniformRange{-0.1F, 0.1F});

    auto weight = ttml::autograd::create_tensor();
    ttml::init::uniform_init(weight, ttml::core::create_shape({1, 1, 64, 64}), ttml::init::UniformRange{-0.1F, 0.1F});

    auto bias = ttml::autograd::create_tensor();
    ttml::init::uniform_init(bias, ttml::core::create_shape({1, 1, 1, 64}), ttml::init::UniformRange{-0.1F, 0.1F});

    auto out = ttml::autograd::create_tensor();
    ttml::init::uniform_init(out, ttml::core::create_shape({64, 1, 256, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    out->set_grad(out->get_value());

    ttml::ops::ttnn_linear_backward(tensor, weight, bias, out, ttml::core::ComputeKernelConfig::precise());
    auto ttnn_tensor_grad = tensor->get_grad();
    auto ttnn_weight_grad = weight->get_grad();
    auto ttnn_bias_grad = bias->get_grad();
    tensor->set_grad(ttnn::Tensor());
    weight->set_grad(ttnn::Tensor());
    bias->set_grad(ttnn::Tensor());

    ttml::ops::moreh_linear_backward(tensor, weight, bias, out, ttml::core::ComputeKernelConfig::precise());
    auto moreh_tensor_grad = tensor->get_grad();
    auto moreh_weight_grad = weight->get_grad();
    auto moreh_bias_grad = bias->get_grad();

    const float eps = 2e-2F;
    compare_tensors(ttnn_tensor_grad, moreh_tensor_grad, eps);
    compare_tensors(ttnn_weight_grad, moreh_weight_grad, eps);
    compare_tensors(ttnn_bias_grad, moreh_bias_grad, eps);
}

// Currently raises SEGFAULT

// TEST(LinearOpTest, TTNNBackwardBadShape_BROKEN) {
//     auto* device = &ttml::autograd::ctx().get_device();
//     auto tensor = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(tensor, ttml::core::create_shape({128, 1, 1, 128}), ttml::init::UniformRange{-0.1F,
//     0.1F});

//     auto weight = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(weight, ttml::core::create_shape({1, 1, 256, 128}), ttml::init::UniformRange{-0.1F,
//     0.1F});

//     auto bias = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(bias, ttml::core::create_shape({1, 1, 1, 256}), ttml::init::UniformRange{-0.1F, 0.1F});

//     auto out = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(out, ttml::core::create_shape({128, 1, 1, 256}), ttml::init::UniformRange{-0.1F, 0.1F});
//     out->set_grad(out->get_value());

//     ttml::ops::ttnn_linear_backward(tensor, weight, bias, out);
//     auto ttnn_tensor_grad = tensor->get_grad();
//     auto ttnn_weight_grad = weight->get_grad();
//     auto ttnn_bias_grad = bias->get_grad();
//     tensor->set_grad(ttnn::Tensor());
//     weight->set_grad(ttnn::Tensor());
//     bias->set_grad(ttnn::Tensor());

//     ttml::ops::moreh_linear_backward(tensor, weight, bias, out);
//     auto moreh_tensor_grad = tensor->get_grad();
//     auto moreh_weight_grad = weight->get_grad();
//     auto moreh_bias_grad = bias->get_grad();

//     const float eps = 2e-2F;
//     bool success = compare_tensors_for_broken(ttnn_tensor_grad, moreh_tensor_grad, eps) &&
//                    compare_tensors_for_broken(ttnn_weight_grad, moreh_weight_grad, eps) &&
//                    compare_tensors_for_broken(ttnn_bias_grad, moreh_bias_grad, eps);
//     EXPECT_FALSE(success);
// }