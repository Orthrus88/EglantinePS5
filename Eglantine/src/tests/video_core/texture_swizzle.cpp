// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The PS5 decoder converts every RGBA video frame from linear pixels into the console's native
// block-linear layout. The optimized implementation copies 16-byte units; compare it with the
// original pixel-at-a-time mapping so an offset error cannot silently corrupt emulator memory.

#include <bit>
#include <cstring>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/alignment.h"
#include "common/bit_util.h"
#include "common/div_ceil.h"
#include "video_core/textures/decoders.h"

namespace {

using namespace Tegra::Texture;

constexpr u32 NativeGobOffset(u32 offset, bool native_tiling) {
    if (!native_tiling) {
        return offset;
    }
    const u32 difference = ((offset >> 5) ^ (offset >> 6)) & 1;
    return offset ^ (difference << 5) ^ (difference << 6);
}

template <u32 mask>
constexpr u32 Pdep(u32 value) {
    u32 result{};
    u32 remaining = mask;
    for (u32 bit = 1; remaining != 0; bit += bit) {
        if ((value & bit) != 0) {
            result |= remaining & (~remaining + 1);
        }
        remaining &= remaining - 1;
    }
    return result;
}

template <u32 mask, u32 increment>
void IncrPdep(u32& value) {
    static constexpr u32 swizzled_increment = Pdep<mask>(increment);
    value = ((value | ~mask) + swizzled_increment) & mask;
}

template <bool ToSwizzled, u32 BytesPerPixel>
void ReferenceImpl(std::span<u8> output, std::span<const u8> input, u32 width, u32 height,
                   u32 depth, u32 block_height, u32 block_depth, u32 stride,
                   bool native_tiling) {
    const u32 pitch = width * BytesPerPixel;
    const u32 gobs_in_x = Common::DivCeilLog2(stride, GOB_SIZE_X_SHIFT);
    const u32 block_size = gobs_in_x << (GOB_SIZE_SHIFT + block_height + block_depth);
    const u32 slice_size =
        Common::DivCeilLog2(height, block_height + GOB_SIZE_Y_SHIFT) * block_size;
    const u32 block_height_mask = (1U << block_height) - 1;
    const u32 block_depth_mask = (1U << block_depth) - 1;
    const u32 x_shift = GOB_SIZE_SHIFT + block_height + block_depth;

    for (u32 slice = 0; slice < depth; ++slice) {
        const u32 offset_z = (slice >> block_depth) * slice_size +
                             ((slice & block_depth_mask) << (GOB_SIZE_SHIFT + block_height));
        for (u32 line = 0; line < height; ++line) {
            const u32 swizzled_y = Pdep<SWIZZLE_Y_BITS>(line);
            const u32 block_y = line >> GOB_SIZE_Y_SHIFT;
            const u32 offset_y = (block_y >> block_height) * block_size +
                                 ((block_y & block_height_mask) << GOB_SIZE_SHIFT);
            u32 swizzled_x{};
            for (u32 column = 0; column < width;
                 ++column, IncrPdep<SWIZZLE_X_BITS, BytesPerPixel>(swizzled_x)) {
                const u32 x = column * BytesPerPixel;
                const u32 offset_x = (x >> GOB_SIZE_X_SHIFT) << x_shift;
                const u32 swizzled_offset = NativeGobOffset(
                    offset_z + offset_y + offset_x + (swizzled_x | swizzled_y), native_tiling);
                const u32 linear_offset =
                    slice * pitch * height + line * pitch + column * BytesPerPixel;
                u8* const destination =
                    &output[ToSwizzled ? swizzled_offset : linear_offset];
                const u8* const source =
                    &input[ToSwizzled ? linear_offset : swizzled_offset];
                std::memcpy(destination, source, BytesPerPixel);
            }
        }
    }
}

template <bool ToSwizzled>
void ReferenceSwizzle(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                      u32 width, u32 height, u32 depth, u32 block_height, u32 block_depth,
                      u32 stride_alignment, bool native_tiling) {
    const u32 stride = Common::AlignUpLog2(width, stride_alignment) * bytes_per_pixel;
    const u32 new_bpp =
        (std::min)(4U, static_cast<u32>(std::countr_zero(width * bytes_per_pixel)));
    const u32 normalized_width = (width * bytes_per_pixel) >> new_bpp;
    const auto run = [&](auto bpp) {
        ReferenceImpl<ToSwizzled, decltype(bpp)::value>(
            output, input, normalized_width, height, depth, block_height, block_depth, stride,
            native_tiling);
    };
    switch (1U << new_bpp) {
    case 1:
        run(std::integral_constant<u32, 1>{});
        break;
    case 2:
        run(std::integral_constant<u32, 2>{});
        break;
    case 4:
        run(std::integral_constant<u32, 4>{});
        break;
    case 8:
        run(std::integral_constant<u32, 8>{});
        break;
    default:
        run(std::integral_constant<u32, 16>{});
        break;
    }
}

std::vector<u8> RandomBytes(size_t size, u32 seed) {
    std::mt19937 random{seed};
    std::vector<u8> bytes(size);
    for (u8& byte : bytes) {
        byte = static_cast<u8>(random() & 0xff);
    }
    return bytes;
}

void CheckRgbaShape(u32 width, u32 height, bool native_tiling, u32 seed) {
    constexpr u32 BytesPerPixel = 4;
    constexpr u32 BlockHeight = 4;
    constexpr u32 BlockDepth = 0;
    constexpr u32 StrideAlignment = 1;
    const size_t linear_size = static_cast<size_t>(width) * height * BytesPerPixel;
    const size_t tiled_size =
        CalculateSize(true, BytesPerPixel, width, height, 1, BlockHeight, BlockDepth);
    constexpr size_t Slack = size_t{1} << 20;

    const auto linear = RandomBytes(linear_size + Slack, seed);
    std::vector<u8> optimized_tiled(tiled_size + Slack);
    std::vector<u8> reference_tiled(tiled_size + Slack);
    SwizzleTexture(optimized_tiled, linear, BytesPerPixel, width, height, 1, BlockHeight,
                   BlockDepth, StrideAlignment, native_tiling);
    ReferenceSwizzle<true>(reference_tiled, linear, BytesPerPixel, width, height, 1, BlockHeight,
                           BlockDepth, StrideAlignment, native_tiling);
    REQUIRE(optimized_tiled == reference_tiled);

    std::vector<u8> optimized_linear(linear_size + Slack);
    std::vector<u8> reference_linear(linear_size + Slack);
    UnswizzleTexture(optimized_linear, optimized_tiled, BytesPerPixel, width, height, 1,
                     BlockHeight, BlockDepth, StrideAlignment, native_tiling);
    ReferenceSwizzle<false>(reference_linear, optimized_tiled, BytesPerPixel, width, height, 1,
                            BlockHeight, BlockDepth, StrideAlignment, native_tiling);
    REQUIRE(optimized_linear == reference_linear);
}

} // namespace

TEST_CASE("RGBA video swizzle matches the per-pixel reference", "[video_core][textures]") {
    u32 seed{};
    for (const bool native_tiling : {false, true}) {
        for (const auto [width, height] : {
                 std::pair{8U, 8U}, std::pair{17U, 9U}, std::pair{320U, 180U},
                 std::pair{1280U, 720U}, std::pair{1920U, 1080U}}) {
            CheckRgbaShape(width, height, native_tiling, ++seed);
        }
    }
}
