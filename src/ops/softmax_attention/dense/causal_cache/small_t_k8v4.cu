// ninfer::ops::detail - asymmetric FP8-K/NVFP4-V split-KV small-T launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_k8v4.cuh"
#include "ops/kv_cache/plane_types.h"
#if defined(NINFER_TIER_VOLTA)
#    include "ops/softmax_attention/dense/causal_cache/small_t_sm70.cuh"
#endif

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_k8v4_partial(const Tensor& q, CacheInput input, const Tensor& positions, float scale,
                         PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                         std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                         Tensor& partial_m, Tensor& partial_l, cudaStream_t stream,
                         std::int32_t split_begin, std::int32_t split_end) {
    // The splits this launch covers, and the capacity their partial slots step by (`splits`). A
    // whole-range launch is `[0, splits)` with the capacity equal to the grid width, exactly as a
    // launch that covered every split before a caller could ask for a range.
    const std::int32_t span = split_end - split_begin;
    if (span <= 0) { return; }
#if defined(NINFER_TIER_VOLTA)
    if (current_device_volta()) {
        if (span != splits) {
            throw std::invalid_argument(
                "a split range needs the tiled kernel; this device runs the Volta one");
        }
        // Volta's own kernel (small_t_sm70.cuh): the same grid, splits and partial outputs. The
        // tiled kernel is the other tiers'; a build of Volta alone does not compile it.
        launch_small_t_sm70_quantized<Geometry, TokenTile, MultiBatch, Masked, CacheInput, true,
                                      false>(q, input, positions, scale, cache, invocation,
                                             logical_capacity, splits, partial_acc, partial_m,
                                             partial_l, stream, invocation.partition);
        return;
    }
#endif
    if constexpr (kBuildPostVolta) {
        with_shape_set([&]<bool TuringShapes>() {
            constexpr int RowCount = TokenTile * Geometry::GroupSize;
            constexpr int RowTiles = (RowCount + 15) / 16;
            constexpr int Warps    = RowTiles == 3 ? 12 : 8;
            // KeyBlock=64 would need 5.5*64*256 = 88KiB of dynamic shared memory here on top of
            // ~30KiB static (q_s/p_s/etc), since unlike before this kernel keeps a real dequantized
            // BF16 copy of K beside its raw FP8 staging buffer -- over sm_86/sm_89's
            // ~99KiB-per-block budget (same constraint as small_t_fp8.cu; see its comment).
            // KeyBlock=32 fits comfortably (~75KiB).
            constexpr int KeyBlock  = 32;
            constexpr int MinBlocks = TokenTile == 1 ? 2 : 1;
            // k_fp8(Bc*D) + k_b16(2*Bc*D) + v_nvfp4(Bc*D/2) + v_f16(2*Bc*D) = 5.5*Bc*D bytes; a
            // compact stage widens V into k_b16 (kCausalSmallTCompactKV), 3.5*Bc*D bytes.
            constexpr std::size_t DynamicBytes =
                (kCausalSmallTCompactKV<RowTiles, KeyBlock, 11 * KeyBlock * kCausalHeadDim / 2,
                                        TuringShapes>
                     ? 7u
                     : 11u) *
                KeyBlock * kCausalHeadDim / 2u;
            static_assert(kCausalSmallTQuantizedFixedBytes<RowTiles, KeyBlock> + DynamicBytes <=
                          shape_max_shared_bytes_per_block(TuringShapes));
            using KernelInput = CacheInput;
            const dim3 grid(q.ne[1] / Geometry::GroupSize, span, invocation.batch_size);
            const auto launch = [&]() {
                auto kernel = causal_attention_small_t_k8v4_tiled_kernel<
                    Geometry, TokenTile, Warps, MinBlocks, KeyBlock, true, MultiBatch, Masked,
                    KernelInput, TuringShapes>;
                raise_dynamic_shared_memory(kernel, static_cast<int>(DynamicBytes));

                const auto q_ptr         = static_cast<const __nv_bfloat16*>(q.data);
                const auto positions_ptr = static_cast<const std::int32_t*>(positions.data);
                constexpr auto kStorage  = KvCacheStorage::Fp8KeyNvfp4Value;
                const auto cache_k_ptr   = static_cast<KvKeyCodeT<kStorage>*>(cache.k_pages.data);
                const auto cache_v_ptr   = static_cast<KvValueCodeT<kStorage>*>(cache.v_pages.data);
                const auto k_scale_ptr =
                    static_cast<KvKeyScaleT<kStorage>*>(cache.k_scale_pages.data);
                const auto v_scale_ptr =
                    static_cast<KvValueScaleT<kStorage>*>(cache.v_scale_pages.data);
                assert_kv_planes<kStorage, decltype(cache_k_ptr), decltype(cache_v_ptr),
                                 decltype(k_scale_ptr), decltype(v_scale_ptr)>();
                const auto tables_ptr = static_cast<const std::int32_t*>(cache.block_tables.data);
                const auto valid_ptr =
                    invocation.valid_columns == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.valid_columns->data);
                const auto rows_ptr =
                    invocation.table_rows == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.table_rows->data);
                auto* partial_acc_ptr = static_cast<float*>(partial_acc.data);
                auto* partial_m_ptr   = static_cast<float*>(partial_m.data);
                auto* partial_l_ptr   = static_cast<float*>(partial_l.data);
                kernel<<<grid, Warps * 32, DynamicBytes, stream>>>(
                    q_ptr, input, positions_ptr, cache_k_ptr, cache_v_ptr, k_scale_ptr, v_scale_ptr,
                    tables_ptr, valid_ptr, rows_ptr, cache.block_tables.ne[0],
                    invocation.full_width, invocation.column_begin, logical_capacity, scale,
                    partial_acc_ptr, partial_m_ptr, partial_l_ptr, invocation.partition, split_begin,
                    splits);
                CUDA_CHECK(cudaGetLastError());
            };

            launch();
        });
    }
}

template <typename Geometry, bool MultiBatch, bool Masked>
void launch_k8v4_reduce(const Tensor& positions, const CausalSmallTInvocation& invocation,
                        std::int32_t logical_capacity, std::int32_t splits,
                        const Tensor& partial_acc, const Tensor& partial_m, const Tensor& partial_l,
                        Tensor& out, cudaStream_t stream) {
    constexpr int Block = 256;
    const dim3 grid(out.ne[1], invocation.width * invocation.batch_size);
    const auto launch = [&]<bool Offset>() {
        causal_attention_small_t_k8v4_reduce_output_kernel<Geometry, MultiBatch, Masked, Offset>
            <<<grid, Block, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(positions.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, logical_capacity, splits,
                static_cast<__nv_bfloat16*>(out.data), invocation.partition,
                invocation.partial_out);
    };
    if (invocation.column_begin == 0) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_k8v4_launch_for(const Tensor& q, CacheInput input,
                                              const Tensor& positions, float scale,
                                              PagedKVBatchLayerView cache,
                                              const CausalSmallTInvocation& invocation,
                                              CausalAttentionExecutionEnvelope envelope,
                                              Tensor& partial_acc, Tensor& partial_m,
                                              Tensor& partial_l, Tensor& out, cudaStream_t stream,
                                              const CausalAttentionFrameHook* frames) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits = causal_attention_split_capacity(q.ne[1], invocation.width, cache.storage,
                                                        envelope, invocation.batch_size);

    // The splits this call covers, and the keys each of them reads, as pages of
    // kPagedKVPageSize: a split's key range is the geometry's, and a frame asks for the pages of
    // the splits it launches. The bound is conservative by a page on each side - a split starts on
    // a 32-key tile boundary, not a page one - and a page copied twice costs nothing while a page
    // not copied is a wrong result.
    const std::int32_t window_tokens = frames == nullptr ? 0 : frames->window_tokens;
    const std::int32_t active_splits =
        window_tokens <= 0
            ? splits
            : causal_attention_active_splits(q.ne[1], invocation.width, invocation.batch_size,
                                             window_tokens, cache.storage, envelope);
    const auto frame_page_range = [&](std::int32_t begin, std::int32_t end, std::int32_t& first,
                                      std::int32_t& count) {
        // Key bounds of the splits' span, in pages, with one page of slack on both sides.
        const auto key_at = [&](std::int32_t split) {
            return static_cast<std::int64_t>(split) * window_tokens / active_splits;
        };
        const std::int64_t lo = key_at(begin) - kPagedKVPageSize;
        const std::int64_t hi = key_at(end) + kPagedKVPageSize;
        const std::int64_t first_page = lo <= 0 ? 0 : lo / kPagedKVPageSize;
        const std::int64_t last_page  = (hi - 1) / kPagedKVPageSize;
        first                         = static_cast<std::int32_t>(first_page);
        count                         = static_cast<std::int32_t>(last_page - first_page + 1);
    };
    const std::int32_t frame_splits =
        frames == nullptr
            ? splits
            : (frames->splits_per_frame > 0 ? frames->splits_per_frame : splits);

    // One dispatch, because the partial launches and the reduce share it, and the frame loop sits
    // between them: every frame's partials land in the slots a whole-range launch would write,
    // then one reduce reads them in split order (design 5.13).
    const auto dispatch = [&](std::int32_t begin, std::int32_t end) {
        const auto launch_partial = [&]<int Tokens, bool MultiBatch, bool Masked>() {
            launch_k8v4_partial<Geometry, Tokens, MultiBatch, Masked>(
                q, input, positions, scale, cache, invocation, logical_capacity, splits, partial_acc,
                partial_m, partial_l, stream, begin, end);
        };
        const auto dispatch_metadata = [&]<int Tokens>() {
            const bool masked = invocation.valid_columns != nullptr;
            if (invocation.batch_size == 1) {
                if (masked) {
                    launch_partial.template operator()<Tokens, false, true>();
                } else {
                    launch_partial.template operator()<Tokens, false, false>();
                }
            } else if (masked) {
                launch_partial.template operator()<Tokens, true, true>();
            } else {
                launch_partial.template operator()<Tokens, true, false>();
            }
        };
        switch (invocation.width) {
        case 1:
            dispatch_metadata.template operator()<1>();
            break;
        case 2:
            dispatch_metadata.template operator()<2>();
            break;
        case 3:
            dispatch_metadata.template operator()<3>();
            break;
        case 4:
            dispatch_metadata.template operator()<4>();
            break;
        case 5:
            dispatch_metadata.template operator()<5>();
            break;
        case 6:
            dispatch_metadata.template operator()<6>();
            break;
        case 7:
            if constexpr (Geometry::GroupSize == 6) {
                dispatch_metadata.template operator()<7>();
                break;
            }
            throw std::invalid_argument("unsupported query-row tile");
        case 8:
            if constexpr (Geometry::GroupSize == 6) {
                dispatch_metadata.template operator()<8>();
                break;
            }
            throw std::invalid_argument("unsupported query-row tile");
        default:
            throw std::invalid_argument("causal_attention_small_t_k8v4_launch: unsupported T");
        }
    };

    if (frames == nullptr || window_tokens <= 0 || frames->splits_per_frame <= 0) {
        dispatch(0, splits);
    } else {
        for (std::int32_t frame = 0, begin = 0; begin < splits; ++frame) {
            const std::int32_t end = begin + frame_splits < splits ? begin + frame_splits : splits;
            std::int32_t first     = 0;
            std::int32_t count     = 0;
            frame_page_range(begin, end, first, count);
            // The caller turns this into its own stream wait, its copy of the frame's pages and
            // its record of the frame being read: this call owns the launches, not the link.
            if (frames->wait != nullptr) { frames->wait(frames->context, frame, first, count); }
            dispatch(begin, end);
            if (frames->consumed != nullptr) {
                frames->consumed(frames->context, frame, first, count);
            }
            begin = end;
        }
    }

    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_k8v4_reduce<Geometry, false, true>(positions, invocation, logical_capacity,
                                                      splits, partial_acc, partial_m, partial_l,
                                                      out, stream);
        } else {
            launch_k8v4_reduce<Geometry, false, false>(positions, invocation, logical_capacity,
                                                       splits, partial_acc, partial_m, partial_l,
                                                       out, stream);
        }
    } else if (masked) {
        launch_k8v4_reduce<Geometry, true, true>(positions, invocation, logical_capacity, splits,
                                                 partial_acc, partial_m, partial_l, out, stream);
    } else {
        launch_k8v4_reduce<Geometry, true, false>(positions, invocation, logical_capacity, splits,
                                                  partial_acc, partial_m, partial_l, out, stream);
    }
}

} // namespace

void causal_attention_small_t_k8v4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream,
    float* partial_out, CausalSplitRange splits, const CausalAttentionFrameHook* frames) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
        .partition     = causal_attention_key_partition(envelope, cache),
        .partial_out   = partial_out,
        .split_begin   = splits.begin,
        .split_end     = splits.end,
    };
    with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
        causal_attention_small_t_k8v4_launch_for<Geometry>(q, input, positions, scale, cache,
                                                           invocation, envelope, partial_acc,
                                                           partial_m, partial_l, out, stream,
                                                           frames);
    });
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 CausalAttentionExecutionEnvelope envelope,
                                                 Tensor& partial_acc, Tensor& partial_m,
                                                 Tensor& partial_l, Tensor& out,
                                                 cudaStream_t stream) {
    const CausalCachedInput input{};
    PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
        .partition     = causal_attention_key_partition(envelope, batch_cache),
    };
    with_causal_query_geometry(q.ne[1], [&]<typename Geometry>() {
        causal_attention_small_t_k8v4_launch_for<Geometry>(q, input, positions, scale, batch_cache,
                                                           invocation, envelope, partial_acc,
                                                           partial_m, partial_l, out, stream,
                                                           nullptr);
    });
}

} // namespace ninfer::ops::detail
