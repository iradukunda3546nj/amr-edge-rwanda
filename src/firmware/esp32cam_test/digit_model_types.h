// Layer descriptor consumed by digit_model.cpp. The table itself is generated
// into digit_model_int8.h by src/model/train.py.
#pragma once
#include <stdint.h>
#include <stddef.h>

typedef enum { AMR_OP_CONV, AMR_OP_DWCONV, AMR_OP_AVGPOOL, AMR_OP_FC } amr_op_t;

typedef struct {
    amr_op_t op;
    int16_t in[3];         // H, W, C
    int16_t out[3];        // H, W, C
    int16_t k_h, k_w;      // kernel / pool window
    int16_t stride_h, stride_w;
    int16_t pad_t, pad_l;  // leading padding (TFLite SAME rule)
    int32_t in_offset;     // -input_zero_point
    int32_t out_offset;    // output_zero_point
    int32_t act_min, act_max;
    int32_t n_quant;       // 1 = per-tensor, out_c = per-channel
    const int8_t *w;
    const int32_t *bias;
    const int32_t *mult;
    const int32_t *shift;
} amr_layer_t;
