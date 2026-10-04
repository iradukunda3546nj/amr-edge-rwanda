"""
AMR-Edge digit recogniser: training, INT8 post-training quantisation, export.

  python src/model/train.py --data <dataset_root> [--epochs 24]

Dataset: Nikić et al., "Lightweight Digit Recognition in Smart Metering System
Using NB-IoT and Federated Learning", Future Internet 16(11):402, 2024
(github.com/dbortnik/digitsDataset, GPL-3.0). Layout: <root>/<class>/<class>_<wheel>_<frame>.png

Outputs (all generated; do not edit by hand):
  src/model/artifacts/digit_int8.tflite     TFLite full-integer model
  src/model/artifacts/report.json           FP32 vs INT8 accuracy, size, MACs
  src/web/model/digit-int8.js               Browser model (ops + int8 weights)
  src/firmware/digit_model_int8.h           Firmware model (ESP-NN parameters)
  tools/test/fixtures/golden.json           TFLite reference-kernel outputs
  tools/test/fixtures/preprocess.json       Python preprocessing outputs
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pathlib
import random
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFilter

from preprocess import OUT_H, OUT_W, normalize_digit

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "src" / "model" / "artifacts"
SEED = 1234
VAL_BLOCK = 200          # frames per block for the leakage-free split
VAL_FRACTION = 0.15


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def list_dataset(root: pathlib.Path):
    items = []
    for cls in range(10):
        for f in sorted((root / str(cls)).glob("*.png")):
            _, wheel, frame = f.stem.split("_")
            items.append((str(f), cls, int(wheel), int(frame)))
    return items


def split(items):
    """Hold out whole blocks of consecutive video frames, so near-duplicate
    neighbouring frames never straddle train and validation."""
    rng = random.Random(SEED)
    blocks = sorted({(w, fr // VAL_BLOCK) for _, _, w, fr in items})
    val_blocks = set(rng.sample(blocks, max(1, int(len(blocks) * VAL_FRACTION))))
    tr = [it for it in items if (it[2], it[3] // VAL_BLOCK) not in val_blocks]
    va = [it for it in items if (it[2], it[3] // VAL_BLOCK) in val_blocks]
    return tr, va


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Field conditions the clean dataset lacks: loose ROI placement, slight
    rotation, both print polarities, blur, glare, sensor noise, poor exposure."""
    h, w = img.shape
    pil = Image.fromarray(img)
    if rng.random() < 0.5:
        pil = Image.fromarray(255 - img)

    # Affine: output->input mapping around the crop centre.
    s = rng.uniform(0.86, 1.14)
    sx = s * rng.uniform(0.92, 1.08)
    a = math.radians(rng.uniform(-4, 4))
    tx, ty = rng.uniform(-0.09, 0.09) * w, rng.uniform(-0.07, 0.07) * h
    cx, cy = w / 2, h / 2
    a00, a01 = math.cos(a) / sx, math.sin(a) / sx
    a10, a11 = -math.sin(a) / s, math.cos(a) / s
    # PIL maps each output pixel to input: in = A @ (out - c) + c + t
    m = (a00, a01, cx - a00 * cx - a01 * cy + tx, a10, a11, cy - a10 * cx - a11 * cy + ty)
    fill = int(np.median(np.asarray(pil)))
    pil = pil.transform((w, h), Image.AFFINE, m, resample=Image.BILINEAR, fillcolor=fill)

    if rng.random() < 0.35:
        pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
    x = np.asarray(pil, dtype=np.float64) / 255.0
    x = np.clip(x, 0, 1) ** rng.uniform(0.6, 1.6)
    x = x * rng.uniform(0.5, 1.0) + rng.uniform(0.0, 0.3)
    if rng.random() < 0.25:                                   # specular glare blob
        gy, gx = rng.uniform(0, h), rng.uniform(0, w)
        yy, xx = np.mgrid[0:h, 0:w]
        x += rng.uniform(0.2, 0.6) * np.exp(-((yy - gy) ** 2 + (xx - gx) ** 2) / (2 * (rng.uniform(4, 14)) ** 2))
    x += rng.normal(0, rng.uniform(0, 0.05), x.shape)
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


# Register-style typefaces available on the build machine. Bahnschrift is a DIN
# 1451 design, the typeface family most meter registers use.
FONT_CANDIDATES = [
    "bahnschrift.ttf", "arial.ttf", "arialbd.ttf", "verdana.ttf", "verdanab.ttf", "tahoma.ttf",
    "tahomabd.ttf", "consola.ttf", "consolab.ttf", "segoeui.ttf", "segoeuib.ttf", "calibri.ttf",
    "calibrib.ttf", "cour.ttf", "courbd.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
    "DejaVuSansMono.ttf", "LiberationSans-Regular.ttf", "LiberationMono-Regular.ttf",
]
FONT_DIRS = ["C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/Library/Fonts", "/System/Library/Fonts/Supplemental"]


def find_fonts() -> list[str]:
    found = []
    for d in FONT_DIRS:
        for f in FONT_CANDIDATES:
            p = pathlib.Path(d) / f
            if p.exists():
                found.append(str(p))
    return found


REJECT = 10                        # class index for "not a digit"
CLASSES = [str(i) for i in range(10)] + ["X"]
# Marks printed beside registers: CE / MID marks, units, makers' letters, separators.
REJECT_GLYPHS = "CEMmBKHNPRSTVWXZAFUYdkLJG3%/*-+=<>?#&"


def synth_wheel(label: int, font_path: str, rng: np.random.Generator, glyph: str | None = None) -> np.ndarray:
    """Render an 80x100 odometer-wheel crop: cylindrical drum shading, a digit
    that may be partly rolled toward its neighbour, optional drum separators
    and window edges. The label is the digit covering the crop centre."""
    from PIL import ImageDraw, ImageFont

    W, H = 80, 100
    glyph_h = rng.uniform(52, 70)
    font = ImageFont.truetype(font_path, int(glyph_h * 1.35))
    bg, fg = (rng.uniform(150, 245), rng.uniform(10, 90)) if rng.random() < 0.6 else (rng.uniform(10, 70), rng.uniform(170, 250))

    img = Image.new("L", (W, H), int(bg))
    draw = ImageDraw.Draw(img)
    pitch = glyph_h * rng.uniform(1.25, 1.5)          # vertical spacing of digits on the drum
    roll = rng.uniform(-0.45, 0.45) * pitch if rng.random() < 0.35 else rng.uniform(-0.06, 0.06) * pitch
    cx = W / 2 + rng.uniform(-3, 3)
    if glyph is not None:                                        # reject sample: a lone non-digit mark
        if glyph == "3":
            glyph = "³"                                    # superscript three of "m3"
        draw.text((cx, H / 2 + rng.uniform(-8, 8)), glyph, fill=int(fg), font=font, anchor="mm")
    else:
        for k in (-1, 0, 1):
            d = (label + k) % 10
            cy = H / 2 + roll + k * pitch
            draw.text((cx, cy), str(d), fill=int(fg), font=font, anchor="mm")

    x = np.asarray(img, dtype=np.float64)
    yy = (np.arange(H) - H / 2) / (H / 2)
    x = x * (1 - rng.uniform(0.15, 0.45) * yy[:, None] ** 2)   # drum curvature falloff
    if rng.random() < 0.5:                                       # dark gaps between wheels
        for side in (0, 1):
            wd = int(rng.uniform(2, 9))
            if side == 0:
                x[:, :wd] = rng.uniform(0, 40)
            else:
                x[:, W - wd:] = rng.uniform(0, 40)
    if rng.random() < 0.3:                                       # register window edge
        e = int(rng.uniform(2, 10))
        rows = slice(0, e) if rng.random() < 0.5 else slice(H - e, H)
        x[rows, :] *= 0.25
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def _make_synthetic(args):
    label, font_path, seed = args
    rng = np.random.default_rng(seed)
    glyph = REJECT_GLYPHS[int(rng.integers(len(REJECT_GLYPHS)))] if label == REJECT else None
    return normalize_digit(augment(synth_wheel(label if glyph is None else 0, font_path, rng, glyph), rng)), label


def build_synthetic(n: int, fonts: list[str], workers: int, reject_fraction: float = 0.18, seed: int = SEED + 99):
    """Digits 0-9 plus a share of reject-class marks."""
    rng = np.random.default_rng(seed)
    labels = [REJECT if rng.random() < reject_fraction else int(rng.integers(10)) for _ in range(n)]
    jobs = [(lab, fonts[int(rng.integers(len(fonts)))], seed * 7 + i) for i, lab in enumerate(labels)]
    with Pool(workers) as pool:
        res = pool.map(_make_synthetic, jobs, chunksize=256)
    return np.stack([r[0] for r in res])[..., None], np.asarray([r[1] for r in res], dtype=np.int32)


def _make_samples(args):
    path, cls, n_aug, seed = args
    rng = np.random.default_rng(seed)
    img = load_gray(path)
    out = [normalize_digit(img)]
    out += [normalize_digit(augment(img, rng)) for _ in range(n_aug)]
    return np.stack(out), cls


def build_arrays(items, n_aug: int, workers: int):
    jobs = [(p, c, n_aug, SEED + i) for i, (p, c, _, _) in enumerate(items)]
    xs, ys = [], []
    with Pool(workers) as pool:
        for x, c in pool.imap(_make_samples, jobs, chunksize=64):
            xs.append(x)
            ys.extend([c] * len(x))
    return np.concatenate(xs)[..., None], np.asarray(ys, dtype=np.int32)


# --------------------------------------------------------------------------- #
# Model: depthwise-separable micro-CNN (MobileNetV1/V2 building blocks)
# --------------------------------------------------------------------------- #
def build_model():
    import tensorflow as tf

    L = tf.keras.layers
    inp = L.Input((OUT_H, OUT_W, 1), name="digit")
    x = L.Conv2D(16, 3, padding="same", use_bias=False)(inp)
    x = L.ReLU()(L.BatchNormalization()(x))

    def ds(x, filters, stride):
        x = L.DepthwiseConv2D(3, stride, padding="same", use_bias=False)(x)
        x = L.ReLU()(L.BatchNormalization()(x))
        x = L.Conv2D(filters, 1, use_bias=False)(x)
        return L.ReLU()(L.BatchNormalization()(x))

    x = ds(x, 32, 2)                  # 16x12
    x = ds(x, 48, 2)                  # 8x6
    x = ds(x, 64, 1)                  # 8x6
    x = L.AveragePooling2D((OUT_H // 4, OUT_W // 4))(x)
    x = L.Reshape((x.shape[-1],))(x)    # static: avoids SHAPE/STRIDED_SLICE ops in the int8 graph
    x = L.Dropout(0.2)(x)
    out = L.Dense(len(CLASSES), name="logits")(x)   # logits; softmax is applied outside the int8 graph
    return tf.keras.Model(inp, out, name="amr_digit")


def count_macs(model) -> int:
    import tensorflow as tf

    macs = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            _, h, w, c = layer.output.shape
            macs += h * w * c * layer.kernel_size[0] * layer.kernel_size[1]
        elif isinstance(layer, tf.keras.layers.Conv2D):
            _, h, w, c = layer.output.shape
            macs += h * w * c * layer.kernel_size[0] * layer.kernel_size[1] * layer.input.shape[-1]
        elif isinstance(layer, tf.keras.layers.Dense):
            macs += layer.input.shape[-1] * layer.units
    return int(macs)


# --------------------------------------------------------------------------- #
# Quantisation + reference evaluation
# --------------------------------------------------------------------------- #
def fixed_batch(model):
    """Batch-1 wrapper: the device runs one crop at a time, and a static batch
    keeps the converted graph free of dynamic SHAPE/STRIDED_SLICE/PACK ops."""
    import tensorflow as tf

    x = tf.keras.Input(shape=model.input_shape[1:], batch_size=1, name="digit")
    return tf.keras.Model(x, model(x))


def quantize(model, calib: np.ndarray) -> bytes:
    import tensorflow as tf

    def rep():
        for i in range(len(calib)):
            yield [calib[i : i + 1].astype(np.float32) / 255.0]

    conv = tf.lite.TFLiteConverter.from_keras_model(fixed_batch(model))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def ref_interpreter(tfl: bytes):
    """Reference kernels (no XNNPACK): the arithmetic TFLM and ESP-NN reproduce."""
    import tensorflow as tf

    it = tf.lite.Interpreter(
        model_content=tfl,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    it.allocate_tensors()
    return it


def run_int8(it, x_u8: np.ndarray) -> np.ndarray:
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    res = np.empty((len(x_u8), len(CLASSES)), dtype=np.int8)
    for i in range(len(x_u8)):
        it.set_tensor(inp["index"], (x_u8[i : i + 1].astype(np.int16) - 128).astype(np.int8))
        it.invoke()
        res[i] = it.get_tensor(out["index"])[0]
    return res


# --------------------------------------------------------------------------- #
# Export: parse the flatbuffer into a flat op list with precomputed requant params
# --------------------------------------------------------------------------- #
def _quantize_multiplier(m: float) -> tuple[int, int]:
    """tflite::QuantizeMultiplier (lite/kernels/internal/quantization_util.cc)."""
    if m == 0.0:
        return 0, 0
    q, shift = math.frexp(m)
    q_fixed = int(math.floor(q * (1 << 31) + 0.5))      # TfLiteRound: half away from zero
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return q_fixed, shift


def parse_tflite(tfl: bytes) -> dict:
    import tflite
    from tflite.BuiltinOperator import BuiltinOperator as BO
    from tflite.Padding import Padding

    names = {v: k for k, v in BO.__dict__.items() if not k.startswith("_")}
    model = tflite.Model.GetRootAsModel(tfl, 0)
    g = model.Subgraphs(0)

    def tensor(i):
        t = g.Tensors(i)
        q = t.Quantization()
        buf = model.Buffers(t.Buffer()).DataAsNumpy()
        return {
            "shape": t.ShapeAsNumpy().tolist(),
            "scale": q.ScaleAsNumpy().astype(np.float32) if q and q.ScaleLength() else None,
            "zp": q.ZeroPointAsNumpy().astype(np.int64) if q and q.ZeroPointLength() else None,
            "data": buf if isinstance(buf, np.ndarray) else None,
            "type": t.Type(),
        }

    def act_range(fused: int, out_t) -> tuple[int, int]:
        zp = int(out_t["zp"][0])
        if fused == 1:      # RELU
            return max(-128, zp), 127
        if fused == 3:      # RELU6
            return max(-128, zp), min(127, zp + int(round(6.0 / float(out_t["scale"][0]))))
        return -128, 127

    ops = []
    for oi in range(g.OperatorsLength()):
        op = g.Operators(oi)
        oc = model.OperatorCodes(op.OpcodeIndex())
        name = names[max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())]
        ins = [tensor(i) for i in op.InputsAsNumpy()]
        out = tensor(op.OutputsAsNumpy()[0])
        rec = {"op": name, "out_shape": out["shape"][1:]}
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            opt = (tflite.Conv2DOptions if name == "CONV_2D" else tflite.DepthwiseConv2DOptions)()
            bo = op.BuiltinOptions()
            opt.Init(bo.Bytes, bo.Pos)
            x, w, b = ins
            s_in, s_out = float(x["scale"][0]), float(out["scale"][0])
            w_scales = w["scale"].astype(np.float64)
            mult, shift = zip(*[_quantize_multiplier(float(np.float64(np.float32(s_in)) * ws / np.float64(np.float32(s_out)))) for ws in w_scales])
            amin, amax = act_range(opt.FusedActivationFunction(), out)
            rec.update(
                in_shape=x["shape"][1:], filter_shape=w["shape"],
                stride=[opt.StrideH(), opt.StrideW()], padding="SAME" if opt.Padding() == Padding.SAME else "VALID",
                depth_multiplier=opt.DepthMultiplier() if name == "DEPTHWISE_CONV_2D" else None,
                in_offset=-int(x["zp"][0]), out_offset=int(out["zp"][0]), act=[amin, amax],
                weights=w["data"].view(np.int8).reshape(w["shape"]), bias=b["data"].view(np.int32),
                mult=np.asarray(mult, np.int32), shift=np.asarray(shift, np.int32),
            )
        elif name == "AVERAGE_POOL_2D":
            opt = tflite.Pool2DOptions()
            bo = op.BuiltinOptions()
            opt.Init(bo.Bytes, bo.Pos)
            amin, amax = act_range(opt.FusedActivationFunction(), out)
            rec.update(in_shape=ins[0]["shape"][1:], filter=[opt.FilterHeight(), opt.FilterWidth()],
                       stride=[opt.StrideH(), opt.StrideW()], act=[amin, amax],
                       padding="SAME" if opt.Padding() == Padding.SAME else "VALID")
        elif name == "FULLY_CONNECTED":
            x, w, b = ins[:3]
            s_in, s_out = float(x["scale"][0]), float(out["scale"][0])
            mult, shift = zip(*[_quantize_multiplier(float(np.float64(np.float32(s_in)) * ws / np.float64(np.float32(s_out)))) for ws in w["scale"].astype(np.float64)])
            rec.update(in_features=int(np.prod(x["shape"][1:])), in_offset=-int(x["zp"][0]), out_offset=int(out["zp"][0]),
                       act=[-128, 127], weights=w["data"].view(np.int8).reshape(w["shape"]), bias=b["data"].view(np.int32),
                       mult=np.asarray(mult, np.int32), shift=np.asarray(shift, np.int32))
        elif name == "RESHAPE":
            pass
        else:
            raise ValueError(f"unsupported op {name}")
        ops.append(rec)

    gi, go = g.Tensors(g.Inputs(0)), g.Tensors(g.Outputs(0))
    return {
        "ops": ops,
        "input": {"shape": gi.ShapeAsNumpy().tolist()[1:], "scale": float(gi.Quantization().Scale(0)), "zero_point": int(gi.Quantization().ZeroPoint(0))},
        "output": {"scale": float(go.Quantization().Scale(0)), "zero_point": int(go.Quantization().ZeroPoint(0))},
    }


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def export_web(graph: dict, report: dict, path: pathlib.Path):
    ops = []
    for o in graph["ops"]:
        r = {k: v for k, v in o.items() if k not in ("weights", "bias", "mult", "shift")}
        for k in ("weights", "bias", "mult", "shift"):
            if k in o:
                r[k] = b64(o[k])
        ops.append(r)
    doc = {"format": "amr-int8-graph/1", "input": graph["input"], "output": graph["output"],
           "classes": CLASSES, "reject_class": REJECT, "ops": ops, "report": report}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "// Generated by src/model/train.py. Do not edit by hand.\n"
        "// INT8 digit recogniser (TFLite full-integer PTQ), executed by js/core/int8-engine.js.\n"
        "(function (root) { root.AMR = root.AMR || {}; root.AMR.digitModel = "
        + json.dumps(doc, separators=(",", ":"))
        + "; })(typeof self !== 'undefined' ? self : globalThis);\n"
    )


def _lead_pad(padding: str, n_in: int, k: int, stride: int, n_out: int) -> int:
    """tflite::ComputePaddingWithOffset leading pad (0 for VALID)."""
    if padding != "SAME":
        return 0
    return max((n_out - 1) * stride + k - n_in, 0) // 2


def export_firmware(graph: dict, report: dict, path: pathlib.Path):
    """C header: weight arrays plus a layer-descriptor table walked by the
    firmware's ESP-NN runner (src/firmware/digit_model.cpp)."""
    def arr(ctype, name, a, per_line=16):
        vals = [str(int(v)) for v in np.asarray(a).ravel()]
        rows = [", ".join(vals[i : i + per_line]) for i in range(0, len(vals), per_line)]
        body = ",\n    ".join(rows)
        return f"static const {ctype} {name}[{len(vals)}] __attribute__((aligned(16))) = {{\n    {body}\n}};\n"

    kinds = {"CONV_2D": "AMR_OP_CONV", "DEPTHWISE_CONV_2D": "AMR_OP_DWCONV", "AVERAGE_POOL_2D": "AMR_OP_AVGPOOL", "FULLY_CONNECTED": "AMR_OP_FC"}
    lines = [
        "// Generated by src/model/train.py. Do not edit by hand.",
        f"// INT8 digit recogniser: {report['int8_bytes']} B TFLite flatbuffer, {report['params']} parameters, {report['macs']} MACs per wheel.",
        f"// Validation accuracy: FP32 {report['val_acc_fp32']:.4f}, INT8 {report['val_acc_int8']:.4f}.",
        "// Layouts follow TFLite: conv OHWI, depthwise 1HWC, fully-connected [out][in]; activations HWC.",
        "#pragma once", "#include <stdint.h>", '#include "digit_model_types.h"', "",
        f"#define DIGIT_IN_H {graph['input']['shape'][0]}",
        f"#define DIGIT_IN_W {graph['input']['shape'][1]}",
        f"#define DIGIT_IN_ZERO_POINT ({graph['input']['zero_point']})",
        f"#define DIGIT_OUT_SCALE {graph['output']['scale']:.9g}f",
        f"#define DIGIT_OUT_ZERO_POINT ({graph['output']['zero_point']})",
        f"#define DIGIT_NUM_CLASSES {len(CLASSES)}",
        f"#define DIGIT_REJECT_CLASS {REJECT}",
        "",
    ]
    descs, biggest = [], 0
    li = 0
    for o in graph["ops"]:
        if o["op"] == "RESHAPE":
            continue
        p = f"L{li}"
        out = o["out_shape"] + [1] * (3 - len(o["out_shape"]))
        if o["op"] == "FULLY_CONNECTED":
            inn = [1, 1, o["in_features"]]
            out = [1, 1, o["out_shape"][-1]]
        else:
            inn = o["in_shape"]
        biggest = max(biggest, int(np.prod(inn)), int(np.prod(out)))
        if "weights" in o:
            lines += [arr("int8_t", f"{p}_W", o["weights"]), arr("int32_t", f"{p}_B", o["bias"]),
                      arr("int32_t", f"{p}_MULT", o["mult"], 8), arr("int32_t", f"{p}_SHIFT", o["shift"])]
            w, b, m, sh = f"{p}_W", f"{p}_B", f"{p}_MULT", f"{p}_SHIFT"
            n_q = len(o["mult"])
        else:
            w = b = m = sh = "NULL"
            n_q = 0
        if o["op"] in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            kh, kw = o["filter_shape"][1], o["filter_shape"][2]
            st = o["stride"]
            pad_t = _lead_pad(o["padding"], inn[0], kh, st[0], out[0])
            pad_l = _lead_pad(o["padding"], inn[1], kw, st[1], out[1])
        elif o["op"] == "AVERAGE_POOL_2D":
            kh, kw = o["filter"]
            st = o["stride"]
            pad_t = _lead_pad(o.get("padding", "VALID"), inn[0], kh, st[0], out[0])
            pad_l = _lead_pad(o.get("padding", "VALID"), inn[1], kw, st[1], out[1])
        else:
            kh = kw = 1
            st = [1, 1]
            pad_t = pad_l = 0
        descs.append(
            f"    {{{kinds[o['op']]}, {{{inn[0]}, {inn[1]}, {inn[2]}}}, {{{out[0]}, {out[1]}, {out[2]}}}, {kh}, {kw}, {st[0]}, {st[1]}, {pad_t}, {pad_l}, "
            f"{o.get('in_offset', 0)}, {o.get('out_offset', 0)}, {o['act'][0]}, {o['act'][1]}, {n_q}, {w}, {b}, {m}, {sh}}},")
        li += 1
    lines += [f"#define DIGIT_NUM_LAYERS {li}", f"#define DIGIT_MAX_TENSOR_BYTES {biggest}", "",
              "static const amr_layer_t DIGIT_LAYERS[DIGIT_NUM_LAYERS] = {", *descs, "};", ""]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--aug", type=int, default=3, help="augmented copies per training image")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--reuse", action="store_true", help="skip training and load artifacts/digit_fp32.keras")
    ap.add_argument("--synthetic", type=float, default=0.4, help="synthetic wheel crops, as a fraction of the real training set")
    args = ap.parse_args()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    ART.mkdir(parents=True, exist_ok=True)

    items = list_dataset(args.data)
    tr_items, va_items = split(items)
    print(f"dataset: {len(items)} crops, train {len(tr_items)}, val {len(va_items)} (block split)")

    t0 = time.time()
    x_tr, y_tr = build_arrays(tr_items, args.aug, args.workers)
    x_va, y_va = build_arrays(va_items, 0, args.workers)
    fonts = find_fonts()
    n_syn = int(len(x_tr) * args.synthetic)
    if fonts and n_syn:
        x_sy, y_sy = build_synthetic(n_syn, fonts, args.workers)
        x_tr, y_tr = np.concatenate([x_tr, x_sy]), np.concatenate([y_tr, y_sy])
        # held-out synthetic set: fonts unseen by the real dataset (cross-font generalisation)
        x_sv, y_sv = build_synthetic(2000, fonts, args.workers, seed=SEED + 4242)
    else:
        x_sv, y_sv = None, None
    print(f"preprocessed {len(x_tr)} train ({n_syn} synthetic, {len(fonts)} fonts) / {len(x_va)} val in {time.time() - t0:.0f}s")

    keras_path = ART / "digit_fp32.keras"
    if args.reuse:
        model = tf.keras.models.load_model(keras_path)
    else:
        model = build_model()
        model.summary(print_fn=lambda s: print("  " + s))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(3e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        model.fit(
            x_tr.astype(np.float32) / 255.0, y_tr, batch_size=256, epochs=args.epochs, shuffle=True,
            validation_data=(x_va.astype(np.float32) / 255.0, y_va), verbose=2,
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5),
                       tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)],
        )
        model.save(keras_path)
    fp32_acc = float(model.evaluate(x_va.astype(np.float32) / 255.0, y_va, verbose=0)[1])

    rng = np.random.default_rng(SEED)
    calib = x_tr[rng.choice(len(x_tr), 1000, replace=False)]
    tfl = quantize(model, calib)
    (ART / "digit_int8.tflite").write_bytes(tfl)
    fp32_bytes = len(tf.lite.TFLiteConverter.from_keras_model(fixed_batch(model)).convert())

    it = ref_interpreter(tfl)
    q_va = run_int8(it, x_va)
    int8_acc = float(np.mean(np.argmax(q_va, 1) == y_va))
    syn_acc = float(np.mean(np.argmax(run_int8(it, x_sv), 1) == y_sv)) if x_sv is not None else None

    graph = parse_tflite(tfl)
    weight_bytes = sum(o["weights"].nbytes + o["bias"].nbytes for o in graph["ops"] if "weights" in o)
    report = {
        "dataset": "Nikic et al. 2024 (dbortnik/digitsDataset)",
        "train_samples": int(len(x_tr)), "val_samples": int(len(x_va)),
        "params": int(model.count_params()), "macs": count_macs(model),
        "fp32_bytes": fp32_bytes, "int8_bytes": len(tfl), "int8_weight_bytes": int(weight_bytes),
        "val_acc_fp32": round(fp32_acc, 4), "val_acc_int8": round(int8_acc, 4),
        "synthetic_val_acc_int8": round(syn_acc, 4) if syn_acc is not None else None,
        "input": f"{OUT_W}x{OUT_H} grayscale",
    }
    (ART / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    export_web(graph, report, ROOT / "src" / "web" / "model" / "digit-int8.js")
    export_firmware(graph, report, ROOT / "src" / "firmware" / "digit_model_int8.h")

    # Golden vectors: JS engine must reproduce the reference-kernel logits exactly.
    fx = ROOT / "tools" / "test" / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    idx = rng.choice(len(x_va), 64, replace=False)
    golden = {"inputs": [b64((x_va[i, ..., 0].astype(np.int16) - 128).astype(np.int8)) for i in idx],
              "logits": [q_va[i].tolist() for i in idx], "labels": [int(y_va[i]) for i in idx]}
    (fx / "golden.json").write_text(json.dumps(golden))

    raw = [load_gray(va_items[i][0]) for i in rng.choice(len(va_items), 24, replace=False)]
    raw += [255 - r for r in raw[:8]]
    pre = {"cases": [{"w": r.shape[1], "h": r.shape[0], "src": b64(r), "expected": b64(normalize_digit(r))} for r in raw]}
    (fx / "preprocess.json").write_text(json.dumps(pre))
    print("exports written")


if __name__ == "__main__":
    main()
