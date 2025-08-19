# open_hlo_pb.py  —— 自动识别/解析 XLA HLO/快照类 .pb（gzip压缩）
import gzip, sys
from google.protobuf.message import DecodeError

# 1) 找到可用的 hlo proto 包（TF or jaxlib，二者其一即可）
hlo_pb2 = None
xla_data_pb2 = None
errs = []

def try_import():
    global hlo_pb2, xla_data_pb2
    try:
        from tensorflow.compiler.xla.service import hlo_pb2 as _hlo_pb2
        hlo_pb2 = _hlo_pb2
    except Exception as e:
        errs.append(("tensorflow.compiler.xla.service.hlo_pb2", e))
    try:
        from tensorflow.compiler.xla import xla_data_pb2 as _xla_data_pb2
        xla_data_pb2 = _xla_data_pb2
    except Exception as e:
        errs.append(("tensorflow.compiler.xla.xla_data_pb2", e))
    if hlo_pb2 is None:
        try:
            from jaxlib.xla_extension import hlo_pb2 as _hlo_pb2
            hlo_pb2 = _hlo_pb2
        except Exception as e:
            errs.append(("jaxlib.xla_extension.hlo_pb2", e))
    if xla_data_pb2 is None:
        try:
            from jaxlib.xla_extension import xla_data_pb2 as _xla_data_pb2
            xla_data_pb2 = _xla_data_pb2
        except Exception as e:
            errs.append(("jaxlib.xla_extension.xla_data_pb2", e))

try_import()
if hlo_pb2 is None and xla_data_pb2 is None:
    print("[!] 没找到 HLO/XLA 的 proto 定义。建议安装其一：tensorflow-cpu 或包含 xla_extension 的 jaxlib。")
    for m, e in errs: print(" -", m, "->", repr(e))
    sys.exit(1)

# 2) 读取（注意：你的 .pb 是 gzip 压缩的）
fn = sys.argv[1] if len(sys.argv) > 1 else "device_mem_OOM_round1_single.pb"
with gzip.open(fn, "rb") as f:
    buf = f.read()

# 3) 依次尝试这些消息类型（覆盖最常见的几类）
candidates = []
if hlo_pb2:
    # service/hlo.proto 里常见的几种容器
    candidates += [
        ("HloModuleProto", getattr(hlo_pb2, "HloModuleProto", None)),
        ("HloProto", getattr(hlo_pb2, "HloProto", None)),
        ("HloModuleGroupProto", getattr(hlo_pb2, "HloModuleGroupProto", None)),
        ("HloSnapshotProto", getattr(hlo_pb2, "HloSnapshotProto", None)),
    ]
if xla_data_pb2:
    # 旧路径里计算表达式也可能是 XlaComputationProto
    candidates += [
        ("XlaComputationProto", getattr(xla_data_pb2, "XlaComputationProto", None)),
    ]
candidates = [(n, t) for n, t in candidates if t is not None]

if not candidates:
    print("[!] 可用候选消息类型为空（hlo_pb2/xla_data_pb2 已导入但不含常见类型）。")
    sys.exit(1)

last_err = None
for name, typ in candidates:
    try:
        msg = typ()
        msg.ParseFromString(buf)
        print(f"[✓] 用 {name} 解析成功")
        # 优先导出 HLO 文本（字段名在不同版本略有差异）
        text = ""
        for attr in ["hlo_module", "module", "ToString"]:
            if hasattr(msg, attr):
                val = getattr(msg, attr)
                text = val() if callable(val) else str(val)
                if text: break
        out = fn + f".{name}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(text if text else str(msg))
        print(f"[→] 已导出：{out}")
        sys.exit(0)
    except DecodeError as e:
        last_err = e
    except Exception as e:
        last_err = e

print("[x] 所有候选消息类型均解析失败。最后一次错误：", repr(last_err))
print("    提示：若这是 MLIR bytecode/stablehlo（非 protobuf），需用 stablehlo-translate 转为文本。")
