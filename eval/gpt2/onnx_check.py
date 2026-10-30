"""Validate the exported GPT-2 ONNX model with the ONNX checker and strict shape inference."""
import collections, sys, time
import onnx
path = sys.argv[1]
start = time.time()
onnx.checker.check_model(path, full_check=True)
print(f"check_model(full_check=True): passed in {time.time()-start:.1f}s")
start = time.time()
inferred = onnx.shape_inference.infer_shapes_path(path, path + ".inferred", check_type=True, strict_mode=True, data_prop=True)
print(f"infer_shapes strict: passed in {time.time()-start:.1f}s")
model = onnx.load(path, load_external_data=False)
ops = collections.Counter(n.op_type for n in model.graph.node)
print("opset", [(o.domain or "ai.onnx", o.version) for o in model.opset_import], "ir", model.ir_version)
print("nodes", len(model.graph.node), "initializers", len(model.graph.initializer))
print("op types", dict(ops.most_common()))
