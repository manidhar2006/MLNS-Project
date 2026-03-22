import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2


def extract_savedmodel(pb_path: str, out_dir: str) -> dict:
    pb = Path(pb_path)
    if not pb.exists():
        raise FileNotFoundError(f"SavedModel protobuf not found: {pb}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = pb.read_bytes()
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(data)

    if not sm.meta_graphs:
        raise RuntimeError("No MetaGraph found in saved_model.pb")

    mg = sm.meta_graphs[0]
    graph_def = mg.graph_def

    arch = {
        "saved_model_schema_version": sm.saved_model_schema_version,
        "tags": list(mg.meta_info_def.tags),
        "signature_def_keys": list(mg.signature_def.keys()),
        "num_nodes": len(graph_def.node),
    }
    (out / "architecture_summary.json").write_text(json.dumps(arch, indent=2))

    nodes = []
    for n in graph_def.node:
        nodes.append({"name": n.name, "op": n.op, "inputs": list(n.input)})
    (out / "architecture_nodes.json").write_text(json.dumps(nodes, indent=2))

    # Only constant tensors are recoverable from pb alone.
    weights = {}
    const_nodes = 0
    for n in graph_def.node:
        if n.op == "Const" and "value" in n.attr:
            const_nodes += 1
            arr = tf.make_ndarray(n.attr["value"].tensor)
            if arr.size > 0:
                weights[n.name] = arr

    embedded_weights_file = None
    if weights:
        embedded_weights_file = str(out / "embedded_weights.npz")
        np.savez_compressed(embedded_weights_file, **weights)

    var_ops = {"VarHandleOp", "VariableV2", "ReadVariableOp", "AssignVariableOp", "ResourceGather"}
    var_nodes = [n.name for n in graph_def.node if n.op in var_ops]
    (out / "variable_nodes.txt").write_text("\n".join(var_nodes))

    # Variable shapes allow us to infer neuron counts even without checkpoint files.
    var_shapes = {}
    for n in graph_def.node:
        if n.op == "VarHandleOp" and "shape" in n.attr:
            shape = [d.size for d in n.attr["shape"].shape.dim]
            var_shapes[n.name] = shape
    (out / "variable_shapes.json").write_text(json.dumps(var_shapes, indent=2))

    dense_layers = []
    for n in graph_def.node:
        if n.op != "VarHandleOp":
            continue
        if n.name.startswith("Adam/"):
            continue
        if not n.name.endswith("/kernel"):
            continue

        shape = [d.size for d in n.attr["shape"].shape.dim]
        if len(shape) != 2:
            continue

        dense_layers.append(
            {
                "layer": n.name.rsplit("/", 1)[0],
                "input_features": shape[0],
                "units": shape[1],
            }
        )

    layer_sizes = {
        "input_features": dense_layers[0]["input_features"] if dense_layers else None,
        "dense_layers": dense_layers,
        "output_units": dense_layers[-1]["units"] if dense_layers else None,
    }
    (out / "layer_sizes.json").write_text(json.dumps(layer_sizes, indent=2))

    # Deep audit from function library (captures ops hidden behind StatefulPartitionedCall).
    fn_lib = graph_def.library.function
    fn_op_counts = Counter()
    dropout_consts = []
    batchnorm_consts = []
    regularizers = []
    activations = Counter()
    dropout_layers = set()

    for fn in fn_lib:
        for n in fn.node_def:
            fn_op_counts[n.op] += 1

            if n.op in {"Relu", "Relu6", "Sigmoid", "Softmax", "Tanh"}:
                activations[n.op] += 1

            if "dropout" in n.name.lower() and n.name.count("/") > 0:
                for token in n.name.split("/"):
                    if token.startswith("dropout"):
                        dropout_layers.add(token)
                        break

            if n.op == "Const" and "value" in n.attr:
                val = tf.make_ndarray(n.attr["value"].tensor)

                if "dropout/GreaterEqual/y" in n.name:
                    dropout_consts.append(
                        {
                            "function": fn.signature.name,
                            "name": n.name,
                            "value": float(val),
                            "meaning": "dropout_rate",
                        }
                    )
                elif "dropout/Const" in n.name:
                    dropout_consts.append(
                        {
                            "function": fn.signature.name,
                            "name": n.name,
                            "value": float(val),
                            "meaning": "dropout_scale_1_over_keep_prob",
                        }
                    )

                if "batchnorm/add/y" in n.name:
                    batchnorm_consts.append(
                        {
                            "function": fn.signature.name,
                            "name": n.name,
                            "value": float(val),
                            "meaning": "epsilon",
                        }
                    )
                elif "AssignMovingAvg" in n.name and "decay" in n.name:
                    batchnorm_consts.append(
                        {
                            "function": fn.signature.name,
                            "name": n.name,
                            "value": float(val),
                            "meaning": "moving_average_update_factor",
                        }
                    )

                if "Regularizer/mul/x" in n.name:
                    regularizers.append(
                        {
                            "function": fn.signature.name,
                            "name": n.name,
                            "value": float(val),
                        }
                    )

    def _first_value(items, meaning):
        for item in items:
            if item.get("meaning") == meaning:
                return item["value"]
        return None

    dropout_rate = _first_value(dropout_consts, "dropout_rate")
    dropout_scale = _first_value(dropout_consts, "dropout_scale_1_over_keep_prob")
    keep_prob = None
    if dropout_scale and dropout_scale != 0:
        keep_prob = 1.0 / dropout_scale

    bn_epsilon = _first_value(batchnorm_consts, "epsilon")
    bn_update_factor = _first_value(batchnorm_consts, "moving_average_update_factor")
    bn_momentum = None
    if bn_update_factor is not None:
        # moving <- moving - (moving - batch_stat) * update_factor
        bn_momentum = 1.0 - bn_update_factor

    model_audit = {
        "model_name_hint": "WDNN",
        "input_features": layer_sizes["input_features"],
        "output_units": layer_sizes["output_units"],
        "dense_layers": dense_layers,
        "hidden_activation": "Relu" if activations.get("Relu", 0) > 0 else None,
        "output_activation": "Sigmoid" if activations.get("Sigmoid", 0) > 0 else None,
        "dropout": {
            "used": len(dropout_layers) > 0,
            "dropout_layers": sorted(dropout_layers),
            "rate": dropout_rate,
            "keep_prob": keep_prob,
        },
        "batch_normalization": {
            "used": len([k for k in var_shapes if k.startswith("batch_normalization") and "/" in k]) > 0,
            "count": len({k.split("/")[0] for k in var_shapes if k.startswith("batch_normalization") and "/" in k}),
            "epsilon": bn_epsilon,
            "momentum": bn_momentum,
            "gamma_beta_moving_stats": True,
        },
        "regularization_constants": regularizers,
        "function_op_counts_top": fn_op_counts.most_common(20),
        "notes": [
            "Graph stores many training functions; serving uses signature 'serving_default'.",
            "Output activation is Sigmoid, usually used for multilabel probabilities.",
            "Adam optimizer slot variables are present in the SavedModel graph.",
        ],
    }
    (out / "model_audit.json").write_text(json.dumps(model_audit, indent=2))

    report = {
        "saved_model_path": str(pb),
        "has_meta_graph": True,
        "meta_graph_count": len(sm.meta_graphs),
        "graph_nodes": len(graph_def.node),
        "const_nodes": const_nodes,
        "embedded_weight_tensors_extracted": len(weights),
        "embedded_weights_file": embedded_weights_file,
        "variable_related_nodes": len(var_nodes),
        "variable_node_examples": var_nodes[:20],
        "inferred_dense_layers": dense_layers,
        "layer_sizes_file": str(out / "layer_sizes.json"),
        "model_audit_file": str(out / "model_audit.json"),
        "note": "Full trained weights require SavedModel variables checkpoint files.",
    }
    (out / "extraction_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = extract_savedmodel("baseline/saved_model.pb", "baseline/extracted")
    print(json.dumps(report, indent=2))
