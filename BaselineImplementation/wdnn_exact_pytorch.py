from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class WDNNConfig:
    input_dim: int = 86392
    hidden_dim: int = 1000
    output_dim: int = 4
    dropout_rate: float = 0.3

    # Keras BatchNorm settings recovered from saved_model.pb
    bn_epsilon: float = 1e-3
    bn_momentum_keras: float = 0.99

    # Regularizer coefficients recovered from saved_model.pb
    kernel_l1: float = 1e-4
    hidden_bias_l2: float = 1e-3
    output_bias_l2: float = 1e-1


class ExactWDNN(nn.Module):
    """PyTorch recreation of the WDNN architecture extracted from SavedModel."""

    def __init__(self, config: WDNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or WDNNConfig()

        # PyTorch BN momentum is update-weight, while Keras momentum is decay.
        bn_momentum_torch = 1.0 - self.config.bn_momentum_keras

        self.dense1 = nn.Linear(self.config.input_dim, self.config.hidden_dim, bias=True)
        self.batch_normalization = nn.BatchNorm1d(
            self.config.hidden_dim,
            eps=self.config.bn_epsilon,
            momentum=bn_momentum_torch,
            affine=True,
            track_running_stats=True,
        )
        self.dropout = nn.Dropout(p=self.config.dropout_rate)

        self.dense2 = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=True)
        self.batch_normalization_1 = nn.BatchNorm1d(
            self.config.hidden_dim,
            eps=self.config.bn_epsilon,
            momentum=bn_momentum_torch,
            affine=True,
            track_running_stats=True,
        )
        self.dropout_1 = nn.Dropout(p=self.config.dropout_rate)

        self.dense3 = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=True)
        self.batch_normalization_2 = nn.BatchNorm1d(
            self.config.hidden_dim,
            eps=self.config.bn_epsilon,
            momentum=bn_momentum_torch,
            affine=True,
            track_running_stats=True,
        )
        self.dropout_2 = nn.Dropout(p=self.config.dropout_rate)

        self.wdnn_final_layer = nn.Linear(self.config.hidden_dim, self.config.output_dim, bias=True)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.batch_normalization(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = torch.relu(x)
        x = self.batch_normalization_1(x)
        x = self.dropout_1(x)

        x = self.dense3(x)
        x = torch.relu(x)
        x = self.batch_normalization_2(x)
        x = self.dropout_2(x)

        x = self.wdnn_final_layer(x)
        x = self.output_activation(x)
        return x

    def regularization_loss(self) -> torch.Tensor:
        """Recreates the SavedModel's dense kernel L1 + bias L2 regularization terms."""
        loss = torch.zeros((), dtype=self.dense1.weight.dtype, device=self.dense1.weight.device)

        dense_layers = [self.dense1, self.dense2, self.dense3]
        for layer in dense_layers:
            loss = loss + self.config.kernel_l1 * layer.weight.abs().sum()
            loss = loss + self.config.hidden_bias_l2 * layer.bias.pow(2).sum()

        loss = loss + self.config.kernel_l1 * self.wdnn_final_layer.weight.abs().sum()
        loss = loss + self.config.output_bias_l2 * self.wdnn_final_layer.bias.pow(2).sum()
        return loss


def _safe_scalar_from_tensor_proto(tensor_proto) -> float | None:
    import tensorflow as tf

    try:
        arr = tf.make_ndarray(tensor_proto)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[0])
    except Exception:
        return None


def _load_saved_model_specs(saved_model_dir: str) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, float], Tuple[int, int]]:
    from tensorflow.core.protobuf import saved_model_pb2

    pb_path = Path(saved_model_dir) / "saved_model.pb"
    if not pb_path.exists():
        raise FileNotFoundError(f"saved_model.pb not found in {saved_model_dir}")

    saved_model = saved_model_pb2.SavedModel()
    saved_model.ParseFromString(pb_path.read_bytes())
    if not saved_model.meta_graphs:
        raise ValueError("No MetaGraph found in saved_model.pb")

    meta_graph = saved_model.meta_graphs[0]
    graph_def = meta_graph.graph_def

    var_shapes: Dict[str, Tuple[int, ...]] = {}
    for node in graph_def.node:
        if node.op != "VarHandleOp":
            continue
        if node.name.startswith("Adam/") or node.name in {"total", "count", "total_1", "count_1"}:
            continue
        shape = tuple(dim.size for dim in node.attr["shape"].shape.dim) if "shape" in node.attr else ()
        var_shapes[node.name] = shape

    const_values: Dict[str, float] = {}
    for fn in graph_def.library.function:
        if "__inference_WDNN_layer_call_and_return_conditional_losses_" not in fn.signature.name:
            continue
        for node in fn.node_def:
            if node.op != "Const" or "value" not in node.attr:
                continue
            value = _safe_scalar_from_tensor_proto(node.attr["value"].tensor)
            if value is not None:
                const_values[node.name] = value

    sig = meta_graph.signature_def["serving_default"]
    input_shape = tuple(dim.size for dim in sig.inputs["input_1"].tensor_shape.dim)
    output_shape = tuple(dim.size for dim in sig.outputs["wdnn_final_layer"].tensor_shape.dim)
    if len(input_shape) != 2 or len(output_shape) != 2:
        raise ValueError("Unexpected signature rank; expected rank-2 input/output")

    return var_shapes, const_values, (input_shape[1], output_shape[1])


def verify_matches_saved_model(model: ExactWDNN, saved_model_dir: str = "./model") -> None:
    def _close(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol

    var_shapes, const_values, (in_dim, out_dim) = _load_saved_model_specs(saved_model_dir)

    checks = []

    checks.append((model.config.input_dim == in_dim, f"input_dim {model.config.input_dim} == {in_dim}"))
    checks.append((model.config.output_dim == out_dim, f"output_dim {model.config.output_dim} == {out_dim}"))

    linear_map = {
        "dense1": model.dense1,
        "dense2": model.dense2,
        "dense3": model.dense3,
        "wdnn_final_layer": model.wdnn_final_layer,
    }
    for name, layer in linear_map.items():
        expected_kernel = var_shapes.get(f"{name}/kernel")
        expected_bias = var_shapes.get(f"{name}/bias")
        actual_kernel = (layer.in_features, layer.out_features)
        actual_bias = (layer.out_features,)
        checks.append((expected_kernel == actual_kernel, f"{name}/kernel shape {actual_kernel} == {expected_kernel}"))
        checks.append((expected_bias == actual_bias, f"{name}/bias shape {actual_bias} == {expected_bias}"))

    bn_map = {
        "batch_normalization": model.batch_normalization,
        "batch_normalization_1": model.batch_normalization_1,
        "batch_normalization_2": model.batch_normalization_2,
    }
    for name, layer in bn_map.items():
        checks.append((var_shapes.get(f"{name}/gamma") == (layer.num_features,), f"{name}/gamma shape matches"))
        checks.append((var_shapes.get(f"{name}/beta") == (layer.num_features,), f"{name}/beta shape matches"))
        checks.append((var_shapes.get(f"{name}/moving_mean") == (layer.num_features,), f"{name}/moving_mean shape matches"))
        checks.append((var_shapes.get(f"{name}/moving_variance") == (layer.num_features,), f"{name}/moving_variance shape matches"))

    expected_eps = const_values.get("batch_normalization/batchnorm/add/y")
    expected_decay = const_values.get("batch_normalization/AssignMovingAvg/decay")
    expected_rate = const_values.get("dropout/dropout/GreaterEqual/y")

    checks.append((expected_eps is not None and _close(model.batch_normalization.eps, expected_eps),
                  f"batchnorm epsilon {model.batch_normalization.eps} == {expected_eps}"))
    checks.append((expected_decay is not None and _close(model.batch_normalization.momentum, expected_decay),
                  f"batchnorm torch momentum {model.batch_normalization.momentum} == decay {expected_decay}"))
    checks.append((expected_rate is not None and _close(model.dropout.p, expected_rate),
                  f"dropout p {model.dropout.p} == {expected_rate}"))

    x = torch.zeros(2, model.config.input_dim)
    y = model.eval()(x)
    checks.append((tuple(y.shape) == (2, model.config.output_dim), f"forward output shape {tuple(y.shape)} == (2, {model.config.output_dim})"))

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        raise AssertionError("Verification failed:\n- " + "\n- ".join(failed))

    print("Verification passed: PyTorch model matches SavedModel architecture metadata.")
    for _, msg in checks:
        print(f"  - {msg}")


if __name__ == "__main__":
    model = ExactWDNN()
    verify_matches_saved_model(model, saved_model_dir="./model")