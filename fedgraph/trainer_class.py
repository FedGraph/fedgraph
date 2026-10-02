import json
import logging
import os
import random
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
import numpy as np
import ray
import tenseal as ts
import torch
import torch.nn.functional as F
import torch_geometric
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torchmetrics.functional.retrieval import retrieval_auroc
from torchmetrics.retrieval import RetrievalHitRate

from fedgraph.gnn_models import (
    GCN,
    GIN,
    GNN_LP,
    AggreGCN,
    AggreGCN_Arxiv,
    GCN_arxiv,
    SAGE_products,
)
from fedgraph.graph_storage import relabel_adjacency_to_mmap_cache
from fedgraph.resource_monitor import collect_resource_snapshot

# Threshold-HE backend is optional. We delay-import OpenFHE bindings here so the
# rest of fedgraph stays importable on systems without the OpenFHE wheel.
try:
    from fedgraph.openfhe_threshold import OpenFHEThresholdCKKS  # noqa: F401

    _OPENFHE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when openfhe is missing
    OpenFHEThresholdCKKS = None  # type: ignore[misc,assignment]
    _OPENFHE_AVAILABLE = False

from fedgraph.train_func import test, train
from fedgraph.utils_lp import (
    check_data_files_existance,
    get_data,
    get_data_loaders_per_time_step,
    get_global_user_item_mapping,
)
from fedgraph.utils_nc import get_1hop_feature_sum


def _resolve_huggingface_artifact_num_hops(args: Any) -> int:
    """Return the hop suffix used by a legacy Hugging Face artifact."""
    configured_hops = getattr(args, "hf_artifact_num_hops", None)
    if configured_hops is None or not isinstance(configured_hops, int):
        return int(args.num_hops)
    if configured_hops < 0:
        raise ValueError("hf_artifact_num_hops must be non-negative")
    return configured_hops


def _uses_legacy_huggingface_fedavg_adjacency(args: Any) -> bool:
    """Whether a FedAvg run needs legacy Hf coordinate conversion."""
    return (
        getattr(args, "method", None) == "FedAvg"
        and getattr(args, "use_huggingface", False) is True
        and _resolve_huggingface_artifact_num_hops(args) > 0
    )


def _uses_local_nc_artifact(args: Any) -> bool:
    """Whether a trainer should load one validated local NC shard."""
    artifact_root = getattr(args, "local_artifact_dir", None)
    return isinstance(artifact_root, (str, Path)) and bool(str(artifact_root))


def _uses_huggingface_local_nc_artifact(args: Any) -> bool:
    """Whether a trainer should download one manifest-style NC Hf shard."""
    repository = getattr(args, "hf_local_artifact_repo", None)
    return isinstance(repository, str) and bool(repository)


_LOCAL_ARTIFACT_COMMON_SHARD_FILES = (
    "metadata.json",
    "local_node_index.pt",
    "communicate_node_index.pt",
    "adj.pt",
    "train_labels.pt",
    "val_labels.pt",
    "test_labels.pt",
    "features.pt",
    "idx_train.pt",
    "idx_val.pt",
    "idx_test.pt",
    "global_node_num.pt",
    "class_num.pt",
)
_LOCAL_ARTIFACT_V2_EXTRA_SHARD_FILES = (
    "adj_global.pt",
    "source_degree.pt",
    "source_offsets.pt",
)
_LOCAL_ARTIFACT_SHARD_FILES = (
    *_LOCAL_ARTIFACT_COMMON_SHARD_FILES,
    *_LOCAL_ARTIFACT_V2_EXTRA_SHARD_FILES,
)

_GRAPH_STORAGE_MODES = {"device", "cpu", "mmap"}
_HOST_GRAPH_STORAGE_MODES = {"cpu", "mmap"}
_MMAP_GRAPH_ARTIFACT_FILES = {"adj.pt", "adj_global.pt", "features.pt"}


def _record_local_artifact_manifest(args: Any, manifest: dict[str, Any]) -> None:
    """Keep only the small runtime contract needed by the trainer."""
    setattr(
        args,
        "_local_artifact_manifest",
        {
            "artifact_version": manifest.get("artifact_version"),
            "hop_semantics": manifest.get("hop_semantics"),
            "adjacency_contract": manifest.get("adjacency_contract"),
        },
    )


def _record_artifact_tensor_path(
    args: Any, file_name: str, path: Union[str, Path]
) -> None:
    artifact_paths = getattr(args, "_artifact_tensor_paths", None)
    if not isinstance(artifact_paths, dict):
        artifact_paths = {}
        setattr(args, "_artifact_tensor_paths", artifact_paths)
    artifact_paths[file_name] = str(Path(path).expanduser().resolve())


def _artifact_tensor_paths(args: Any) -> dict[str, str]:
    paths = getattr(args, "_artifact_tensor_paths", None)
    return paths if isinstance(paths, dict) else {}


def _resolve_graph_storage_mode(args: Any) -> str:
    """Return the graph tensor placement policy implemented by this trainer."""
    configured_mode = getattr(args, "graph_storage_mode", "device")
    mode = configured_mode.lower() if isinstance(configured_mode, str) else "device"
    if mode not in _GRAPH_STORAGE_MODES:
        choices = ", ".join(sorted(_GRAPH_STORAGE_MODES))
        raise ValueError(f"graph_storage_mode must be one of: {choices}")
    if mode in _HOST_GRAPH_STORAGE_MODES and int(getattr(args, "num_hops", 0)) != 0:
        raise ValueError(
            f"graph_storage_mode='{mode}' currently supports only num_hops=0; "
            "bounded 1/2-hop preprocessing is deferred to the chunking stage"
        )
    return mode


def _load_artifact_tensor(
    path: Union[str, Path], *, memory_map: bool = False
) -> torch.Tensor:
    """Load one artifact tensor into CPU memory or a file-backed CPU storage."""
    tensor_path = Path(path)
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Artifact tensor is missing: {tensor_path}")
    load_options: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": True,
    }
    if memory_map:
        load_options["mmap"] = True
    try:
        tensor = torch.load(tensor_path, **load_options)
    except RuntimeError as error:
        if not memory_map:
            raise
        raise RuntimeError(
            f"Unable to memory-map artifact tensor {tensor_path}. The file must "
            "use the zip-based format produced by the current torch.save."
        ) from error
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected a tensor in {tensor_path}, found {type(tensor).__name__}"
        )
    return tensor


def _remap_legacy_huggingface_fedavg_indexes(
    local_node_index: torch.Tensor,
    communicate_node_index: torch.Tensor,
    indexes: torch.Tensor,
    index_name: str,
) -> torch.Tensor:
    """Convert legacy communication-row positions to local feature-row positions."""
    if local_node_index.ndim != 1 or communicate_node_index.ndim != 1:
        raise ValueError("legacy node-index tensors must be one-dimensional")
    if indexes.ndim != 1:
        raise ValueError(f"{index_name} must be one-dimensional")

    local_node_index = local_node_index.long()
    communicate_node_index = communicate_node_index.long()
    indexes = indexes.long()
    if local_node_index.numel() > 1 and not bool(
        torch.all(local_node_index[1:] >= local_node_index[:-1])
    ):
        raise ValueError("local_node_index must be sorted for legacy FedAvg")
    if indexes.numel() == 0:
        return indexes
    if bool(torch.any(indexes < 0)) or bool(
        torch.any(indexes >= communicate_node_index.numel())
    ):
        raise ValueError(
            f"{index_name} contains positions outside communicate_node_index"
        )

    node_ids = communicate_node_index[indexes]
    local_positions = torch.searchsorted(local_node_index, node_ids)
    in_local_range = local_positions < local_node_index.numel()
    matches = torch.zeros_like(in_local_range)
    matches[in_local_range] = (
        local_node_index[local_positions[in_local_range]] == node_ids[in_local_range]
    )
    if not bool(torch.all(matches)):
        missing_count = int((~matches).sum().item())
        raise ValueError(
            f"{index_name} contains {missing_count} nodes outside the local partition"
        )
    return local_positions


def load_trainer_data_from_hugging_face(trainer_id, args):
    artifact_num_hops = _resolve_huggingface_artifact_num_hops(args)
    memory_map_graph = _resolve_graph_storage_mode(args) == "mmap"
    repo_name = f"FedGraph/fedgraph_{args.dataset}_{args.n_trainer}trainer_{artifact_num_hops}hop_iid_beta_{args.iid_beta}_trainer_id_{trainer_id}"

    def download_and_load_tensor(file_name, optional=False):
        try:
            file_path = hf_hub_download(
                repo_id=repo_name, repo_type="dataset", filename=file_name
            )
        except EntryNotFoundError:
            if optional:
                return None
            raise
        _record_artifact_tensor_path(args, file_name, file_path)
        tensor = _load_artifact_tensor(
            file_path,
            memory_map=memory_map_graph and file_name in _MMAP_GRAPH_ARTIFACT_FILES,
        )
        print(f"Loaded {file_name}, size: {tensor.size()}")
        return tensor

    print(
        f"Loading client data {trainer_id} from {artifact_num_hops}-hop "
        "Hugging Face artifact"
    )
    local_node_index = download_and_load_tensor("local_node_index.pt")
    communicate_node_global_index = download_and_load_tensor(
        "communicate_node_index.pt"
    )
    global_edge_index_client = download_and_load_tensor("adj.pt")
    train_labels = download_and_load_tensor("train_labels.pt")
    val_labels = download_and_load_tensor("val_labels.pt", optional=True)
    test_labels = download_and_load_tensor("test_labels.pt")
    features = download_and_load_tensor("features.pt")
    in_com_train_node_local_indexes = download_and_load_tensor("idx_train.pt")
    in_com_val_node_local_indexes = download_and_load_tensor(
        "idx_val.pt", optional=True
    )
    in_com_test_node_local_indexes = download_and_load_tensor("idx_test.pt")
    if val_labels is None or in_com_val_node_local_indexes is None:
        val_labels = torch.empty(0, dtype=train_labels.dtype)
        in_com_val_node_local_indexes = torch.empty(
            0, dtype=in_com_train_node_local_indexes.dtype
        )
    global_node_num = download_and_load_tensor("global_node_num.pt", optional=True)
    class_num = download_and_load_tensor("class_num.pt", optional=True)
    if global_node_num is None or class_num is None:
        warnings.warn(
            "Hugging Face trainer data does not contain global metadata; "
            "falling back to inference for compatibility with existing repositories.",
            UserWarning,
        )
    return (
        local_node_index,
        communicate_node_global_index,
        global_edge_index_client,
        train_labels,
        val_labels,
        test_labels,
        features,
        in_com_train_node_local_indexes,
        in_com_val_node_local_indexes,
        in_com_test_node_local_indexes,
        global_node_num,
        class_num,
    )


def load_trainer_data_from_huggingface_local_artifact(
    trainer_id: int, args: Any
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Download one manifest-style Hf shard, then use the local validator.

    This deliberately does not change the historical one-repository-per-trainer
    loader. The snapshot allowlist keeps a worker from downloading sibling
    shards in a large artifact repository.
    """
    repository = getattr(args, "hf_local_artifact_repo", None)
    if not isinstance(repository, str) or not repository:
        raise ValueError("hf_local_artifact_repo is required for artifact loading")

    shard_name = f"trainer-{trainer_id:03d}"
    allow_patterns = [
        "manifest.json",
        *[
            f"shards/{shard_name}/{file_name}"
            for file_name in _LOCAL_ARTIFACT_SHARD_FILES
        ],
    ]
    download_kwargs: dict[str, Any] = {
        "repo_id": repository,
        "repo_type": "dataset",
        "allow_patterns": allow_patterns,
    }
    revision = getattr(args, "hf_local_artifact_revision", None)
    if revision:
        download_kwargs["revision"] = revision
    cache_dir = getattr(args, "hf_local_artifact_cache_dir", None)
    if cache_dir:
        download_kwargs["cache_dir"] = cache_dir

    artifact_root = Path(snapshot_download(**download_kwargs))
    print(
        "Loading client data "
        f"{trainer_id} from Hugging Face local artifact {repository}"
    )
    local_args = SimpleNamespace(
        local_artifact_dir=str(artifact_root),
        use_huggingface=False,
        num_hops=args.num_hops,
        n_trainer=args.n_trainer,
        graph_storage_mode=getattr(args, "graph_storage_mode", "device"),
        graph_relabel_chunk_edges=getattr(args, "graph_relabel_chunk_edges", 1_000_000),
    )
    loaded_data = load_trainer_data_from_local_artifact(trainer_id, local_args)
    local_paths = _artifact_tensor_paths(local_args)
    if local_paths:
        setattr(args, "_artifact_tensor_paths", local_paths)
    local_manifest = getattr(local_args, "_local_artifact_manifest", None)
    if isinstance(local_manifest, dict):
        setattr(args, "_local_artifact_manifest", local_manifest)
    return loaded_data


def load_trainer_data_from_local_artifact(trainer_id: int, args: Any) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Load and validate one complete manifest-style shard on its Ray worker."""
    configured_root = getattr(args, "local_artifact_dir", None)
    if not configured_root:
        raise ValueError("local_artifact_dir is required for local artifact loading")
    if getattr(args, "use_huggingface", False):
        raise ValueError("local_artifact_dir and use_huggingface cannot be combined")

    artifact_root = Path(configured_root).expanduser().resolve()
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Local artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_version = manifest.get("artifact_version")
    hop_semantics = manifest.get("hop_semantics")
    runtime_hops = int(getattr(args, "num_hops", 0))
    if (artifact_version, hop_semantics) not in {(1, 0), (2, 2)}:
        raise ValueError(
            "local artifact must use the version/hop contract (1, 0) or (2, 2)"
        )
    if hop_semantics != runtime_hops:
        raise ValueError(
            "local artifact hop_semantics does not match the requested experiment: "
            f"{hop_semantics} != {runtime_hops}"
        )
    _record_local_artifact_manifest(args, manifest)
    if manifest.get("n_trainer") != int(args.n_trainer):
        raise ValueError(
            "local artifact n_trainer does not match the requested experiment: "
            f"{manifest.get('n_trainer')} != {args.n_trainer}"
        )
    if trainer_id < 0 or trainer_id >= int(manifest["n_trainer"]):
        raise ValueError(f"trainer_id {trainer_id} is outside the local artifact")

    shard_dir = artifact_root / "shards" / f"trainer-{trainer_id:03d}"
    metadata_path = shard_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Local artifact metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("trainer_id") != trainer_id:
        raise ValueError("local artifact metadata has an unexpected trainer ID")

    expected_tensor_files = set(_LOCAL_ARTIFACT_COMMON_SHARD_FILES[1:])
    if artifact_version == 2:
        expected_tensor_files.update(_LOCAL_ARTIFACT_V2_EXTRA_SHARD_FILES)
        if (
            manifest.get("neighborhood_hops") != 1
            or manifest.get("message_flow") != "source_to_target"
            or manifest.get("feature_alignment") != "local_node_index"
        ):
            raise ValueError("2-hop artifact has incompatible graph semantics")
        contract = manifest.get("adjacency_contract")
        required_contract: dict[str, dict[str, object]] = {
            "pretrain_adjacency": {
                "file": "adj_global.pt",
                "coordinate_system": "global_node_id",
                "sorted_by": "source_then_input_order",
            },
            "training_adjacency": {
                "file": "adj.pt",
                "coordinate_system": "communicate_node_position",
                "mapping_file": "communicate_node_index.pt",
                "sorted_by": "source_then_input_order",
            },
            "source_degree": {
                "file": "source_degree.pt",
                "alignment": "communicate_node_position",
                "includes_self_loop": False,
            },
            "source_offsets": {
                "file": "source_offsets.pt",
                "alignment": "communicate_node_position_plus_terminal",
                "indexes": "training_adjacency_columns",
            },
        }
        if not isinstance(contract, dict) or any(
            not isinstance(contract.get(section), dict)
            or any(contract[section].get(key) != value for key, value in fields.items())
            for section, fields in required_contract.items()
        ):
            raise ValueError("2-hop artifact has an invalid adjacency contract")
    metadata_files = metadata.get("files")
    if not isinstance(metadata_files, dict) or set(metadata_files) != (
        expected_tensor_files
    ):
        raise ValueError("local artifact metadata has an unexpected tensor file list")

    memory_map_graph = _resolve_graph_storage_mode(args) == "mmap"

    def load_shard_tensor(file_name: str) -> torch.Tensor:
        _record_artifact_tensor_path(args, file_name, shard_dir / file_name)
        return _load_artifact_tensor(
            shard_dir / file_name,
            memory_map=(memory_map_graph and file_name in _MMAP_GRAPH_ARTIFACT_FILES),
        )

    local_node_index = load_shard_tensor("local_node_index.pt")
    communicate_node_index = load_shard_tensor("communicate_node_index.pt")
    adjacency = load_shard_tensor(
        "adj_global.pt" if artifact_version == 2 else "adj.pt"
    )
    train_labels = load_shard_tensor("train_labels.pt")
    val_labels = load_shard_tensor("val_labels.pt")
    test_labels = load_shard_tensor("test_labels.pt")
    features = load_shard_tensor("features.pt")
    idx_train = load_shard_tensor("idx_train.pt")
    idx_val = load_shard_tensor("idx_val.pt")
    idx_test = load_shard_tensor("idx_test.pt")
    global_node_num = load_shard_tensor("global_node_num.pt")
    class_num = load_shard_tensor("class_num.pt")

    owned_node_count = local_node_index.numel()
    communicate_node_count = communicate_node_index.numel()
    validation_chunk_size = int(getattr(args, "graph_relabel_chunk_edges", 1_000_000))
    if validation_chunk_size < 1:
        raise ValueError("graph_relabel_chunk_edges must be positive")

    def is_strictly_increasing(values: torch.Tensor) -> bool:
        previous: Optional[int] = None
        for start in range(0, values.numel(), validation_chunk_size):
            stop = min(start + validation_chunk_size, values.numel())
            chunk = values[start:stop]
            if chunk.numel() == 0:
                continue
            if previous is not None and int(chunk[0]) <= previous:
                return False
            if chunk.numel() > 1 and not bool(torch.all(chunk[1:] > chunk[:-1])):
                return False
            previous = int(chunk[-1])
        return True

    if local_node_index.ndim != 1 or communicate_node_index.ndim != 1:
        raise ValueError("local and communicate node indexes must be one-dimensional")
    if not is_strictly_increasing(local_node_index):
        raise ValueError("local_node_index must be sorted and unique")
    if not is_strictly_increasing(communicate_node_index):
        raise ValueError("communicate_node_index must be sorted and unique")
    if features.ndim != 2 or features.size(0) != owned_node_count:
        raise ValueError("features must contain one row per local node")
    if adjacency.ndim != 2 or adjacency.size(0) != 2:
        raise ValueError("adjacency must be an edge-index tensor with shape [2, E]")

    split_row_count = owned_node_count
    if artifact_version == 1:
        if not torch.equal(local_node_index, communicate_node_index):
            raise ValueError("0-hop local and communicate node indexes must match")
        if owned_node_count != metadata.get("node_count"):
            raise ValueError("local artifact node count does not match metadata")
        if adjacency.numel() and (
            int(adjacency.min()) < 0 or int(adjacency.max()) >= owned_node_count
        ):
            raise ValueError(
                "adjacency contains an endpoint outside local feature rows"
            )
        if adjacency.size(1) != metadata.get("internal_edge_count"):
            raise ValueError("local artifact edge count does not match metadata")
    else:
        if owned_node_count != metadata.get("owned_node_count") or (
            communicate_node_count != metadata.get("communicate_node_count")
        ):
            raise ValueError("2-hop artifact node counts do not match metadata")
        for start in range(0, owned_node_count, validation_chunk_size):
            owned_chunk = local_node_index[
                start : min(start + validation_chunk_size, owned_node_count)
            ]
            positions = torch.searchsorted(communicate_node_index, owned_chunk)
            if torch.any(positions >= communicate_node_count) or not torch.equal(
                communicate_node_index[positions], owned_chunk
            ):
                raise ValueError(
                    "2-hop communication index must contain every local node"
                )

        edge_count = int(adjacency.size(1))
        if edge_count != metadata.get("induced_edge_count"):
            raise ValueError("2-hop artifact edge count does not match metadata")

        # The baseline returns adj_global.pt for pretraining. Inspect the
        # pre-relabeled training view and chunk indexes without retaining them.
        for file_name in ("adj.pt", "source_degree.pt", "source_offsets.pt"):
            _record_artifact_tensor_path(args, file_name, shard_dir / file_name)
        training_adjacency = _load_artifact_tensor(
            shard_dir / "adj.pt", memory_map=True
        )
        source_degree = _load_artifact_tensor(
            shard_dir / "source_degree.pt", memory_map=True
        )
        source_offsets = _load_artifact_tensor(
            shard_dir / "source_offsets.pt", memory_map=True
        )
        if training_adjacency.shape != adjacency.shape:
            raise ValueError("2-hop global and local adjacency shapes must match")
        if source_degree.shape != (communicate_node_count,) or (
            source_offsets.shape != (communicate_node_count + 1,)
        ):
            raise ValueError("2-hop source degree or offset shape is invalid")
        if int(source_offsets[0]) != 0 or int(source_offsets[-1]) != edge_count:
            raise ValueError("2-hop source offsets do not cover the adjacency")
        for start in range(0, communicate_node_count, validation_chunk_size):
            stop = min(start + validation_chunk_size, communicate_node_count)
            degree_chunk = source_degree[start:stop]
            if torch.any(degree_chunk < 0) or not torch.equal(
                source_offsets[start + 1 : stop + 1] - source_offsets[start:stop],
                degree_chunk,
            ):
                raise ValueError("2-hop source degrees and offsets disagree")

        previous_source: Optional[int] = None
        for start in range(0, edge_count, validation_chunk_size):
            stop = min(start + validation_chunk_size, edge_count)
            global_chunk = adjacency[:, start:stop]
            local_chunk = training_adjacency[:, start:stop]
            if local_chunk.numel() and (
                int(local_chunk.min()) < 0
                or int(local_chunk.max()) >= communicate_node_count
            ):
                raise ValueError("2-hop local adjacency endpoint is out of range")
            if not torch.equal(communicate_node_index[local_chunk], global_chunk):
                raise ValueError("2-hop global and local adjacency views disagree")
            edge_positions = torch.arange(
                start, stop, dtype=source_offsets.dtype, device=source_offsets.device
            )
            expected_local_sources = (
                torch.searchsorted(source_offsets, edge_positions, right=True) - 1
            )
            if not torch.equal(local_chunk[0], expected_local_sources):
                raise ValueError("2-hop source offsets do not index adjacency rows")
            sources = global_chunk[0]
            if sources.numel() and (
                (previous_source is not None and int(sources[0]) < previous_source)
                or (
                    sources.numel() > 1
                    and not bool(torch.all(sources[1:] >= sources[:-1]))
                )
            ):
                raise ValueError("2-hop pretraining adjacency must be source-sorted")
            if sources.numel():
                previous_source = int(sources[-1])
        split_row_count = communicate_node_count

    for indexes, labels, split in (
        (idx_train, train_labels, "train"),
        (idx_val, val_labels, "val"),
        (idx_test, test_labels, "test"),
    ):
        if indexes.ndim != 1 or labels.ndim != 1 or indexes.numel() != labels.numel():
            raise ValueError(f"local artifact {split} indexes and labels are invalid")
        if indexes.numel() and (
            int(indexes.min()) < 0 or int(indexes.max()) >= split_row_count
        ):
            raise ValueError(f"local artifact {split} indexes are outside local rows")

    print(
        f"Loaded local artifact v{artifact_version} shard {trainer_id} "
        f"({hop_semantics}-hop) from {shard_dir}"
    )
    return (
        local_node_index,
        communicate_node_index,
        adjacency,
        train_labels,
        val_labels,
        test_labels,
        features,
        idx_train,
        idx_val,
        idx_test,
        global_node_num,
        class_num,
    )


class Trainer_General:
    """
    A general trainer class for training GCN in a federated learning setup, which includes functionalities
    required for training GCN models on a subset of a distributed dataset, handling local training and testing,
    parameter updates, and feature aggregation.

    Parameters
    ----------
    rank : int
        Unique identifier for the training instance (typically representing a trainer in federated learning).
    local_node_index : torch.Tensor
        Indices of nodes local to this trainer.
    communicate_node_index : torch.Tensor
        Indices of nodes that participate in communication during training.
    adj : torch.Tensor
        The adjacency matrix representing the graph structure.
    train_labels : torch.Tensor
        Labels of the training data.
    test_labels : torch.Tensor
        Labels of the testing data.
    features : torch.Tensor
        Node features for the entire graph.
    idx_train : torch.Tensor
        Indices of training nodes.
    idx_test : torch.Tensor
        Indices of test nodes.
    args_hidden : int
        Number of hidden units in the GCN model.
    global_node_num : int
        Total number of nodes in the global graph.
    class_num : int
        Number of classes for classification.
    device : torch.device
        The device (CPU or GPU) on which the model will be trained.
    args : Any
        Additional arguments required for model initialization and training.
    """

    def __init__(
        self,
        rank: int,
        # local_node_index: torch.Tensor,
        # communicate_node_index: torch.Tensor,
        # adj: torch.Tensor,
        # train_labels: torch.Tensor,
        # test_labels: torch.Tensor,
        # features: torch.Tensor,
        # idx_train: torch.Tensor,
        # idx_test: torch.Tensor,
        args_hidden: int,
        # global_node_num: int,
        # class_num: int,
        device: torch.device,
        args: Any,
        local_node_index: torch.Tensor = None,
        communicate_node_index: torch.Tensor = None,
        adj: torch.Tensor = None,
        train_labels: torch.Tensor = None,
        val_labels: torch.Tensor = None,
        test_labels: torch.Tensor = None,
        features: torch.Tensor = None,
        idx_train: torch.Tensor = None,
        idx_val: torch.Tensor = None,
        idx_test: torch.Tensor = None,
        global_node_num: Optional[int] = None,
        class_num: Optional[int] = None,
    ):
        # from gnn_models import GCN_Graph_Classification
        # Per-trainer seed = global_seed * 1000 + rank  (lets us vary across runs
        # while keeping different trainers distinct within a run).
        # Per-trainer seed = global_seed * 1000 + rank.  ``args`` may be a
        # Mock in unit tests where ``args.seed`` is not a real int, so guard
        # the conversion and fall back to the original ``manual_seed(rank)``.
        _seed_attr = getattr(args, "seed", 42)
        try:
            _global_seed = int(_seed_attr)
            torch.manual_seed(_global_seed * 1000 + rank)
        except (TypeError, ValueError):
            torch.manual_seed(rank)
        graph_storage_mode = _resolve_graph_storage_mode(args)
        adjacency_artifact_path: Optional[Path] = None
        pre_relabelled_adjacency_path: Optional[Path] = None
        loads_artifact_data = (
            local_node_index is None
            or communicate_node_index is None
            or adj is None
            or train_labels is None
            or test_labels is None
            or features is None
            or idx_train is None
            or idx_test is None
        )
        if loads_artifact_data:
            setattr(args, "_artifact_tensor_paths", {})
        if graph_storage_mode == "mmap" and not loads_artifact_data:
            raise ValueError(
                "graph_storage_mode='mmap' requires file-backed trainer artifacts; "
                "directly supplied tensors have no artifact files to memory-map"
            )
        self.artifact_load_time_sec: Optional[float] = None
        self.artifact_load_process_rss_bytes: Optional[int] = None
        self.artifact_load_process_peak_rss_bytes: Optional[int] = None
        if loads_artifact_data:
            artifact_load_start = time.perf_counter()
            (
                local_node_index,
                communicate_node_index,
                adj,
                train_labels,
                val_labels,
                test_labels,
                features,
                idx_train,
                idx_val,
                idx_test,
                global_node_num,
                class_num,
            ) = (
                load_trainer_data_from_local_artifact(rank, args)
                if _uses_local_nc_artifact(args)
                else (
                    load_trainer_data_from_huggingface_local_artifact(rank, args)
                    if _uses_huggingface_local_nc_artifact(args)
                    else load_trainer_data_from_hugging_face(rank, args)
                )
            )
            artifact_paths = _artifact_tensor_paths(args)
            recorded_adjacency_path = artifact_paths.get(
                "adj_global.pt", artifact_paths.get("adj.pt")
            )
            if recorded_adjacency_path:
                adjacency_artifact_path = Path(recorded_adjacency_path)
            if artifact_paths.get("adj_global.pt") and artifact_paths.get("adj.pt"):
                pre_relabelled_adjacency_path = Path(artifact_paths["adj.pt"])
            self.artifact_load_time_sec = time.perf_counter() - artifact_load_start
            artifact_load_snapshot = collect_resource_snapshot(
                source="trainer",
                event="artifact_load_complete",
                trainer_id=rank,
            )
            self.artifact_load_process_rss_bytes = artifact_load_snapshot[
                "process_rss_bytes"
            ]
            self.artifact_load_process_peak_rss_bytes = artifact_load_snapshot[
                "process_peak_rss_bytes"
            ]
            print(
                f"Trainer {rank} artifact load completed in "
                f"{self.artifact_load_time_sec:.3f}s; "
                "process RSS="
                f"{self.artifact_load_process_rss_bytes}; "
                "process peak RSS="
                f"{self.artifact_load_process_peak_rss_bytes} bytes"
            )
        if val_labels is None:
            val_labels = torch.empty(0, dtype=train_labels.dtype)
        if idx_val is None:
            idx_val = torch.empty(0, dtype=idx_train.dtype)
        self.rank = rank  # rank = trainer ID
        self.args = args
        self.adjacency_artifact_path = adjacency_artifact_path
        self.pre_relabelled_adjacency_path = pre_relabelled_adjacency_path
        artifact_paths = _artifact_tensor_paths(args)
        self.source_degree_artifact_path = artifact_paths.get("source_degree.pt")
        self.source_offsets_artifact_path = artifact_paths.get("source_offsets.pt")
        artifact_manifest = getattr(args, "_local_artifact_manifest", {})
        self.artifact_version = (
            artifact_manifest.get("artifact_version")
            if isinstance(artifact_manifest, dict)
            else None
        )
        self.adjacency_relabel_time_sec: Optional[float] = None
        self.adjacency_relabel_cache_path: Optional[str] = None
        self.adjacency_relabel_cache_hit: Optional[bool] = None
        self.adjacency_relabel_strategy: Optional[str] = None
        self.adjacency_relabel_source_edge_count: Optional[int] = None
        self.adjacency_relabel_output_edge_count: Optional[int] = None
        self.adjacency_relabel_dropped_edge_count: Optional[int] = None
        uses_legacy_huggingface_fedavg = _uses_legacy_huggingface_fedavg_adjacency(args)
        if uses_legacy_huggingface_fedavg:
            idx_train = _remap_legacy_huggingface_fedavg_indexes(
                local_node_index, communicate_node_index, idx_train, "idx_train"
            )
            idx_val = _remap_legacy_huggingface_fedavg_indexes(
                local_node_index, communicate_node_index, idx_val, "idx_val"
            )
            idx_test = _remap_legacy_huggingface_fedavg_indexes(
                local_node_index, communicate_node_index, idx_test, "idx_test"
            )

        self.device = device
        self.graph_storage_mode = graph_storage_mode
        self.graph_device = (
            torch.device("cpu")
            if self.graph_storage_mode in _HOST_GRAPH_STORAGE_MODES
            else self.device
        )
        self.features_memory_mapped = self.graph_storage_mode == "mmap"
        self.adjacency_memory_mapped = self.graph_storage_mode == "mmap"

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_losses: list = []
        self.train_accs: list = []

        self.test_losses: list = []
        self.test_accs: list = []
        self.val_losses: list = []
        self.val_accs: list = []

        self.local_node_index = local_node_index.to(self.graph_device)
        self.communicate_node_index = (
            self.local_node_index
            if uses_legacy_huggingface_fedavg
            else communicate_node_index.to(self.graph_device)
        )

        if uses_legacy_huggingface_fedavg:
            # k_hop_subgraph allocates several edge-sized masks and a global-ID
            # mapping. Relabel before applying the graph placement policy.
            self.adj = adj
            self.relabel_adj(local_node_index)
            self.adj = self.adj.to(self.graph_device)
        else:
            self.adj = adj.to(self.graph_device)
        self.train_labels = train_labels.to(self.graph_device)
        self.val_labels = val_labels.to(self.graph_device)
        self.test_labels = test_labels.to(self.graph_device)
        self.features = features.to(self.graph_device)
        self.idx_train = idx_train.to(self.graph_device)
        self.idx_val = idx_val.to(self.graph_device)
        self.idx_test = idx_test.to(self.graph_device)

        self.local_step = args.local_step
        self.args_hidden = args_hidden
        # self.global_node_num = global_node_num
        # self.class_num = class_num
        self.model = None
        self.optimizer = None
        self.last_sampled_batch_phase: Optional[str] = None
        self.last_sampled_batch_node_count: Optional[int] = None
        self.last_sampled_batch_edge_count: Optional[int] = None
        self.last_sampled_batch_bytes: Optional[int] = None
        self.global_node_num = (
            int(global_node_num.item())
            if isinstance(global_node_num, torch.Tensor)
            else global_node_num
        )
        self.class_num = (
            int(class_num.item()) if isinstance(class_num, torch.Tensor) else class_num
        )
        self.feature_aggregation = None
        if self.args.method == "FedAvg":
            self.feature_aggregation = self.features

    def get_info(self):
        label_nums = [
            int(labels.max().item()) + 1
            for labels in (self.train_labels, self.val_labels, self.test_labels)
            if labels.numel() > 0
        ]
        info = {
            "features_num": len(self.features),
            "label_num": max(label_nums, default=None),
            "global_node_num": self.global_node_num,
            "class_num": self.class_num,
            "feature_shape": self.features.shape[1],
            "len_in_com_train_node_local_indexes": len(self.idx_train),
            "len_in_com_val_node_local_indexes": len(self.idx_val),
            "len_in_com_test_node_local_indexes": len(self.idx_test),
            "artifact_version": self.artifact_version,
        }
        if self.args.method != "FedAvg":
            info["communicate_node_global_index"] = (
                self.communicate_node_index.detach().cpu()
            )
        return info

    def init_model(self, global_node_num, class_num):
        self.global_node_num = global_node_num
        self.class_num = class_num
        self.feature_shape = None

        self.scale_factor = 1e3
        self.param_history = []

        # seems that new trainer process will not inherit sys.path from parent, need to reimport!
        if self.args.num_hops >= 1:
            if self.args.dataset == "ogbn-arxiv":
                print("running AggreGCN_Arxiv")
                self.model = AggreGCN_Arxiv(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
            else:
                self.model = AggreGCN(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
        else:
            gnn_model = getattr(self.args, "gnn_model", "auto")
            if gnn_model == "graphsage" or self.args.dataset == "ogbn-products":
                print("Running SAGE_products")
                self.model = SAGE_products(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
            elif (
                "ogbn" in self.args.dataset
            ):  # ogbn large datasets default to GCN_arxiv
                print("Running GCN_arxiv")
                self.model = GCN_arxiv(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
            else:  # small datasets
                self.model = GCN(
                    nfeat=self.features.shape[1],
                    nhid=self.args_hidden,
                    nclass=class_num,
                    dropout=0.5,
                    NumLayers=self.args.num_layers,
                ).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=5e-4
        )

    @torch.no_grad()
    def update_params(self, params: tuple, current_global_epoch: int) -> None:
        """
        Updates the model parameters with global parameters received from the server.

        Parameters
        ----------
        params : tuple
            A tuple containing the global parameters from the server.
        current_global_epoch : int
            The current global epoch number.
        """
        # load global parameter from global server
        if self.model is None:
            return

        self.model.to("cpu")
        for (
            p,
            mp,
        ) in zip(params, self.model.parameters()):
            mp.data = p
        self.model.to(self.device)

    def verify_param_ranges(self, params, stage="pre-encryption"):
        """Verify parameter ranges and print statistics"""
        stats = []
        for i, p in enumerate(params):
            if isinstance(p, torch.Tensor):
                p = p.detach().cpu()
            stats.append(
                {
                    "layer": i,
                    "min": float(p.min()),
                    "max": float(p.max()),
                    "mean": float(p.mean()),
                    "std": float(p.std()),
                }
            )
            print(f"{stage} Layer {i} stats:")
            print(f"Range: [{stats[-1]['min']:.6f}, {stats[-1]['max']:.6f}]")
            print(f"Mean: {stats[-1]['mean']:.6f}")
            print(f"Std: {stats[-1]['std']:.6f}")
        return stats

    def get_local_feature_sum(self) -> torch.Tensor:
        """
        Computes the sum of features of all 1-hop neighbors for each node and normalizes the result.

        Returns
        -------
        normalized_sum : torch.Tensor
            The normalized sum of features of 1-hop neighbors for each node
        """
        if self.global_node_num is None:
            raise RuntimeError(
                "Trainer model metadata must be initialized before feature aggregation"
            )

        # Create a large matrix with known local node features
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features

        # Sum of features of all 1-hop nodes for each node
        one_hop_neighbor_feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer,
            self.adj,
            self.device,
            norm_type=getattr(self.args, "norm_type", "none"),
        )
        if hasattr(self.args, "use_encryption") and self.args.use_encryption:
            print(
                f"Trainer {self.rank} - Original feature sum (first 10 and last 10 elements): "
                f"{one_hop_neighbor_feature_sum.flatten()[:10].tolist()} ... {one_hop_neighbor_feature_sum.flatten()[-10:].tolist()}"
            )

        return one_hop_neighbor_feature_sum

    @torch.no_grad()
    def get_indexed_local_feature_sum(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only global feature-sum rows affected by local feature rows.

        This is mathematically equivalent to ``get_local_feature_sum`` on the
        returned rows, but never materializes a ``global_node_num x feature_dim``
        tensor. The server receives dense feature values only for active rows,
        paired with their global node IDs.
        """
        if self.global_node_num is None:
            raise RuntimeError(
                "Trainer model metadata must be initialized before feature aggregation"
            )
        if self.adj.ndim != 2 or self.adj.size(0) != 2:
            raise ValueError(
                "adj must be an edge-index tensor with shape [2, num_edges]"
            )

        local_node_ids = self.local_node_index.long()
        if local_node_ids.numel() != self.features.size(0):
            raise ValueError(
                "local_node_index must contain one global ID per feature row"
            )
        if local_node_ids.numel() > 1 and not bool(
            torch.all(local_node_ids[1:] >= local_node_ids[:-1])
        ):
            raise ValueError("local_node_index must be sorted for indexed aggregation")

        source_nodes = self.adj[0].long()
        target_nodes = self.adj[1].long()
        target_positions = torch.searchsorted(local_node_ids, target_nodes)
        valid_positions = target_positions < local_node_ids.numel()
        target_is_local = torch.zeros_like(valid_positions)
        target_is_local[valid_positions] = (
            local_node_ids[target_positions[valid_positions]]
            == target_nodes[valid_positions]
        )

        contributing_sources = source_nodes[target_is_local]
        contributing_features = self.features[target_positions[target_is_local]]
        row_ids = torch.unique(
            torch.cat([contributing_sources, local_node_ids]), sorted=True
        )
        row_values = torch.zeros(
            (row_ids.numel(), self.features.size(1)),
            dtype=self.features.dtype,
            device=self.device,
        )

        norm_type = getattr(self.args, "norm_type", "none")
        if norm_type not in {"none", "row", "sym"}:
            raise ValueError(
                f"Unknown norm_type: {norm_type}. Use 'sym', 'row', or 'none'."
            )

        if norm_type == "none":
            edge_weights = torch.ones(
                contributing_sources.numel(),
                dtype=self.features.dtype,
                device=self.device,
            )
            self_weights = torch.ones(
                local_node_ids.numel(),
                dtype=self.features.dtype,
                device=self.device,
            )
        else:
            degree_nodes, edge_counts = torch.unique(
                source_nodes, sorted=True, return_counts=True
            )

            def degrees_for(node_ids: torch.Tensor) -> torch.Tensor:
                positions = torch.searchsorted(degree_nodes, node_ids)
                valid = positions < degree_nodes.numel()
                matches = torch.zeros_like(valid)
                matches[valid] = degree_nodes[positions[valid]] == node_ids[valid]
                degrees = torch.ones(
                    node_ids.numel(), dtype=self.features.dtype, device=self.device
                )
                degrees[matches] = (
                    edge_counts[positions[matches]].to(self.features.dtype) + 1
                )
                return degrees

            source_degrees = degrees_for(contributing_sources)
            local_degrees = degrees_for(local_node_ids)
            if norm_type == "row":
                edge_weights = source_degrees.reciprocal()
                self_weights = local_degrees.reciprocal()
            else:
                target_degrees = degrees_for(target_nodes[target_is_local])
                edge_weights = source_degrees.rsqrt() * target_degrees.rsqrt()
                self_weights = local_degrees.reciprocal()

        if contributing_sources.numel() > 0:
            source_positions = torch.searchsorted(row_ids, contributing_sources)
            row_values.index_add_(
                0,
                source_positions,
                contributing_features * edge_weights.unsqueeze(1),
            )

        local_positions = torch.searchsorted(row_ids, local_node_ids)
        row_values.index_add_(
            0,
            local_positions,
            self.features * self_weights.unsqueeze(1),
        )
        return row_ids.detach().cpu(), row_values.detach().cpu()

    def get_local_feature_sum_og(self) -> torch.Tensor:
        """
        Computes the sum of features of all 1-hop neighbors for each node, used for plain text version.

        Returns
        -------
        one_hop_neighbor_feature_sum : torch.Tensor
            The sum of features of 1-hop neighbors for each node
        """

        computation_start = time.time()
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features
        one_hop_neighbor_feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer,
            self.adj,
            self.device,
            norm_type=getattr(self.args, "norm_type", "none"),
        )
        computation_time = time.time() - computation_start

        data_size = (
            one_hop_neighbor_feature_sum.element_size()
            * one_hop_neighbor_feature_sum.nelement()
        )

        print(f"Trainer {self.rank} - Computation time: {computation_time:.4f} seconds")
        print(f"Trainer {self.rank} - Data size: {data_size / 1024:.2f} KB")
        print(f"Trainer {self.rank} - Feature sum statistics:")
        print(f"Shape: {one_hop_neighbor_feature_sum.shape}")
        print(f"Mean: {one_hop_neighbor_feature_sum.mean().item():.6f}")
        print(f"Std: {one_hop_neighbor_feature_sum.std().item():.6f}")
        print(f"Min: {one_hop_neighbor_feature_sum.min().item():.6f}")
        print(f"Max: {one_hop_neighbor_feature_sum.max().item():.6f}")
        print(f"Non-zeros: {(one_hop_neighbor_feature_sum != 0).sum().item()}")

        return one_hop_neighbor_feature_sum, computation_time, data_size

    def load_feature_aggregation(self, feature_aggregation: torch.Tensor) -> None:
        """
        Loads the aggregated features into the trainer. Used for plain text version

        Parameters
        ----------
        feature_aggregation : torch.Tensor
            The aggregated features to be loaded.
        """
        # load_start = time.time()
        self.feature_aggregation = feature_aggregation.float().to(self.graph_device)
        # load_time = time.time() - load_start
        # data_size = (
        #     self.feature_aggregation.element_size()
        #     * self.feature_aggregation.nelement()
        # )
        # print(f"Trainer {self.rank} - Load time: {load_time:.4f} seconds")
        # print(f"Trainer {self.rank} - Data size: {data_size / 1024:.2f} KB")

        # return load_time

    def encrypt_feature_sum(self, feature_sum):
        feature_sum = self.get_local_feature_sum()
        # does not scale
        flattened_sum = feature_sum.flatten()
        enc_sum = ts.ckks_vector(self.he_context, flattened_sum.tolist()).serialize()

        return enc_sum, feature_sum.shape

    def decrypt_feature_sum(self, encrypted_sum, shape):
        decrypted_rows = [
            ts.ckks_vector_from(self.he_context, enc_row).decrypt()
            for enc_row in encrypted_sum
        ]

        decrypted_array = np.array(decrypted_rows)
        return torch.from_numpy(decrypted_array).float().reshape(shape)

    def get_encrypted_local_feature_sum(self, ct_output_path=None):
        # Check HE backend and route accordingly
        if hasattr(self, "he_backend") and self.he_backend == "openfhe":
            if getattr(self, "use_lowrank", False):
                return self._get_openfhe_lowrank_encrypted_feature_sum(ct_output_path)
            return self._get_openfhe_encrypted_local_feature_sum(ct_output_path)
        else:
            return self._get_tenseal_encrypted_local_feature_sum()

    def _get_openfhe_encrypted_local_feature_sum(self, ct_output_path=None):
        """OpenFHE encryption of local feature sum, chunked and serialized to files."""
        import json

        import openfhe

        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features
        feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer,
            self.adj,
            self.device,
            norm_type=getattr(self.args, "norm_type", "none"),
        )

        if not hasattr(self, "openfhe_cc"):
            raise RuntimeError("OpenFHE context not available on trainer")

        encryption_start = time.time()
        feature_list = feature_sum.flatten().tolist()

        # CKKS can only encrypt ring_dim/2 values per ciphertext
        slot_count = self.openfhe_cc.cc.GetRingDimension() // 2
        num_chunks = (len(feature_list) + slot_count - 1) // slot_count

        # Encrypt each chunk and serialize to numbered files
        if ct_output_path:
            base, ext = os.path.splitext(ct_output_path)
            for i in range(num_chunks):
                chunk = feature_list[i * slot_count : (i + 1) * slot_count]
                ct = self.openfhe_cc.encrypt(chunk)
                chunk_path = f"{base}_chunk{i}{ext}"
                openfhe.SerializeToFile(chunk_path, ct, openfhe.BINARY)

            # Write metadata
            meta_path = f"{base}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "num_chunks": num_chunks,
                        "slot_count": slot_count,
                        "total_elements": len(feature_list),
                    },
                    f,
                )

        encryption_time = time.time() - encryption_start
        return feature_sum.shape, encryption_time

    def _get_openfhe_lowrank_encrypted_feature_sum(self, ct_output_path=None):
        """Low-rank compress feature sum, then encrypt with OpenFHE threshold HE."""
        import json

        import openfhe

        from fedgraph.low_rank.compression_utils import svd_compress

        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features
        feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer,
            self.adj,
            self.device,
            norm_type=getattr(self.args, "norm_type", "none"),
        )

        if not hasattr(self, "openfhe_cc"):
            raise RuntimeError("OpenFHE context not available on trainer")

        encryption_start = time.time()

        # SVD compress: feature_sum (N x F) -> U (N x rank), S (rank,), V (F x rank)
        rank = getattr(self.args, "fixed_rank", 50)
        rank = min(rank, min(feature_sum.shape))
        U, S, V = svd_compress(feature_sum.cpu(), rank)

        # Flatten U, S, V into one list for encryption
        # Layout: [U_flat | S_flat | V_flat]
        u_flat = U.flatten().tolist()
        s_flat = S.flatten().tolist()
        v_flat = V.flatten().tolist()
        all_values = u_flat + s_flat + v_flat

        # Encrypt in chunks
        slot_count = self.openfhe_cc.cc.GetRingDimension() // 2
        num_chunks = (len(all_values) + slot_count - 1) // slot_count

        if ct_output_path:
            base, ext = os.path.splitext(ct_output_path)
            for i in range(num_chunks):
                chunk = all_values[i * slot_count : (i + 1) * slot_count]
                ct = self.openfhe_cc.encrypt(chunk)
                openfhe.SerializeToFile(f"{base}_chunk{i}{ext}", ct, openfhe.BINARY)

            # Write metadata including SVD shape info for reconstruction
            meta_path = f"{base}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "num_chunks": num_chunks,
                        "slot_count": slot_count,
                        "total_elements": len(all_values),
                        "lowrank": True,
                        "rank": rank,
                        "U_shape": list(U.shape),
                        "S_len": len(s_flat),
                        "V_shape": list(V.shape),
                        "original_shape": list(feature_sum.shape),
                    },
                    f,
                )

        encryption_time = time.time() - encryption_start
        return feature_sum.shape, encryption_time

    def _get_tenseal_encrypted_local_feature_sum(self):
        """TenSEAL encryption of local feature sum (existing implementation)"""
        # Same feature sum computation as original
        new_feature_for_trainer = torch.zeros(
            self.global_node_num, self.features.shape[1]
        ).to(self.device)
        new_feature_for_trainer[self.local_node_index] = self.features
        feature_sum = get_1hop_feature_sum(
            new_feature_for_trainer,
            self.adj,
            self.device,
            norm_type=getattr(self.args, "norm_type", "none"),
        )

        # Encrypt the feature sum
        encryption_start = time.time()
        flattened = feature_sum.flatten().tolist()
        encrypted = ts.ckks_vector(self.he_context, flattened).serialize()
        encryption_time = time.time() - encryption_start

        return encrypted, feature_sum.shape, encryption_time

    def setup_openfhe_nonlead(self, cc_path, lead_pk_path, output_pk_path):
        """Setup OpenFHE as non-lead party using file-based serialization."""
        import openfhe

        from fedgraph.openfhe_threshold import OpenFHEThresholdCKKS

        # Deserialize context from file
        cc, ok = openfhe.DeserializeCryptoContext(cc_path, openfhe.BINARY)
        if not ok:
            raise RuntimeError("Failed to deserialize CryptoContext")

        # Enable features on deserialized context
        cc.Enable(openfhe.PKE)
        cc.Enable(openfhe.KEYSWITCH)
        cc.Enable(openfhe.LEVELEDSHE)
        cc.Enable(openfhe.ADVANCEDSHE)
        cc.Enable(openfhe.MULTIPARTY)

        # Deserialize lead public key
        lead_pk, ok = openfhe.DeserializePublicKey(lead_pk_path, openfhe.BINARY)
        if not ok:
            raise RuntimeError("Failed to deserialize lead public key")

        # Initialize wrapper with deserialized context
        self.openfhe_cc = OpenFHEThresholdCKKS(cc=cc)

        # Generate non-lead share
        kp2 = self.openfhe_cc.generate_nonlead_share(lead_pk)

        # Serialize the joint public key (kp2.publicKey) to file
        openfhe.SerializeToFile(output_pk_path, kp2.publicKey, openfhe.BINARY)

        self.he_backend = "openfhe"
        self.use_lowrank = getattr(self.args, "use_lowrank", False)
        print(f"Trainer {self.rank}: Generated non-lead key share (designated)")
        return True

    def set_openfhe_public_key(self, cc_path, joint_pk_path):
        """Set the joint public key for encryption-only trainers using file-based serialization."""
        import openfhe

        from fedgraph.openfhe_threshold import OpenFHEThresholdCKKS

        # Deserialize context
        cc, ok = openfhe.DeserializeCryptoContext(cc_path, openfhe.BINARY)
        if not ok:
            raise RuntimeError("Failed to deserialize CryptoContext")

        cc.Enable(openfhe.PKE)
        cc.Enable(openfhe.KEYSWITCH)
        cc.Enable(openfhe.LEVELEDSHE)
        cc.Enable(openfhe.ADVANCEDSHE)
        cc.Enable(openfhe.MULTIPARTY)

        # Deserialize joint public key
        joint_pk, ok = openfhe.DeserializePublicKey(joint_pk_path, openfhe.BINARY)
        if not ok:
            raise RuntimeError("Failed to deserialize joint public key")

        self.openfhe_cc = OpenFHEThresholdCKKS(cc=cc)
        self.openfhe_cc.set_public_key(joint_pk)
        self.he_backend = "openfhe"
        self.use_lowrank = getattr(self.args, "use_lowrank", False)
        print(f"Trainer {self.rank}: Set joint public key (encryption only)")
        return True

    def openfhe_partial_decrypt_main_batch(self, he_dir, num_chunks):
        """Batch partial decryption of all chunks at once."""
        import openfhe

        if not hasattr(self, "openfhe_cc"):
            raise RuntimeError("OpenFHE context not initialized on trainer")

        for chunk_idx in range(num_chunks):
            agg_ct_path = os.path.join(he_dir, f"agg_ct_{chunk_idx}.bin")
            partial_path = os.path.join(he_dir, f"partial_main_{chunk_idx}.bin")

            agg_ct, ok = openfhe.DeserializeCiphertext(agg_ct_path, openfhe.BINARY)
            if not ok:
                raise RuntimeError(f"Failed to deserialize chunk {chunk_idx}")

            partial_list = self.openfhe_cc.cc.MultipartyDecryptMain(
                [agg_ct], self.openfhe_cc.secret_key_share
            )
            openfhe.SerializeToFile(partial_path, partial_list[0], openfhe.BINARY)

        print(
            f"Trainer {self.rank}: Batch partial decryption done ({num_chunks} chunks)"
        )
        return True

    def load_encrypted_feature_aggregation(self, encrypted_data):
        encrypted_sum, shape = encrypted_data

        # Check if this is OpenFHE decrypted data (already a tensor) or TenSEAL encrypted data
        if isinstance(encrypted_sum, torch.Tensor):
            # OpenFHE path: data is already decrypted tensor (on CPU from server).
            # Move to trainer's device for indexing and downstream training.
            decryption_start = time.time()
            encrypted_sum = encrypted_sum.to(self.device)
            self.feature_aggregation = encrypted_sum[self.communicate_node_index]
            decryption_time = time.time() - decryption_start
            return decryption_time
        else:
            # TenSEAL path: need to decrypt
            decryption_start = time.time()
            decrypted = ts.ckks_vector_from(self.he_context, encrypted_sum).decrypt()

            # reshape and store
            self.feature_aggregation = torch.tensor(decrypted).reshape(shape)[
                self.communicate_node_index
            ]

            return time.time() - decryption_start

    def get_encrypted_params(self):
        """Get encrypted parameters with proper scaling"""
        params_list = []
        metadata = []

        for param in self.model.parameters():
            param_data = param.cpu().detach()
            # scale
            max_abs_val = torch.max(torch.abs(param_data))
            scale = 1e3 if max_abs_val < 1e-3 else 1e2

            scaled_params = (param_data * scale).flatten().tolist()
            encrypted = ts.ckks_vector(self.he_context, scaled_params).serialize()

            params_list.append(encrypted)
            metadata.append({"shape": param_data.shape, "scale": scale})

        return params_list, metadata

    def load_encrypted_params(self, encrypted_data: tuple, current_global_epoch: int):
        """Load encrypted parameters with rescaling"""
        params_list, metadata = encrypted_data

        self.model.to("cpu")  # type: ignore[attr-defined]

        # load each layer's parameters
        for param, enc_param, meta in zip(
            self.model.parameters(), params_list, metadata  # type: ignore[attr-defined]
        ):
            decrypted = ts.ckks_vector_from(self.he_context, enc_param).decrypt()  # type: ignore[attr-defined]
            param_data = torch.tensor(decrypted).reshape(meta["shape"])
            param_data = param_data / meta["scale"]  # Reverse scaling
            param.data.copy_(param_data)

        self.model.to(self.device)  # type: ignore[attr-defined]
        return True

    def use_fedavg_feature(self) -> None:
        self.feature_aggregation

    def relabel_adj(self, node_index: Optional[torch.Tensor] = None) -> None:
        """
        Relabel the adjacency matrix against communication or explicitly supplied IDs.
        """
        started_at = time.perf_counter()
        relabel_node_index = (
            self.communicate_node_index if node_index is None else node_index
        )
        source_edge_count = int(self.adj.size(1))
        if node_index is None and self.pre_relabelled_adjacency_path is not None:
            local_adjacency = _load_artifact_tensor(
                self.pre_relabelled_adjacency_path,
                memory_map=(
                    self.graph_storage_mode == "mmap"
                    or self.graph_device.type == "cuda"
                ),
            )
            if local_adjacency.ndim != 2 or local_adjacency.size(0) != 2:
                raise ValueError(
                    "pre-relabeled artifact adjacency must have shape [2, E]"
                )
            output_edge_count = int(local_adjacency.size(1))
            if output_edge_count != source_edge_count:
                raise ValueError(
                    "pretraining and training artifact adjacency edge counts differ"
                )

            # Drop the global-ID GPU view before copying the local-coordinate
            # view so both edge tensors never occupy VRAM simultaneously.
            if self.adj.device.type == "cuda":
                self.adj = torch.empty((2, 0), dtype=torch.long, device="cpu")
                torch.cuda.empty_cache()
            self.adj = local_adjacency.to(self.graph_device)
            self.adjacency_artifact_path = self.pre_relabelled_adjacency_path
            self.adjacency_memory_mapped = self.graph_storage_mode == "mmap"
            self.adjacency_relabel_time_sec = time.perf_counter() - started_at
            self.adjacency_relabel_cache_path = str(self.pre_relabelled_adjacency_path)
            self.adjacency_relabel_strategy = "artifact-precomputed"
            self.adjacency_relabel_source_edge_count = source_edge_count
            self.adjacency_relabel_output_edge_count = output_edge_count
            self.adjacency_relabel_dropped_edge_count = 0
            print(
                "NC_ADJ_RELABEL, "
                f"trainer={self.rank}, mode=artifact-precomputed, "
                f"source_edges={source_edge_count}, output_edges={output_edge_count}, "
                f"dropped_edges=0, time_sec={self.adjacency_relabel_time_sec:.6f}, "
                f"source={self.pre_relabelled_adjacency_path}"
            )
            return

        if self.graph_storage_mode == "mmap":
            cache_dir = getattr(self.args, "graph_relabel_cache_dir", None)
            if not cache_dir:
                raise ValueError(
                    "graph_storage_mode='mmap' requires graph_relabel_cache_dir "
                    "when adjacency relabeling is needed"
                )
            if self.adjacency_artifact_path is None:
                raise ValueError(
                    "memory-mapped adjacency relabeling requires the source "
                    "artifact path"
                )
            result = relabel_adjacency_to_mmap_cache(
                self.adj,
                relabel_node_index,
                source_path=self.adjacency_artifact_path,
                cache_dir=Path(cache_dir),
                trainer_id=int(self.rank),
                chunk_edges=int(
                    getattr(self.args, "graph_relabel_chunk_edges", 1_000_000)
                ),
            )
            self.adj = result.edge_index
            self.adjacency_memory_mapped = True
            self.adjacency_relabel_time_sec = result.elapsed_sec
            self.adjacency_relabel_cache_path = str(result.cache_path)
            self.adjacency_relabel_cache_hit = result.cache_hit
            self.adjacency_relabel_strategy = "mmap-cache"
            self.adjacency_relabel_source_edge_count = result.source_edge_count
            self.adjacency_relabel_output_edge_count = result.edge_count
            self.adjacency_relabel_dropped_edge_count = result.dropped_edge_count
            print(
                "NC_ADJ_RELABEL, "
                f"trainer={self.rank}, mode=mmap, cache_hit={result.cache_hit}, "
                f"source_edges={result.source_edge_count}, "
                f"output_edges={result.edge_count}, "
                f"dropped_edges={result.dropped_edge_count}, "
                f"time_sec={result.elapsed_sec:.6f}, cache={result.cache_path}"
            )
            return

        max_node_id = -1
        if relabel_node_index.numel() > 0:
            max_node_id = max(max_node_id, int(relabel_node_index.max().item()))
        if self.adj.numel() > 0:
            max_node_id = max(max_node_id, int(self.adj.max().item()))

        # PyG otherwise infers num_nodes from adj.max(). A legacy Hf shard can
        # own an isolated high-ID node that does not occur in its edge list.
        # The explicit bound preserves that node while relabeling its local
        # induced adjacency.
        num_nodes = max_node_id + 1
        # print(f"Max value in adj: {self.adj.max()}")
        # print(
        #     f"Max value in communicate_node_index: {self.communicate_node_index.max()}"
        # )
        # distinct_values = torch.unique(self.adj.flatten())
        # print(f"Number of distinct values in adj: {distinct_values.numel()}")
        # print(f"distinct local: {len(self.local_node_index)}")
        # print(f"distinct communic: {len(self.communicate_node_index)}")
        # time.sleep(30)
        _, self.adj, __, ___ = torch_geometric.utils.k_hop_subgraph(
            relabel_node_index,
            0,
            self.adj,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        self.adjacency_relabel_time_sec = time.perf_counter() - started_at
        self.adjacency_relabel_cache_hit = False
        self.adjacency_relabel_strategy = "computed"
        self.adjacency_relabel_source_edge_count = source_edge_count
        self.adjacency_relabel_output_edge_count = int(self.adj.size(1))
        self.adjacency_relabel_dropped_edge_count = (
            source_edge_count - self.adjacency_relabel_output_edge_count
        )
        # print(f"Max value in adj: {self.adj.max()}")
        # print(
        #     f"Max value in communicate_node_index: {self.communicate_node_index.max()}"
        # )
        # distinct_values = torch.unique(self.adj.flatten())
        # print(f"Number of distinct values in adj: {distinct_values.numel()}")
        # print(f"distinct communic: {len(self.communicate_node_index)}")

    def _uses_mini_batch(self) -> bool:
        batch_size = getattr(self.args, "batch_size", 0)
        return isinstance(batch_size, int) and batch_size > 0

    def _require_supported_graph_execution(self, use_mini_batch: bool) -> None:
        if (
            self.graph_storage_mode in _HOST_GRAPH_STORAGE_MODES
            and self.device.type == "cuda"
            and not use_mini_batch
        ):
            raise ValueError(
                f"graph_storage_mode='{self.graph_storage_mode}' with CUDA requires "
                "a positive batch_size; "
                "full-batch execution would move the complete graph to VRAM"
            )

    def _make_neighbor_loader(
        self, data: Data, indexes: torch.Tensor, *, shuffle: bool
    ) -> NeighborLoader:
        return NeighborLoader(
            data,
            num_neighbors=[-1] * self.args.num_layers,
            batch_size=self.args.batch_size,
            input_nodes=indexes,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(
                self.graph_storage_mode in _HOST_GRAPH_STORAGE_MODES
                and self.device.type == "cuda"
            ),
        )

    def _sampled_batch_to_device(
        self,
        batch: Data,
        labels: torch.Tensor,
        *,
        phase: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seed_node_count = int(batch.batch_size)
        input_ids = batch.input_id.to(labels.device)
        seed_labels = labels[input_ids].to(self.device, non_blocking=True)
        batch_features = batch.x.to(self.device, non_blocking=True)
        batch_adjacency = batch.edge_index.to(self.device, non_blocking=True)
        seed_node_index = torch.arange(seed_node_count, device=self.device)

        self.last_sampled_batch_phase = phase
        self.last_sampled_batch_node_count = int(batch.x.size(0))
        self.last_sampled_batch_edge_count = int(batch.edge_index.size(1))
        self.last_sampled_batch_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (batch.x, batch.edge_index, seed_labels)
        )
        return batch_features, batch_adjacency, seed_labels, seed_node_index

    def train(self, current_global_round: int) -> None:
        """
        Performs local training for a specified number of iterations. This method
        updates the model using the loaded feature aggregation and the adjacency matrix.

        Parameters
        ----------
        current_global_round : int
            The current global training round.
        """
        torch.cuda.empty_cache()
        assert self.model is not None
        if self.feature_aggregation is None:
            raise ValueError(
                "feature_aggregation has not been set. Ensure pre-training "
                "communication is completed."
            )

        use_mini_batch = self._uses_mini_batch()
        self._require_supported_graph_execution(use_mini_batch)
        self.model.to(self.device)

        data = (
            Data(x=self.feature_aggregation, edge_index=self.adj)
            if use_mini_batch
            else None
        )
        full_batch_features = None
        full_batch_adjacency = None
        full_batch_labels = None
        full_batch_indexes = None
        if not use_mini_batch:
            full_batch_features = self.feature_aggregation.to(self.device)
            full_batch_adjacency = self.adj.to(self.device)
            full_batch_labels = self.train_labels.to(self.device)
            full_batch_indexes = self.idx_train.to(self.device)

        loss_train = 0.0
        acc_train = 0.0
        for iteration in range(self.local_step):
            self.model.train()
            if use_mini_batch:
                assert data is not None
                loader = self._make_neighbor_loader(data, self.idx_train, shuffle=True)
                batch = next(iter(loader), None)
                if batch is None:
                    loss_train, acc_train = 0.0, 0.0
                else:
                    (
                        batch_features,
                        batch_adjacency,
                        seed_labels,
                        seed_node_index,
                    ) = self._sampled_batch_to_device(
                        batch, self.train_labels, phase="train"
                    )
                    loss_train, acc_train = train(
                        iteration,
                        self.model,
                        self.optimizer,
                        batch_features,
                        batch_adjacency,
                        seed_labels,
                        seed_node_index,
                    )
            else:
                assert full_batch_features is not None
                assert full_batch_adjacency is not None
                assert full_batch_labels is not None
                assert full_batch_indexes is not None
                class_num = self.class_num
                assert (
                    full_batch_labels.min() >= 0
                ), f"train_labels contains negative values: {full_batch_labels.min()}"
                assert full_batch_labels.max() < class_num, (
                    "train_labels contains a value out of range: "
                    f"{full_batch_labels.max()} (number of classes: {class_num})"
                )
                loss_train, acc_train = train(
                    iteration,
                    self.model,
                    self.optimizer,
                    full_batch_features,
                    full_batch_adjacency,
                    full_batch_labels,
                    full_batch_indexes,
                )

            self.train_losses.append(loss_train)
            self.train_accs.append(acc_train)

    def _local_eval(
        self,
        labels: torch.Tensor,
        indexes: torch.Tensor,
        losses: list,
        accuracies: list,
    ) -> list:
        if self.model is None or self.feature_aggregation is None:
            return [0.0, 0.0]
        if (
            labels is None
            or indexes is None
            or labels.numel() == 0
            or indexes.numel() == 0
        ):
            return [0.0, 0.0]

        use_mini_batch = self._uses_mini_batch()
        self._require_supported_graph_execution(use_mini_batch)
        self.model = self.model.to(self.device)
        if use_mini_batch:
            data = Data(x=self.feature_aggregation, edge_index=self.adj)
            loader = self._make_neighbor_loader(data, indexes, shuffle=False)
            total_loss = torch.zeros((), device=self.device)
            total_correct = torch.zeros((), device=self.device, dtype=torch.long)
            total_examples = 0

            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    seed_node_count = int(batch.batch_size)
                    if seed_node_count == 0:
                        continue
                    (
                        batch_features,
                        batch_adjacency,
                        seed_labels,
                        _,
                    ) = self._sampled_batch_to_device(batch, labels, phase="evaluation")
                    seed_output = self.model(batch_features, batch_adjacency)[
                        :seed_node_count
                    ]
                    total_loss += F.nll_loss(seed_output, seed_labels, reduction="sum")
                    total_correct += (seed_output.argmax(dim=-1) == seed_labels).sum()
                    total_examples += seed_node_count

            if total_examples == 0:
                local_loss, local_acc = 0.0, 0.0
            else:
                local_loss = (total_loss / total_examples).item()
                local_acc = (total_correct.float() / total_examples).item()
        else:
            feats = self.feature_aggregation.to(self.device)
            adj = self.adj.to(self.device)
            device_labels = labels.to(self.device)
            device_indexes = indexes.to(self.device)
            local_loss, local_acc = test(
                self.model, feats, adj, device_labels, device_indexes
            )
        losses.append(local_loss)
        accuracies.append(local_acc)
        return [local_loss, local_acc]

    def local_val(self) -> list:
        """
        Evaluates the model on the local validation dataset.

        Returns
        -------
        (list) : list
            A list containing the validation loss and accuracy
            [local_val_loss, local_val_acc].
        """
        return self._local_eval(
            self.val_labels,
            self.idx_val,
            self.val_losses,
            self.val_accs,
        )

    def local_test(self) -> list:
        """
        Evaluates the model on the local test dataset.

        Returns
        -------
        (list) : list
            A list containing the test loss and accuracy [local_test_loss, local_test_acc].
        """
        return self._local_eval(
            self.test_labels,
            self.idx_test,
            self.test_losses,
            self.test_accs,
        )

    def get_params(self) -> tuple:
        """
        Retrieves a CPU snapshot of the current model parameters.

        Returns
        -------
        (tuple) : tuple
            Detached CPU tensors suitable for transfer to the server.
        """
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        if self.model is not None:
            # Ray deserializes return values on the receiving process before
            # Server.aggregate_weights can move them to CPU. Returning CUDA
            # Parameters therefore fails when the Ray head has no GPU.
            return tuple(
                parameter.detach().to(device="cpu", copy=True)
                for parameter in self.model.parameters()
            )
        return ()

    def get_all_loss_accuray(self) -> list:
        """
        Returns all recorded training and testing losses and accuracies.

        Returns
        -------
        (list) : list
            A list containing arrays of training losses, training accuracies, testing losses, and testing accuracies.
        """
        return [
            np.array(self.train_losses),
            np.array(self.train_accs),
            np.array(self.test_losses),
            np.array(self.test_accs),
        ]

    def get_rank(self) -> int:
        """
        Returns the rank (trainer ID) of the trainer.

        Returns
        -------
        (int) : int
            The rank (trainer ID) of this trainer instance.
        """
        return self.rank


class Trainer_GC:
    """
    A trainer class specified for graph classification tasks, which includes functionalities required
    for training GIN models on a subset of a distributed dataset, handling local training and testing,
    parameter updates, and feature aggregation.

    Parameters
    ----------
    model: object
        The model to be trained, which is based on the GIN model.
    trainer_id: int
        The ID of the trainer.
    trainer_name: str
        The name of the trainer.
    train_size: int
        The size of the training dataset.
    dataLoader: dict
        The dataloaders for training, validation, and testing.
    optimizer: object
        The optimizer for training.
    args: Any
        The arguments for the training.

    Attributes
    ----------
    model: object
        The model to be trained, which is based on the GIN model.
    id: int
        The ID of the trainer.
    name: str
        The name of the trainer.
    train_size: int
        The size of the training dataset.
    dataloader: dict
        The dataloaders for training, validation, and testing.
    optimizer: object
        The optimizer for training.
    args: object
        The arguments for the training.
    W: dict
        The weights of the model.
    dW: dict
        The gradients of the model.
    W_old: dict
        The cached weights of the model.
    gconv_names: list
        The names of the gconv layers.
    train_stats: Any
        The training statistics of the model.
    weights_norm: float
        The norm of the weights of the model.
    grads_norm: float
        The norm of the gradients of the model.
    conv_grads_norm: float
        The norm of the gradients of the gconv layers.
    conv_weights_Norm: float
        The norm of the weights of the gconv layers.
    conv_dWs_norm: float
        The norm of the gradients of the gconv layers.
    """

    def __init__(
        self,
        model: Any,
        trainer_id: int,
        trainer_name: str,
        train_size: int,
        dataloader: dict,
        optimizer: object,
        args: Any,
    ) -> None:
        self.model = model.to(args.device)
        self.id = trainer_id
        self.name = trainer_name
        self.train_size = train_size
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.args = args

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {
            key: torch.zeros_like(value) for key, value in self.model.named_parameters()
        }
        self.W_old = {
            key: value.data.clone() for key, value in self.model.named_parameters()
        }

        self.gconv_names: Any = None

        self.train_stats: dict[str, list[Any]] = {
            "trainingAccs": [],
            "valAccs": [],
            "trainingLosses": [],
            "valLosses": [],
            "testAccs": [],
            "testLosses": [],
        }
        self.weights_norm = 0.0
        self.grads_norm = 0.0
        self.conv_grads_norm = 0.0
        self.conv_weights_norm = 0.0
        self.conv_dWs_norm = 0.0

    ########### Public functions ###########
    def update_params(self, server_params: Any) -> None:
        """
        Update the model parameters by downloading the global model weights from the server.

        Parameters
        ----------
        server: Server_GC
            The server object that contains the global model weights.
        """
        self.gconv_names = server_params.keys()  # gconv layers
        for k in server_params:
            self.W[k].data = server_params[k].data.clone()

    def reset_params(self) -> None:
        """
        Reset the weights of the model to the cached weights.
        The implementation is copying the cached weights (W_old) to the model weights (W).

        """
        self.__copy_weights(target=self.W, source=self.W_old, keys=self.gconv_names)

    def cache_weights(self) -> None:
        """
        Cache the weights of the model.
        The implementation is copying the model weights (W) to the cached weights (W_old).
        """
        for name in self.W.keys():
            self.W_old[name].data = self.W[name].data.clone()

    def compute_update_norm(self, keys: dict) -> float:
        """
        Compute the max update norm (i.e., dW) for the trainer
        """
        dW = {}
        for k in keys:
            dW[k] = self.dW[k]

        curr_dW = torch.norm(
            torch.cat([value.flatten() for value in dW.values()])
        ).item()

        return curr_dW

    def compute_mean_norm(self, total_size: int, keys: dict) -> torch.Tensor:
        """
        Compute the mean update norm (i.e., dW) for the trainer
        Returns
        -------
        curr_dW: Tensor
        """
        dW = {}
        for k in keys:
            dW[k] = self.dW[k] * self.train_size / total_size

        curr_dW = torch.cat([value.flatten() for value in dW.values()])

        return curr_dW

    def set_stats_norms(self, train_stats: Any, is_gcfl: bool = False) -> None:
        """
        Set the norms of the weights and gradients of the model, as well as the statistics of the training.

        Parameters
        ----------
        train_stats: dict
            The training statistics of the model.
        is_gcfl: bool, optional
            Whether the training is for GCFL. The default is False.
        """
        self.train_stats = train_stats

        self.weights_norm = torch.norm(self.__flatten(self.W)).item()

        if self.gconv_names is not None:
            weights_conv = {key: self.W[key] for key in self.gconv_names}
            self.conv_weights_norm = torch.norm(self.__flatten(weights_conv)).item()

            grads_conv = {key: self.W[key].grad for key in self.gconv_names}
            self.conv_grads_norm = torch.norm(self.__flatten(grads_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        self.grads_norm = torch.norm(self.__flatten(grads)).item()

        if is_gcfl and self.gconv_names is not None:
            dWs_conv = {key: self.dW[key] for key in self.gconv_names}
            self.conv_dWs_norm = torch.norm(self.__flatten(dWs_conv)).item()

    def local_train(
        self, local_epoch: int, train_option: str = "basic", mu: float = 1
    ) -> None:
        """
        This function is a interface of the trainer class to train the model locally.
        It will call the train function specified for the training option, based on the args provided.

        Parameters
        ----------
        local_epoch: int
            The number of local epochs
        train_option: str, optional
            The training option. The possible values are 'basic', 'prox', and 'gcfl'. The default is 'basic'.
            'basic' - self-train and FedAvg
            'prox' - FedProx that includes the proximal term
            'gcfl' - GCFL, GCFL+ and GCFL+dWs
        mu: float, optional
            The proximal term. The default is 1.
        """
        assert train_option in ["basic", "prox", "gcfl"], "Invalid training option."

        if train_option == "gcfl":
            self.__copy_weights(target=self.W_old, source=self.W, keys=self.gconv_names)

        if train_option in ["basic", "prox"]:
            train_stats = self.__train(
                model=self.model,
                dataloaders=self.dataloader,
                optimizer=self.optimizer,
                local_epoch=local_epoch,
                device=self.args.device,
            )
        elif train_option == "gcfl":
            train_stats = self.__train(
                model=self.model,
                dataloaders=self.dataloader,
                optimizer=self.optimizer,
                local_epoch=local_epoch,
                device=self.args.device,
                prox=True,
                gconv_names=self.gconv_names,
                Ws=self.W,
                Wt=self.W_old,
                mu=mu,
            )

        if train_option == "gcfl":
            self.__subtract_weights(
                target=self.dW, minuend=self.W, subtrahend=self.W_old
            )
        self.set_stats_norms(train_stats)

    def local_test(self, test_option: str = "basic", mu: float = 1) -> tuple:
        """
        Final test of the model on the test dataset based on the test option.

        Parameters
        ----------
        test_option: str, optional
            The test option. The possible values are 'basic' and 'prox'. The default is 'basic'.
            'basic' - self-train, FedAvg, GCFL, GCFL+ and GCFL+dWs
            'prox' - FedProx that includes the proximal term
        mu: float, optional
            The proximal term. The default is 1.

        Returns
        -------
        (test_loss, test_acc, trainer_name, trainingAccs, valAccs): tuple(float, float, string, float, float)
            The average loss and accuracy, trainer's name, trainer.train_stats["trainingAccs"][-1], trainer.train_stats["valAccs"][-1]
        """
        assert test_option in ["basic", "prox"], "Invalid test option."
        if test_option == "basic":
            return self.__eval(
                model=self.model,
                test_loader=self.dataloader["test"],
                device=self.args.device,
            )
        elif test_option == "prox":
            return self.__eval(
                model=self.model,
                test_loader=self.dataloader["test"],
                device=self.args.device,
                prox=True,
                gconv_names=self.gconv_names,
                mu=mu,
                Wt=self.W_old,
            )
        else:
            raise ValueError("Invalid test option.")

    def get_train_size(self) -> int:
        return self.train_size

    def get_weights(self, ks: Any) -> dict[str, Any]:
        data: dict[str, Any] = {}
        W = {}
        dW = {}
        for k in ks:
            W[k], dW[k] = self.W[k], self.dW[k]
        data["W"] = W
        data["dW"] = dW
        data["train_size"] = self.train_size
        return data

    def get_total_weight(self) -> Any:
        return self.W

    def get_dW(self) -> Any:
        return self.dW

    def get_name(self) -> str:
        return self.name

    def get_id(self) -> Any:
        return self.id

    def get_conv_grads_norm(self) -> Any:
        return self.conv_grads_norm

    def get_conv_dWs_norm(self) -> Any:
        return self.conv_dWs_norm

    ########### Private functions ###########
    def __train(
        self,
        model: Any,
        dataloaders: dict,
        optimizer: Any,
        local_epoch: int,
        device: str,
        prox: bool = False,
        gconv_names: Any = None,
        Ws: Any = None,
        Wt: Any = None,
        mu: float = 0,
    ) -> dict:
        """
        Train the model on the local dataset.

        Parameters
        ----------
        model: object
            The model to be trained
        dataloaders: dict
            The dataloaders for training, validation, and testing
        optimizer: Any
            The optimizer for training
        local_epoch: int
            The number of local epochs
        device: str
            The device to run the training
        prox: bool, optional
            Whether to add the proximal term. The default is False.
        gconv_names: Any, optional
            The names of the gconv layers. The default is None.
        Ws: Any, optional
            The weights of the model. The default is None.
        Wt: Any, optional
            The target weights. The default is None.
        mu: float, optional
            The proximal term. The default is 0.

        Returns
        -------
        (results): dict
            The training statistics

        Note
        ----
        If prox is True, the function will add the proximal term to the loss function.
        Make sure to provide the required arguments `gconv_names`, `Ws`, `Wt`, and `mu` for the proximal term.
        """
        if prox:
            assert (
                (gconv_names is not None)
                and (Ws is not None)
                and (Wt is not None)
                and (mu != 0)
            ), "Please provide the required arguments for the proximal term."

        losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        if prox:
            convGradsNorm = []
        train_loader, val_loader, test_loader = (
            dataloaders["train"],
            dataloaders["val"],
            dataloaders["test"],
        )

        for _ in range(local_epoch):
            model.train()
            loss_train, acc_train, num_graphs = 0.0, 0.0, 0

            for _, batch in enumerate(train_loader):
                batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)
                loss += (
                    mu / 2.0 * self.__prox_term(model, gconv_names, Wt) if prox else 0.0
                )  # add the proximal term if required
                loss.backward()
                optimizer.step()
                loss_train += loss.item() * batch.num_graphs
                acc_train += pred.max(dim=1)[1].eq(label).sum().item()
                num_graphs += batch.num_graphs

            loss_train /= num_graphs  # get the average loss per graph
            acc_train /= num_graphs  # get the average average per graph

            loss_val, acc_val, _, _, _ = self.__eval(model, val_loader, device)
            loss_test, acc_test, _, _, _ = self.__eval(model, test_loader, device)

            losses_train.append(loss_train)
            accs_train.append(acc_train)
            losses_val.append(loss_val)
            accs_val.append(acc_val)
            losses_test.append(loss_test)
            accs_test.append(acc_test)

            if prox:
                convGradsNorm.append(self.__calc_grads_norm(gconv_names, Ws))

        # record the losses and accuracies for each epoch
        res_dict = {
            "trainingLosses": losses_train,
            "trainingAccs": accs_train,
            "valLosses": losses_val,
            "valAccs": accs_val,
            "testLosses": losses_test,
            "testAccs": accs_test,
        }
        if prox:
            res_dict["convGradsNorm"] = convGradsNorm

        return res_dict

    def __eval(
        self,
        model: GIN,
        test_loader: Any,
        device: str,
        prox: bool = False,
        gconv_names: Any = None,
        mu: float = 0,
        Wt: Any = None,
    ) -> tuple:
        """
        Validate and test the model on the local dataset.

        Parameters
        ----------
        model: GIN
            The model to be tested
        test_loader: Any
            The dataloader for testing
        device: str
            The device to run the testing
        prox: bool, optional
            Whether to add the proximal term. The default is False.
        gconv_names: Any, optional
            The names of the gconv layers. The default is None.
        mu: float, optional
            The proximal term. The default is None.
        Wt: Any, optional
            The target weights. The default is None.

        Returns
        -------
        (test_loss, test_acc, trainer_name, trainingAccs, valAccs): tuple(float, float, string, float, float)
            The average loss and accuracy, trainer's name, trainer.train_stats["trainingAccs"][-1], trainer.train_stats["valAccs"][-1]

        Note
        ----
        If prox is True, the function will add the proximal term to the loss function.
        Make sure to provide the required arguments `gconv_names`, `Ws`, `Wt`, and `mu` for the proximal term.
        """
        if prox:
            assert (
                (gconv_names is not None) and (mu is not None) and (Wt != 0)
            ), "Please provide the required arguments for the proximal term."

        model.eval()
        total_loss, total_acc, num_graphs = 0.0, 0.0, 0

        for batch in test_loader:
            batch.to(device)
            with torch.no_grad():
                pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)
                loss += (
                    mu / 2.0 * self.__prox_term(model, gconv_names, Wt) if prox else 0.0
                )

            total_loss += loss.item() * batch.num_graphs
            total_acc += pred.max(dim=1)[1].eq(label).sum().item()
            num_graphs += batch.num_graphs

        current_training_acc = -1
        current_val_acc = -1
        if self.train_stats["trainingAccs"]:
            current_training_acc = self.train_stats["trainingAccs"][-1]
        if self.train_stats["valAccs"]:
            current_val_acc = self.train_stats["valAccs"][-1]

        return (
            total_loss / num_graphs,
            total_acc / num_graphs,
            self.name,
            current_training_acc,  # if no data then return -1 for 1st train round
            current_val_acc,  # if no data then return -1 for 1st train round
        )

    def __prox_term(self, model: Any, gconv_names: Any, Wt: Any) -> torch.tensor:
        """
        Compute the proximal term.

        Parameters
        ----------
        model: Any
            The model to be trained
        gconv_names: Any
            The names of the gconv layers
        Wt: Any
            The target weights

        Returns
        -------
        prox: torch.tensor
            The proximal term
        """
        prox = torch.tensor(0.0, requires_grad=True)
        for name, param in model.named_parameters():
            # only add the prox term for sharing layers (gConv)
            if name in gconv_names:
                prox = prox + torch.norm(param - Wt[name]).pow(
                    2
                )  # force the weights to be close to the old weights
        return prox

    def __calc_grads_norm(self, gconv_names: Any, Ws: Any) -> float:
        """
        Calculate the norm of the gradients of the gconv layers.

        Parameters
        ----------
        model: Any
            The model to be trained
        gconv_names: Any
            The names of the gconv layers
        Wt: Any
            The target weights

        Returns
        -------
        convGradsNorm: float
            The norm of the gradients of the gconv layers
        """
        grads_conv = {k: Ws[k].grad for k in gconv_names}
        convGradsNorm = torch.norm(self.__flatten(grads_conv)).item()
        return convGradsNorm

    def __copy_weights(
        self, target: dict, source: dict, keys: Union[list, None]
    ) -> None:
        """
        Copy the source weights to the target weights.

        Parameters
        ----------
        target: dict
            The target weights
        source: dict
            The source weights
        keys: list, optional
            The keys to be copied. The default is None.
        """
        if keys is not None:
            for name in keys:
                target[name].data = source[name].data.clone()

    def __subtract_weights(self, target: dict, minuend: dict, subtrahend: dict) -> None:
        """
        Subtract the subtrahend from the minuend and store the result in the target.

        Parameters
        ----------
        target: dict
            The target weights
        minuend: dict
            The minuend
        subtrahend: dict
            The subtrahend
        """
        for name in target:
            target[name].data = (
                minuend[name].data.clone() - subtrahend[name].data.clone()
            )

    def __flatten(self, w: dict) -> torch.tensor:
        """
        Flatten the gradients of a trainer into a 1D tensor.

        Parameters
        ----------
        w: dict
            The gradients of a trainer
        """
        return torch.cat([v.flatten() for v in w.values()])

    def calculate_weighted_weight(self, key: Any) -> torch.tensor:
        weighted_weight = torch.mul(self.W[key].data, self.train_size)
        return weighted_weight


class Trainer_LP:
    """
    A trainer class specified for graph link prediction tasks, which includes functionalities required
    for training GNN models on a subset of a distributed dataset, handling local training and testing,
    parameter updates, and feature aggregation.

    Parameters
    ----------
    client_id : int
        The ID of the client.
    country_code : str
        The country code of the client. Each client is associated with one country code.
    user_id_mapping : dict
        The mapping of user IDs.
    item_id_mapping : dict
        The mapping of item IDs.
    number_of_users : int
        The number of users.
    number_of_items : int
        The number of items.
    meta_data : tuple
        The metadata of the dataset.
    hidden_channels : int, optional
        The number of hidden channels in the GNN model. The default is 64.
    """

    def __init__(
        self,
        client_id: int,
        country_code: str,
        user_id_mapping: dict,
        item_id_mapping: dict,
        number_of_users: int,
        number_of_items: int,
        meta_data: tuple,
        dataset_path: str,
        hidden_channels: int = 64,
    ):
        self.client_id = client_id
        self.country_code = country_code
        print(f"checking code and file path: {country_code},{dataset_path}")
        file_path = dataset_path
        country_codes: List[str] = [self.country_code]
        check_data_files_existance(country_codes, file_path)
        # global user_id and item_id
        self.data = get_data(
            self.country_code, user_id_mapping, item_id_mapping, file_path
        )
        self.model = GNN_LP(
            number_of_users, number_of_items, meta_data, hidden_channels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: '{self.device}'")
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_train_test_data_at_current_time_step(
        self,
        start_time_float_format: float,
        end_time_float_format: float,
        use_buffer: bool = False,
        buffer_size: int = 10,
    ) -> None:
        """
        Get the training and testing data at the current time step.

        Parameters
        ----------
        start_time_float_format : float
            The start time in float format.
        end_time_float_format : float
            The end time in float format.
        use_buffer : bool, optional
            Whether to use the buffer. The default is False.
        buffer_size : int, optional
            The size of the buffer. The default is 10.
        """
        (
            print("loading buffer_train_data_list")
            if use_buffer
            else print("loading train_data and test_data")
        )

        load_res = get_data_loaders_per_time_step(
            self.data,
            start_time_float_format,
            end_time_float_format,
            use_buffer,
            buffer_size,
        )

        if use_buffer:
            (
                self.global_train_data,
                self.test_data,
                self.buffer_train_data_list,
            ) = load_res
        else:
            self.train_data, self.test_data = load_res

    def train(
        self, client_id: int, local_updates: int, use_buffer: bool = False
    ) -> tuple:
        """
        Perform local training for a specified number of iterations.

        Parameters
        ----------
        local_updates : int
            The number of local updates.
        use_buffer : bool, optional
            Whether to use the buffer. The default is False.

        Returns
        -------
        (loss, train_finish_times) : tuple
            [0] The loss of the model
            [1] The time taken for each local update
        """
        train_finish_times = []
        if use_buffer:
            probabilities = [1 / len(self.buffer_train_data_list)] * len(
                self.buffer_train_data_list
            )

        for i in range(local_updates):
            if use_buffer:
                train_data = random.choices(
                    self.buffer_train_data_list, weights=probabilities, k=1
                )[0].to(self.device)
            else:
                train_data = self.train_data.to(self.device)

            start_train_time = time.time()

            self.optimizer.zero_grad()
            pred = self.model(train_data)
            ground_truth = train_data["user", "select", "item"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            self.optimizer.step()

            train_finish_time = time.time() - start_train_time
            train_finish_times.append(train_finish_time)
            print(
                f"client {self.client_id} local steps {i} loss {loss:.4f} train time {train_finish_time:.4f}"
            )

        return client_id, loss, train_finish_times

    def test(self, clientId: int, use_buffer: bool = False) -> tuple:
        """
        Test the model on the test data.

        Parameters
        ----------
        use_buffer : bool, optional
            Whether to use the buffer. The default is False.

        Returns
        -------
        (auc, hit_rate_at_2, traveled_user_hit_rate_at_2) : tuple
            [0] The AUC score
            [1] The hit rate at 2
            [2] The hit rate at 2 for traveled users
        """
        preds, ground_truths = [], []
        self.test_data.to(self.device)
        with torch.no_grad():
            if not use_buffer:
                self.train_data.to(self.device)
                preds.append(self.model.pred(self.train_data, self.test_data))
            else:
                self.global_train_data.to(self.device)
                preds.append(self.model.pred(self.global_train_data, self.test_data))
            ground_truths.append(self.test_data["user", "select", "item"].edge_label)

        pred = torch.cat(preds, dim=0)
        ground_truth = torch.cat(ground_truths, dim=0)
        auc = retrieval_auroc(pred, ground_truth)
        hit_rate_evaluator = RetrievalHitRate(top_k=2)
        hit_rate_at_2 = hit_rate_evaluator(
            pred,
            ground_truth,
            indexes=self.test_data["user", "select", "item"].edge_label_index[0],
        )
        traveled_user_hit_rate_at_2 = hit_rate_evaluator(
            pred[self.traveled_user_edge_indices],
            ground_truth[self.traveled_user_edge_indices],
            indexes=self.test_data["user", "select", "item"].edge_label_index[0][
                self.traveled_user_edge_indices
            ],
        )
        print(f"Test AUC: {auc:.4f}")
        print(f"Test Hit Rate at 2: {hit_rate_at_2:.4f}")
        print(f"Test Traveled User Hit Rate at 2: {traveled_user_hit_rate_at_2:.4f}")
        return clientId, auc, hit_rate_at_2, traveled_user_hit_rate_at_2

    def calculate_traveled_user_edge_indices(self, file_path: str) -> None:
        """
        Calculate the indices of the edges of the traveled users.

        Parameters
        ----------
        file_path : str
            The path to the file containing the traveled users.
        """
        with open(file_path, "r") as a:
            traveled_users = torch.tensor(
                [int(line.split("\t")[0]) for line in a]
            )  # read the user IDs of the traveled users
        mask = torch.isin(
            self.test_data["user", "select", "item"].edge_label_index[0], traveled_users
        )  # mark the indices of the edges of the traveled users as True or False
        self.traveled_user_edge_indices = torch.where(mask)[
            0
        ]  # get the indices of the edges of the traveled users

    def set_model_parameter(
        self, model_state_dict: dict, gnn_only: bool = False
    ) -> None:
        """
        Load the model parameters from the global server.

        Parameters
        ----------
        model_state_dict : dict
            The model parameters to be loaded.
        gnn_only : bool, optional
            Whether to load only the GNN parameters. The default is False.
        """
        if gnn_only:
            self.model.gnn.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

    def get_model_parameter(self, gnn_only: bool = False) -> dict:
        """
        Get the model parameters.

        Parameters
        ----------
        gnn_only : bool, optional
            Whether to get only the GNN parameters. The default is False.

        Returns
        -------
        dict
            The model parameters.
        """
        if gnn_only:
            return self.model.gnn.state_dict()
        else:
            return self.model.state_dict()
