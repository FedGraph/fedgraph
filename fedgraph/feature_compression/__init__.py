"""Detachable feature-dimension compression primitives."""

from .projection import (
    ProjectedFeatureArtifact,
    SharedFeatureBasisArtifact,
    feature_basis_sha256,
    load_shared_feature_basis,
    project_features_in_memory,
    project_features_to_file,
    save_shared_feature_basis,
)
from .shared_pca import (
    FeatureSecondMoment,
    SharedFeatureBasis,
    aggregate_feature_second_moments,
    compute_feature_second_moment,
    solve_shared_feature_basis,
)

__all__ = [
    "FeatureSecondMoment",
    "ProjectedFeatureArtifact",
    "SharedFeatureBasis",
    "SharedFeatureBasisArtifact",
    "aggregate_feature_second_moments",
    "compute_feature_second_moment",
    "feature_basis_sha256",
    "load_shared_feature_basis",
    "project_features_in_memory",
    "project_features_to_file",
    "save_shared_feature_basis",
    "solve_shared_feature_basis",
]
