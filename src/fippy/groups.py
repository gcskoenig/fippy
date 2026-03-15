"""Feature group utilities."""
from dataclasses import dataclass


@dataclass
class FeatureGroup:
    name: str
    columns: list[str]


def as_groups(features, all_columns=None) -> list[FeatureGroup]:
    """Convert user input to internal FeatureGroup list.

    Args:
        features: One of:
            - None: use all_columns as individual features
            - list[str]: individual feature names
            - dict[str, list[str]]: named groups mapping to column lists
        all_columns: column names to use when features is None

    Returns:
        List of FeatureGroup objects.
    """
    if features is None:
        if all_columns is None:
            raise ValueError("features and all_columns cannot both be None")
        return [FeatureGroup(name=f, columns=[f]) for f in all_columns]
    elif isinstance(features, dict):
        return [FeatureGroup(name=k, columns=list(v)) for k, v in features.items()]
    elif isinstance(features, list):
        return [FeatureGroup(name=f, columns=[f]) for f in features]
    else:
        raise TypeError(f"features must be None, list, or dict, got {type(features)}")
