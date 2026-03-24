from typing import Any, Sequence


RNA_SS_REQUIRED_ADAPTER_SUBSTRINGS = (
    "constraint_embedder.substructure_z_embedder",
    "constraint_embedder.substructure_log_alpha",
)
RNA_SS_REQUIRED_ADAPTER_SUBSTRINGS_ZERO_INIT = (
    "constraint_embedder.substructure_z_embedder",
)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def resolve_substructure_initialize_method(configs: Any) -> str:
    model_configs = _config_get(configs, "model", None)
    constraint_configs = _config_get(model_configs, "constraint_embedder", None)
    substructure_configs = _config_get(constraint_configs, "substructure_embedder", None)

    default_initialize_method = _config_get(constraint_configs, "initialize_method", "zero")
    branch_initialize_method = _config_get(
        substructure_configs,
        "initialize_method",
        "inherit",
    )
    if branch_initialize_method in (None, "", "inherit"):
        return str(default_initialize_method)
    return str(branch_initialize_method)


def parse_adapter_keywords(raw_keywords: str) -> list[str]:
    return [keyword.strip() for keyword in str(raw_keywords).split(",") if keyword.strip()]


def collect_required_adapter_param_substrings(configs: Any) -> list[str]:
    required = []
    rna_ss_configs = _config_get(configs, "rna_ss", {})
    if _config_get(rna_ss_configs, "enable", False):
        if resolve_substructure_initialize_method(configs) == "zero":
            required.extend(RNA_SS_REQUIRED_ADAPTER_SUBSTRINGS_ZERO_INIT)
        else:
            required.extend(RNA_SS_REQUIRED_ADAPTER_SUBSTRINGS)
    return required


def validate_required_adapter_matches(
    param_names: Sequence[str],
    adapter_keywords: Sequence[str],
    required_substrings: Sequence[str],
) -> None:
    if not required_substrings:
        return

    missing = []
    for required in required_substrings:
        matching_param_names = [name for name in param_names if required in name]
        if not matching_param_names:
            missing.append(f"{required} (parameter not found)")
            continue
        if not any(
            any(keyword in name for keyword in adapter_keywords)
            for name in matching_param_names
        ):
            missing.append(f"{required} (not matched by adapter_keywords)")

    if missing:
        raise RuntimeError(
            "Required adapter parameters are not covered by two_stage.adapter_keywords: "
            + ", ".join(missing)
        )
