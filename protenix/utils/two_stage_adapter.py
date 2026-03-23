from typing import Any, Sequence


RNA_SS_REQUIRED_ADAPTER_SUBSTRINGS = (
    "constraint_embedder.substructure_z_embedder",
    "constraint_embedder.substructure_log_alpha",
)


def parse_adapter_keywords(raw_keywords: str) -> list[str]:
    return [keyword.strip() for keyword in str(raw_keywords).split(",") if keyword.strip()]


def collect_required_adapter_param_substrings(configs: Any) -> list[str]:
    required = []
    rna_ss_configs = configs.get("rna_ss", {}) if hasattr(configs, "get") else {}
    if getattr(rna_ss_configs, "get", None) and rna_ss_configs.get("enable", False):
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
