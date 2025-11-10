"""Shared tool registry for ADK and MCP surfaces."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional

import yaml

from .baseline_tools import tabular_baseline
from .cv_tools import cv_split
from .dataset_tools import dataset_list, dataset_load_csv
from .report_tools import report_md
from .web_fetch import web_fetch


@dataclass(frozen=True)
class ToolSpec:
    """Describe a callable tool and its exported names."""

    name: str
    fn: Callable[..., Dict]
    mcp_name: Optional[str] = None
    description: Optional[str] = None


# Central registry of tools that can be exposed to either surface.
_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="web_fetch",
        fn=web_fetch,
        mcp_name="web_fetch",
        description="Fetch and normalize a web page.",
    ),
    ToolSpec(
        name="dataset_list",
        fn=dataset_list,
        description="List datasets mounted under /kaggle/input.",
    ),
    ToolSpec(
        name="dataset_load_csv",
        fn=dataset_load_csv,
        mcp_name="dataset.load_csv",
        description="Load a Kaggle CSV file and cache as feather.",
    ),
    ToolSpec(
        name="cv_split",
        fn=cv_split,
        mcp_name="cv.split",
        description="Generate cross-validation folds for cached data.",
    ),
    ToolSpec(
        name="tabular_baseline",
        fn=tabular_baseline,
        mcp_name="tabular.baseline",
        description="Fit a simple tabular baseline model.",
    ),
    ToolSpec(
        name="report_md",
        fn=report_md,
        mcp_name="report.md",
        description="Generate a markdown report for experiment notes.",
    ),
)


def _load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        # Default to all tools when no policy file exists.
        return {spec.name for spec in _TOOL_SPECS}
    data = yaml.safe_load(path.read_text()) or {}
    tools = data.get("tools", [])
    return {name for name in tools if isinstance(name, str)}


def load_allowlist(path: Path | None = None) -> set[str]:
    """Load the allowlist of tool names from disk."""

    policy_path = path or Path(__file__).resolve().parent.parent / "policies" / "tool_allowlist.yaml"
    return _load_allowlist(policy_path)


def iter_allowed_specs(allowlist: Iterable[str] | None = None) -> Iterator[ToolSpec]:
    allowed = set(allowlist) if allowlist is not None else {spec.name for spec in _TOOL_SPECS}
    for spec in _TOOL_SPECS:
        if spec.name in allowed:
            yield spec


def get_adk_tools(allowlist: Iterable[str] | None = None) -> Dict[str, Callable[..., Dict]]:
    """Return ADK tool name → callable mapping filtered by the allowlist."""

    return {spec.name: spec.fn for spec in iter_allowed_specs(allowlist)}


def get_mcp_tools(allowlist: Iterable[str] | None = None) -> Dict[str, Callable[..., Dict]]:
    """Return MCP tool name → callable mapping filtered by the allowlist."""

    tools = {}
    for spec in iter_allowed_specs(allowlist):
        if spec.mcp_name:
            tools[spec.mcp_name] = spec.fn
    return tools


def get_tool_specs() -> Mapping[str, ToolSpec]:
    """Expose the immutable registry for inspection/testing."""

    return {spec.name: spec for spec in _TOOL_SPECS}
