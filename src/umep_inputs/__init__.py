"""
Public API for generating SOLWEIG input datasets.
"""

from umep_inputs.solweig.api import (
    load_buildings,
    load_dems,
    load_chm,
    load_landcover,
    generate_solweig_inputs,
)

__all__ = [
    "load_buildings",
    "load_dems",
    "load_chm",
    "load_landcover",
    "generate_solweig_inputs",
]
