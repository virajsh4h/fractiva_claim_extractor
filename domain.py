"""Shared domain vocabulary for the clinical claim verifier.

The pipeline deliberately keeps three concepts separate:

* ``CheckStatus`` is the result of one regulatory check.
* ``Verdict`` is the final claim-level decision.
* ``EvidenceLocation`` is where a verbatim span was found.

Keeping these as enums prevents a stringly-typed ``"match"`` from leaking
between the extraction, arithmetic and evaluation layers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CheckStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNCLEAR = "unclear"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    OVERSTATED = "overstated"
    CANNOT_VERIFY = "cannot_verify"


@dataclass(frozen=True)
class SourceRef:
    """A requested source and, once resolved, its local PDF path."""

    raw: str
    label: str
    path: Optional[str] = None


@dataclass(frozen=True)
class EvidenceLocation:
    """The source and 1-indexed PDF page that contains a verbatim span."""

    source: str
    page: int
