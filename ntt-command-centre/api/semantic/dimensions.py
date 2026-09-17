"""
The filter model: which columns may slice the product, and on what grain.

A dimension earns a place here only if it is safe to slice by. Everything else
in the 44-column extract is drill-down evidence at the end of an insight —
never a slicer, never a KPI tile, never a headline axis.

**Grain is not decoration.** Stage, forecast category, order type and the dates
are constant within an opportunity, so counting them means counting distinct
opportunities (2,050). LOB and Portfolio vary between the lines of one
opportunity, so counting them means counting lines (3,034). Getting this
backwards produces a page where the stage bars sum to more deals than exist.
Each dimension therefore declares its own `count_basis` and `measures.count_on`
reads it, so no call site has to remember.

`validated` marks the four dimensions whose categorical mix was checked against
the client's own distribution workbook and matched to within 0.1 percentage
points (see the `Distribution Comparison` sheet). The others are legitimate
slicers but carry no such external validation, and the Context lens says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .loader import (
    FORECAST_ORDER,
    LOB_ORDER,
    ORDER_TYPES,
    PORTFOLIO_ORDER,
    STAGE_ORDER,
    facts,
)

CountBasis = Literal["lines", "opportunities"]


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    column: str
    count_basis: CountBasis
    #: Why this grain, in words the UI prints in the chart footer.
    basis_note: str
    validated: bool = False
    #: Fixed display order where the dimension is ordinal; None = by value.
    order: tuple[str, ...] | None = None
    description: str = ""

    def values(self, df: pd.DataFrame | None = None) -> list[str]:
        d = facts() if df is None else df
        if self.column not in d.columns:
            return []
        present = [v for v in d[self.column].dropna().unique()]
        if self.order:
            known = [v for v in self.order if v in present]
            extra = sorted(v for v in present if v not in self.order)
            return known + extra
        # Order by open value, so the most consequential value is first in a
        # filter list rather than alphabetically wherever it happens to land.
        g = d.groupby(self.column)["acv_revenue"].sum().sort_values(ascending=False)
        return [str(v) for v in g.index]

    @property
    def count_noun(self) -> str:
        return "lines" if self.count_basis == "lines" else "opportunities"


REGISTRY: dict[str, Dimension] = {
    "stage": Dimension(
        "stage", "Stage", "stage", "opportunities",
        "Stage is constant within an opportunity, so this counts distinct opportunities.",
        validated=True, order=STAGE_ORDER,
        description="Where the deal sits on the sales ladder, Identification through Deal Won/Lost.",
    ),
    "forecast": Dimension(
        "forecast", "Forecast Category", "forecast_category", "opportunities",
        "Forecast category is constant within an opportunity.",
        validated=True, order=FORECAST_ORDER,
        description=(
            "The rep's own commitment level. Warning: in this extract it encodes the "
            "outcome — every Closed opportunity won and every Omitted one lost — so it "
            "is safe as a filter and fatal as a model feature."
        ),
    ),
    "lob": Dimension(
        "lob", "Line of Business", "lob", "lines",
        "LOB varies between the lines of one opportunity, so this counts lines.",
        validated=True, order=LOB_ORDER,
        description="The four NTT lines of business: Networking, Security, Data Center, Digital Workplace.",
    ),
    "portfolio": Dimension(
        "portfolio", "Portfolio", "portfolio", "lines",
        "Portfolio varies between the lines of one opportunity, so this counts lines.",
        validated=True, order=PORTFOLIO_ORDER,
        description="The five offering types: Product, Technical Services, SDIS, VBR, Consulting Services.",
    ),
    "industry": Dimension(
        "industry", "Industry", "industry", "opportunities",
        "Industry is an account attribute and constant within an opportunity.",
        description="The account's vertical. Twelve values; Transportation & Logistics is the largest.",
    ),
    "orderType": Dimension(
        "orderType", "Order Type", "order_type", "opportunities",
        "Order type is constant within an opportunity.",
        order=ORDER_TYPES,
        description="New Business, Renewal or Expansion.",
    ),
    "quarter": Dimension(
        "quarter", "Fiscal Quarter", "fiscal_quarter", "opportunities",
        "Derived from the close date, which is constant within an opportunity.",
        description="NTT's fiscal quarter of the close date. FY26 runs April 2026 to March 2027.",
    ),
    "country": Dimension(
        "country", "Country", "country", "opportunities",
        "Country is an account attribute and constant within an opportunity.",
        description="United States or Canada.",
    ),
    "rep": Dimension(
        "rep", "Opportunity Owner", "owner", "opportunities",
        "The owner is constant within an opportunity.",
        description="The rep who owns the deal. 70 in this extract.",
    ),
    "account": Dimension(
        "account", "Account", "account_name", "opportunities",
        "The account is constant within an opportunity.",
        description="The customer. 357 named accounts across 363 account codes.",
    ),
}

DIM_ORDER: tuple[str, ...] = tuple(REGISTRY)

#: Dimensions that exist only on a derived frame (risk, anomalies) rather than
#: on the fact table. They filter downstream results, not `slice_frame`.
DERIVED_DIMS: dict[str, str] = {
    "riskBand": "risk_band",
    "anomalyCategory": "category",
}


def dim_column(key: str) -> str | None:
    d = REGISTRY.get(key)
    return d.column if d else None


def get(key: str) -> Dimension:
    if key not in REGISTRY:
        raise KeyError(
            f"'{key}' is not a filterable dimension. The filter model is "
            f"{list(REGISTRY)} — everything else in the extract is drill-down "
            f"evidence, not a slicer."
        )
    return REGISTRY[key]


def describe(df: pd.DataFrame | None = None) -> list[dict]:
    """The `dimensions` block of /api/meta."""
    return [
        {
            "key": d.key,
            "label": d.label,
            "column": d.column,
            "countBasis": d.count_basis,
            "basisNote": d.basis_note,
            "validated": d.validated,
            "description": d.description,
            "values": d.values(df),
        }
        for d in REGISTRY.values()
    ]
