"""Advanced Dataset Profiling and Statistical Context Generation for LLMs.

Adapted from AI Data Analyst references (Groq + Pandas + NumPy + SciPy):
  - In-depth descriptive statistics (mean, std, min, 25%, 50%, 75%, max)
  - Automatic type coercion (dates, currency symbols ₹/$, numerics)
  - Skewness and distribution analysis
  - Missing value and duplicate detection
  - Time-series monthly aggregation, growth rate, and trend detection
  - Categorical value distributions and percentage breakdowns
  - Correlation insights across numeric dimensions
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nexusai.core.config import get_settings
from nexusai.core.logging import get_logger

logger = get_logger(__name__)

# Keywords indicating tabular data analysis intent
DATASET_QUERY_KEYWORDS = {
    "dataset", "dataframe", "excel", "csv", "xlsx", "table", "spreadsheet",
    "rows", "columns", "stats", "statistics", "describe", "distribution",
    "average", "mean", "median", "std", "maximum", "minimum", "highest", "lowest",
    "skew", "outlier", "missing", "duplicate", "trend", "growth", "revenue",
    "sales", "order", "customer", "product", "supplier", "coffee", "payments",
    "maintenance", "transaction", "frequency", "breakdown", "correlation"
}


class DatasetProfiler:
    """Profiles CSV and Excel datasets into dense, precise text summaries for LLM context."""

    def __init__(self, search_dirs: list[Path] | None = None) -> None:
        settings = get_settings()
        self._search_dirs = search_dirs or [
            Path(settings.UPLOAD_DIR),
            Path(".data/documents"),
            Path(".data/uploads"),
            Path("Test_Files"),
            Path("./Test_Files"),
            Path("../Test_Files"),
            Path("apps/backend/.data/documents"),
            Path("apps/backend/.data/uploads"),
        ]
        self._cache: dict[str, tuple[float, str]] = {}

    def load_dataframe(self, file_path: Path | str) -> pd.DataFrame | None:
        """Load CSV or Excel spreadsheet with automatic encoding fallbacks and type coercion."""
        p = Path(file_path)
        if not p.is_file():
            return None

        suffix = p.suffix.lower()
        df: pd.DataFrame | None = None

        if suffix in {".xlsx", ".xls"}:
            try:
                df = pd.read_excel(p, na_values=["NA", "N/A", "missing", "", "null", "NULL", "None"])
            except Exception as exc:
                logger.warning("Could not read Excel file '%s': %s", p.name, exc)
                return None
        else:
            # CSV / TSV fallback encodings
            for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                try:
                    df = pd.read_csv(
                        p,
                        encoding=enc,
                        na_values=["NA", "N/A", "missing", "", "null", "NULL", "None"],
                        on_bad_lines="skip",
                    )
                    break
                except Exception:
                    continue

        if df is None or df.empty:
            return None

        # Smart type coercion (from reference ai_data_analyst.py)
        for col in df.columns:
            col_str = str(col).lower()
            if any(k in col_str for k in ["date", "time", "created", "paid"]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
            elif df[col].dtype == object:
                # Strip currency symbols and commas before converting to numeric
                try:
                    cleaned = df[col].astype(str).str.replace(r"[₹$,\s]", "", regex=True)
                    numeric_s = pd.to_numeric(cleaned, errors="coerce")
                    if isinstance(numeric_s, pd.Series) and numeric_s.notna().mean() > 0.7:
                        df[col] = numeric_s
                except Exception:
                    pass

        return df

    def build_stats_context(self, df: pd.DataFrame, filename: str = "") -> str:
        """Build compact descriptive stats context for the LLM."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        lines = [
            f"Dataset Name: {filename or 'Active Dataset'}",
            f"Shape: {len(df):,} rows x {len(df.columns)} columns",
            f"Numeric Columns ({len(numeric_cols)}): {numeric_cols}",
            f"Categorical Columns ({len(cat_cols)}): {cat_cols[:10]}",
            f"Date Columns ({len(date_cols)}): {date_cols}",
        ]

        if numeric_cols:
            desc = df[numeric_cols].describe().round(2)
            lines.append("\nNumeric Summary Statistics (count, mean, std, min, 25%, 50%, 75%, max):")
            lines.append(desc.to_string())

        if cat_cols:
            lines.append("\nTop Categorical Frequencies:")
            for col in cat_cols[:6]:
                vc = df[col].value_counts().head(5)
                lines.append(f"- {col}: {vc.to_dict()}")

        return "\n".join(lines)

    def generate_text_insight(self, df: pd.DataFrame, filename: str = "") -> str:
        """Generate high-density analytical bullet points (from reference ai_data_analyst.py)."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        lines = []

        # 1. Numeric means, std, range
        if numeric_cols:
            desc = df[numeric_cols].describe()
            for col in numeric_cols[:5]:
                try:
                    mean = desc[col]["mean"]
                    mx = desc[col]["max"]
                    mn = desc[col]["min"]
                    std = desc[col]["std"]
                    lines.append(f"- {col}: mean {mean:,.2f} +/- {std:,.2f}, range {mn:,.2f} to {mx:,.2f}")
                except Exception:
                    pass

            # Skewness detection
            try:
                skews = df[numeric_cols].skew().abs().sort_values(ascending=False)
                if len(skews) > 0 and skews.iloc[0] > 1.0:
                    lines.append(
                        f"- Distribution note: {skews.index[0]} is skewed (skewness coefficient = {skews.iloc[0]:.2f})."
                    )
            except Exception:
                pass

            # Correlation insights (from reference analyser.py)
            if len(numeric_cols) >= 2:
                try:
                    corr = df[numeric_cols].corr()
                    pairs: list[tuple[str, str, float]] = []
                    cols_list = list(corr.columns)
                    for i in range(len(cols_list)):
                        for j in range(i + 1, len(cols_list)):
                            val = corr.iloc[i, j]
                            if not np.isnan(val):
                                pairs.append((cols_list[i], cols_list[j], float(val)))
                    if pairs:
                        pairs.sort(key=lambda p: abs(p[2]), reverse=True)
                        top_p = pairs[0]
                        lines.append(
                            f"- Correlation: strongest relationship between {top_p[0]} and {top_p[1]} (r = {top_p[2]:.2f})."
                        )
                except Exception:
                    pass

        # 2. Categorical breakdowns
        if cat_cols:
            for col in cat_cols[:3]:
                try:
                    vc = df[col].value_counts()
                    if len(vc) > 0:
                        top = vc.index[0]
                        pct = (vc.iloc[0] / max(1, len(df))) * 100.0
                        uniques = df[col].nunique()
                        lines.append(f"- Common {col}: '{top}' represents {pct:.1f}% ({uniques} distinct values).")
                except Exception:
                    pass

        # 3. Temporal span and trends (from reference analyst_processor.py)
        if date_cols:
            for col in date_cols[:2]:
                try:
                    valid_dates = df[col].dropna()
                    if len(valid_dates) > 0:
                        lines.append(
                            f"- Timeline ({col}): spans {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}."
                        )
                except Exception:
                    pass

        # 4. Financial / Monthly aggregation trends
        trend_summary = self.analyze_financial_and_trends(df)
        if trend_summary:
            lines.append(f"- Trend analysis: {trend_summary}")

        # 5. Missing values and duplicates
        miss = df.isna().sum()
        miss_cols = miss[miss > 0]
        if not miss_cols.empty:
            summary_miss = ", ".join(f"{col} ({cnt})" for col, cnt in miss_cols.head(4).items())
            lines.append(f"- Data quality: missing cells in {summary_miss}.")
        else:
            lines.append("- Data quality: zero missing values detected.")

        dups = int(df.duplicated().sum())
        if dups > 0:
            lines.append(f"- Duplication: {dups:,} duplicate rows identified.")

        return "\n".join(lines)

    def analyze_financial_and_trends(self, df: pd.DataFrame) -> str:
        """Compute periodic aggregation and growth trends (from reference analyst_processor.py)."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        if not numeric_cols or not date_cols:
            return ""

        date_col = date_cols[0]
        # Look for financial/quantity columns like amount, revenue, freight, quantity, price, total
        target_num: str | None = None
        for cand in ["amount", "revenue", "total_amount", "freight", "quantity", "price", "unitPrice"]:
            for col in numeric_cols:
                if cand.lower() in col.lower():
                    target_num = col
                    break
            if target_num:
                break
        if not target_num:
            target_num = numeric_cols[0]

        try:
            valid_df = df.dropna(subset=[date_col, target_num]).copy()
            if len(valid_df) < 2:
                return ""

            valid_df["_period"] = valid_df[date_col].dt.to_period("M")
            grouped = valid_df.groupby("_period")[target_num].sum().sort_index()

            if len(grouped) < 2:
                return f"Total {target_num} is {valid_df[target_num].sum():,.2f} across the observed timeframe."

            prev_val = float(grouped.iloc[-2])
            curr_val = float(grouped.iloc[-1])
            tot_val = float(valid_df[target_num].sum())
            avg_val = float(grouped.mean())

            if prev_val != 0:
                growth = ((curr_val - prev_val) / prev_val) * 100.0
                growth_text = f"grew by {growth:.1f}%" if growth >= 0 else f"dipped by {abs(growth):.1f}%"
            else:
                growth_text = "held steady"

            return (
                f"Total {target_num} is {tot_val:,.2f} (average monthly: {avg_val:,.2f}). "
                f"Period-over-period {target_num} {growth_text} entering latest recorded period."
            )
        except Exception as exc:
            logger.debug("Trend analysis computation skipped: %s", exc)
            return ""

    def generate_full_profile(self, df: pd.DataFrame, filename: str = "") -> str:
        """Compose statistical context, data summary, and insights."""
        stats_block = self.build_stats_context(df, filename=filename)
        insights_block = self.generate_text_insight(df, filename=filename)

        return (
            f"=== DATASET STATISTICAL PROFILE: {filename or 'Spreadsheet'} ===\n"
            f"{stats_block}\n\n"
            f"Key Analytical Insights:\n"
            f"{insights_block}\n"
            f"==========================================================="
        )

    def find_and_profile_relevant_datasets(self, query: str, max_files: int = 2) -> str:
        """Search available datasets on disk matching query keywords, profile them, and return context."""
        query_lower = query.lower()
        matched_files: list[Path] = []

        seen_names: set[str] = set()
        for base_dir in self._search_dirs:
            if not base_dir.is_dir():
                continue
            for f in sorted(base_dir.glob("*")):
                if f.suffix.lower() not in {".csv", ".tsv", ".xlsx", ".xls"}:
                    continue
                # Avoid duplicates
                clean_stem = re.sub(r"^[0-9a-f]{16,32}_", "", f.name).lower()
                if clean_stem in seen_names:
                    continue

                # Check relevance
                file_text = f.name.lower()
                is_direct_match = any(word in file_text for word in query_lower.split() if len(word) >= 3)
                is_general_dataset_query = any(k in query_lower for k in DATASET_QUERY_KEYWORDS)

                if is_direct_match:
                    matched_files.insert(0, f)
                    seen_names.add(clean_stem)
                elif is_general_dataset_query and len(matched_files) < max_files:
                    matched_files.append(f)
                    seen_names.add(clean_stem)

                if len(matched_files) >= max_files:
                    break
            if len(matched_files) >= max_files:
                break

        if not matched_files:
            return ""

        profiles: list[str] = []
        for f in matched_files[:max_files]:
            mtime = f.stat().st_mtime
            cache_key = f"{f.resolve()}_{mtime}"
            if cache_key in self._cache:
                profiles.append(self._cache[cache_key][1])
                continue

            try:
                df = self.load_dataframe(f)
                if df is not None and not df.empty:
                    display_name = re.sub(r"^[0-9a-f]{16,32}_", "", f.name)
                    prof = self.generate_full_profile(df, filename=display_name)
                    self._cache[cache_key] = (mtime, prof)
                    profiles.append(prof)
            except Exception as exc:
                logger.warning("Could not profile dataset '%s': %s", f.name, exc)

        return "\n\n".join(profiles)

    def find_relevant_dataframe(self, query: str) -> tuple[pd.DataFrame | None, str]:
        """Find the most relevant dataset matching the user query."""
        query_lower = query.lower()
        matched_file: Path | None = None
        named_fallback: Path | None = None
        uuid_fallback: Path | None = None

        for base_dir in self._search_dirs:
            if not base_dir.is_dir():
                continue
            for f in sorted(base_dir.glob("*")):
                if f.suffix.lower() not in {".csv", ".tsv", ".xlsx", ".xls"}:
                    continue
                clean_name = re.sub(r"^[0-9a-f]{16,32}_", "", f.name).lower()
                fname_lower = f.name.lower()

                # Prefer human-named semantic datasets over raw UUID temporary files
                is_uuid = bool(re.match(r"^[0-9a-f-]{32,}\.(csv|xlsx|xls)$", f.name.lower()))
                if not is_uuid and named_fallback is None:
                    named_fallback = f
                elif uuid_fallback is None:
                    uuid_fallback = f

                for kw in ["order", "customer", "product", "supplier", "coffee", "sales", "revenue", "freight", "stock"]:
                    if kw in query_lower and (kw in fname_lower or kw in clean_name):
                        matched_file = f
                        break
                if matched_file is not None:
                    break
            if matched_file is not None:
                break

        chosen = matched_file or named_fallback or uuid_fallback
        if chosen is None:
            return None, ""

        df = self.load_dataframe(chosen)
        clean_name = re.sub(r"^[0-9a-f]{16,32}_", "", chosen.name)
        return df, clean_name

    def generate_chart_spec(
        self,
        query: str,
        df: pd.DataFrame | None = None,
        filename: str = "",
    ) -> dict[str, Any] | None:
        """Generate a structured Plotly/Seaborn-compatible chart specification based on dataset analysis."""
        query_lower = query.lower()
        viz_keywords = {
            "visualise", "visualize", "plot", "chart", "graph", "histogram",
            "pie", "donut", "bar", "distribution", "trend", "breakdown",
            "analytics", "correlate", "show revenue", "show orders", "sales trend"
        }
        if not any(k in query_lower for k in viz_keywords):
            return None

        if df is None:
            df, filename = self.find_relevant_dataframe(query)

        if df is None or df.empty:
            return None

        # Detect valid named columns
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) and not str(c).lower().startswith("unnamed")]
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not str(c).lower().endswith("id") and not str(c).lower().startswith("unnamed")]
        cat_cols = [c for c in df.select_dtypes(include=[object, "category"]).columns if not str(c).lower().endswith("id") and not str(c).lower().startswith("unnamed")]

        # Determine best visualization strategy
        chart_type = "bar"
        title = f"Data Analytics & Distribution: {filename}"
        x_label = "Category"
        y_label = "Count / Volume"
        labels: list[str] = []
        series_data: list[float] = []
        series_name = "Count"
        kpis: list[dict[str, str]] = []

        is_trend_query = any(k in query_lower for k in ["trend", "over time", "month", "monthly", "time", "date", "growth"])
        is_donut_query = any(k in query_lower for k in ["pie", "donut", "share", "percentage", "proportion", "breakdown"])

        # Strategy 1: Time Series / Trend
        if date_cols and (is_trend_query or not cat_cols):
            dcol = date_cols[0]
            val_col = num_cols[0] if num_cols else None
            df_temp = df.dropna(subset=[dcol]).copy()
            df_temp["_period"] = df_temp[dcol].dt.to_period("M").astype(str)

            if val_col:
                agg = df_temp.groupby("_period")[val_col].sum().sort_index()
                series_name = f"Total {str(val_col).replace('_', ' ').title()}"
                y_label = series_name
            else:
                agg = df_temp.groupby("_period").size().sort_index()
                series_name = "Order Volume"
                y_label = "Volume"

            chart_type = "line"
            title = f"{series_name} Trend Over Time"
            x_label = "Month"
            labels = [str(idx) for idx in agg.index]
            series_data = [round(float(v), 2) for v in agg.values]

            total_val = sum(series_data)
            avg_val = float(np.mean(series_data)) if series_data else 0.0
            peak_val = max(series_data) if series_data else 0.0
            kpis = [
                {"label": "Total Cumulative", "value": f"{total_val:,.1f}"},
                {"label": "Monthly Average", "value": f"{avg_val:,.1f}"},
                {"label": "Peak Record", "value": f"{peak_val:,.1f}"},
                {"label": "Time Periods", "value": str(len(labels))},
            ]

        # Strategy 2: Categorical Distribution / Donut
        elif cat_cols and (is_donut_query or (len(df[cat_cols[0]].unique()) <= 6 and not is_trend_query)):
            ccol = cat_cols[0]
            for col in cat_cols:
                if any(k in str(col).lower() for k in ["country", "category", "region", "status", "type", "in_or_out"]):
                    ccol = col
                    break

            val_col = num_cols[0] if (num_cols and any(k in query_lower for k in ["revenue", "freight", "amount", "sales", "sum"])) else None
            if val_col:
                agg = df.groupby(ccol)[val_col].sum().sort_values(ascending=False).head(7)
                series_name = f"Sum of {val_col}"
            else:
                agg = df[ccol].value_counts().head(7)
                series_name = "Share"

            chart_type = "donut"
            title = f"Proportional Distribution by {str(ccol).title()}"
            labels = [str(k) for k in agg.index]
            series_data = [round(float(v), 2) for v in agg.values]
            total_sum = sum(series_data) or 1.0

            kpis = [
                {"label": "Total Evaluated", "value": f"{total_sum:,.0f}"},
                {"label": "Top Category", "value": f"{labels[0]} ({(series_data[0]/total_sum*100):.1f}%)" if labels else "N/A"},
                {"label": "Unique Segments", "value": str(len(labels))},
            ]

        # Strategy 3: Bar Comparison / Ranking (Default)
        else:
            ccol = cat_cols[0] if cat_cols else df.columns[0]
            for col in cat_cols:
                for kw in ["country", "company", "product", "city", "category", "ship", "name"]:
                    if kw in str(col).lower():
                        ccol = col
                        break

            val_col = num_cols[0] if num_cols else None
            for col in num_cols:
                for kw in ["price", "freight", "amount", "revenue", "stock", "quantity", "units"]:
                    if kw in str(col).lower():
                        val_col = col
                        break

            if val_col and ccol:
                agg = df.groupby(ccol)[val_col].mean().sort_values(ascending=False).head(8)
                series_name = f"Average {str(val_col).replace('_', ' ').title()}"
                y_label = series_name
            elif ccol:
                agg = df[ccol].value_counts().head(8)
                series_name = "Record Count"
                y_label = "Count"
            elif num_cols:
                val_col = num_cols[0]
                hist, bin_edges = np.histogram(df[val_col].dropna(), bins=7)
                labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(len(hist))]
                series_data = [float(h) for h in hist]
                agg = pd.Series(series_data, index=labels)
                series_name = f"{val_col} Distribution"
                y_label = "Frequency"
            else:
                agg = pd.Series([len(df)], index=["Total Rows"])

            chart_type = "bar"
            title = f"{series_name} by {str(ccol).replace('_', ' ').title()}"
            x_label = str(ccol).replace('_', ' ').title()
            labels = [str(k) for k in agg.index]
            series_data = [round(float(v), 2) for v in agg.values]

            kpis = [
                {"label": "Total Dataset Rows", "value": f"{len(df):,}"},
                {"label": "Top Dimension", "value": str(labels[0]) if labels else "N/A"},
                {"label": "Peak Metric", "value": f"{series_data[0]:,.1f}" if series_data else "0"},
                {"label": "Mean Value", "value": f"{float(np.mean(series_data)):,.1f}" if series_data else "0"},
            ]

        clean_series_data = [
            0.0 if (v is None or np.isnan(v) or np.isinf(v)) else round(float(v), 2)
            for v in series_data
        ]

        return {
            "type": chart_type,
            "title": title,
            "subtitle": f"Seaborn/Plotly Analytics Specification &bull; {filename or 'Active Dataset'} ({len(df)} records)",
            "x_label": x_label,
            "y_label": y_label,
            "labels": labels,
            "series": [
                {
                    "name": series_name,
                    "data": clean_series_data,
                    "color": "#F59E0B" if chart_type == "bar" else "#10B981" if chart_type == "donut" else "#3B82F6",
                }
            ],
            "kpis": kpis,
        }


_DEFAULT_PROFILER: DatasetProfiler | None = None


def get_dataset_profiler() -> DatasetProfiler:
    """Singleton getter for the DatasetProfiler."""
    global _DEFAULT_PROFILER
    if _DEFAULT_PROFILER is None:
        _DEFAULT_PROFILER = DatasetProfiler()
    return _DEFAULT_PROFILER

