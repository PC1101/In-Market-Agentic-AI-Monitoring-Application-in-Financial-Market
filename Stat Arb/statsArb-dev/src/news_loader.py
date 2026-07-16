"""
FNSPID (Financial News and Stock Price Integration Dataset) loader.

FNSPID is a free, static dataset on HuggingFace (Zihan1004/FNSPID) with
15.7 million financial headlines from Bloomberg, Reuters, Benzinga, and
NASDAQ across 4,775 S&P 500 companies, covering 1999-2023.

No API rate limits — download once (~2-3 GB), all queries run locally.

Alternative paid source: Benzinga API (pip install benzinga, covers 2012+).
"""
import os
import warnings

import pandas as pd

FNSPID_REPO = "Zihan1004/FNSPID"
FNSPID_RAW_DEFAULT = "data/news/fnspid_raw.parquet"

# Column name aliases — FNSPID may differ across releases
_DATE_COLS   = ["Date", "date", "DATE", "published_at", "publishedAt"]
_TICKER_COLS = ["Stock_symbol", "ticker", "Ticker", "TICKER", "symbol", "Symbol"]
_TEXT_COLS   = ["Article", "article", "headline", "Headline", "title", "Title", "text"]


class FNSPIDLoader:
    """
    Download, cache, and filter the FNSPID financial news dataset.

    Parameters
    ----------
    etf : str
        Sector ETF ticker (e.g. 'xlf'). Names the sector-filtered cache.
    cache_dir : str
        Directory for local cache files (default: data/news).
    raw_path : str
        Path for the full-dataset parquet (default: data/news/fnspid_raw.parquet).
    """

    def __init__(
        self,
        etf: str,
        cache_dir: str = "data/news",
        raw_path: str = FNSPID_RAW_DEFAULT,
    ):
        self.etf = etf.lower()
        self.cache_dir = cache_dir
        self.raw_path = raw_path
        self.filtered_path = os.path.join(cache_dir, f"fnspid_{self.etf}.parquet")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_raw(self, force: bool = False):
        """
        One-time download of the full FNSPID dataset from HuggingFace.
        Converts to Parquet for fast local queries.

        Tries the `datasets` library first; falls back to `huggingface_hub`
        direct file download.
        """
        if os.path.exists(self.raw_path) and not force:
            print(
                f"[FNSPIDLoader] Raw dataset already at {self.raw_path}. "
                "Pass force=True to re-download."
            )
            return

        os.makedirs(os.path.dirname(self.raw_path) or ".", exist_ok=True)
        print(f"[FNSPIDLoader] Downloading FNSPID from HuggingFace ({FNSPID_REPO})...")
        print("  One-time download, ~2-3 GB. May take several minutes.")

        try:
            self._download_via_datasets()
        except Exception as e:
            warnings.warn(
                f"[FNSPIDLoader] `datasets` library failed: {e}. "
                "Falling back to huggingface_hub direct download..."
            )
            self._download_via_hub()

    def _download_via_datasets(self):
        from datasets import load_dataset   # type: ignore

        print("[FNSPIDLoader] Loading via `datasets` library (memory-safe column selection)...")
        ds = load_dataset(FNSPID_REPO, split="train", trust_remote_code=True)

        # Identify the columns we actually need before pulling into pandas.
        # This avoids loading price/volume columns from a 23 GB dataset.
        all_cols = ds.column_names
        print(f"  Dataset columns: {all_cols}")

        keep = self._select_needed_columns(all_cols)
        print(f"  Keeping: {keep}")
        ds_slim = ds.select_columns(keep)

        # Convert in batches to avoid OOM on large datasets
        print("  Converting to pandas (batched)...")
        df = ds_slim.to_pandas()
        df = self._normalise_columns(df)
        df.to_parquet(self.raw_path, index=False)
        print(f"[FNSPIDLoader] Saved {len(df):,} rows → {self.raw_path}")

    def _download_via_hub(self):
        from huggingface_hub import hf_hub_download   # type: ignore

        # FNSPID actual filenames (note the typo "exteral" in the repo).
        # Prefer the larger nasdaq file; fall back to All_external if it fails.
        for filename in [
            "Stock_news/nasdaq_exteral_data.csv",   # 23 GB — primary (typo is intentional)
            "Stock_news/All_external.csv",           # 5.7 GB — fallback
            "nasdaq_exteral_data.csv",
            "nasdaq_external_data.csv",
        ]:
            try:
                csv_path = hf_hub_download(
                    repo_id=FNSPID_REPO,
                    filename=filename,
                    repo_type="dataset",
                )
                print(f"[FNSPIDLoader] Parsing {filename} in chunks...")
                self._csv_to_parquet_chunked(csv_path)
                return   # success — stop trying further files
            except Exception as exc:
                print(f"  {filename} failed: {exc}")
                # _csv_to_parquet_chunked cleans up its own partial file on error

        raise RuntimeError(
            f"Could not find FNSPID data file on HuggingFace ({FNSPID_REPO}). "
            "Check the repo for current file names and update _download_via_hub()."
        )

    def _csv_to_parquet_chunked(self, csv_path: str, chunksize: int = 500_000):
        """
        Convert a large FNSPID CSV to parquet in chunks to avoid OOM.
        Only the date, ticker, and article columns are retained.

        dtype=str forces ALL columns to string on read — prevents pandas from
        inferring 'Article' as float in sparse/NaN-heavy opening chunks, which
        causes a schema conflict when actual text arrives in later chunks.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Remove any partial file from a previous failed attempt so the
        # schema doesn't conflict with this run.
        if os.path.exists(self.raw_path):
            os.remove(self.raw_path)

        writer = None
        keep = None
        total = 0

        for chunk in pd.read_csv(
            csv_path,
            chunksize=chunksize,
            dtype=str,          # force all cols to string — prevents double/float inference
            keep_default_na=False,  # treat empty cells as "" not NaN
        ):
            if keep is None:
                keep = self._select_needed_columns(list(chunk.columns))
                print(f"  Keeping columns: {keep}")

            chunk = chunk[keep].copy()
            chunk = self._normalise_columns(chunk)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(self.raw_path, table.schema)
            writer.write_table(table)
            total += len(chunk)
            print(f"  Written {total:,} rows...", end="\r")

        if writer:
            writer.close()
        print(f"\n[FNSPIDLoader] Saved {total:,} rows → {self.raw_path}")

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter_sector(self, tickers: list, force: bool = False) -> pd.DataFrame:
        """
        Filter the raw FNSPID dataset to the tickers in a sector ETF.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols for the sector (from the price CSV).
        force : bool
            Re-filter even if the filtered cache already exists.

        Returns
        -------
        pd.DataFrame
            Columns: date (DatetimeIndex), ticker, article. Sorted by date.
        """
        if os.path.exists(self.filtered_path) and not force:
            print(f"[FNSPIDLoader] Loading filtered cache: {self.filtered_path}")
            df = pd.read_parquet(self.filtered_path)
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            return df

        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(
                f"Raw FNSPID not found at {self.raw_path}. "
                "Run:  python scripts/download_fnspid.py"
            )

        print(
            f"[FNSPIDLoader] Filtering FNSPID to "
            f"{len(tickers)} {self.etf.upper()} tickers..."
        )
        df = pd.read_parquet(self.raw_path)

        ticker_set = {t.upper() for t in tickers}
        mask = df["ticker"].str.upper().isin(ticker_set)
        filtered = df[mask].copy()

        matched = set(filtered["ticker"].str.upper().unique())
        missing = ticker_set - matched
        print(
            f"  {len(filtered):,} headlines matched. "
            f"{len(matched)}/{len(ticker_set)} tickers found in FNSPID."
        )
        if missing:
            warnings.warn(
                f"[FNSPIDLoader] {len(missing)} tickers not in FNSPID: "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}. "
                "These will contribute NaN stress → no suppression for those names."
            )

        os.makedirs(self.cache_dir, exist_ok=True)
        filtered.to_parquet(self.filtered_path, index=False)
        print(f"[FNSPIDLoader] Saved filtered cache → {self.filtered_path}")
        return filtered

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def load_titles_by_date(
        self, start_dt: str, end_dt: str, tickers: list = None
    ) -> dict:
        """
        Load sector headlines grouped by publication date.

        Returns
        -------
        dict
            {datetime.date -> list[str]} — all sector headlines for each date.
            Keys are publication dates (NOT trade dates — shifting is done
            in RegimeFilter.build() / compute_stress.py).
        """
        if os.path.exists(self.filtered_path):
            df = pd.read_parquet(self.filtered_path)
        elif tickers is not None:
            df = self.filter_sector(tickers)
        else:
            raise FileNotFoundError(
                f"Filtered FNSPID not found at {self.filtered_path}. "
                f"Run:  python scripts/cache_news.py --etf {self.etf}"
            )

        # Strip timezone from stored dates if present (handles parquets written
        # before the tz_localize(None) fix was applied).
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        mask = (
            (df["date"] >= pd.Timestamp(start_dt))
            & (df["date"] <= pd.Timestamp(end_dt))
        )
        sub = df[mask]
        if sub.empty:
            warnings.warn(
                f"[FNSPIDLoader] No headlines found for {self.etf} "
                f"{start_dt}..{end_dt}."
            )
            return {}

        groups = sub.groupby(sub["date"].dt.date)["article"].apply(list)
        return groups.to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_needed_columns(all_cols: list) -> list:
        """
        From all available columns, return only the date, ticker, and text columns.
        Drops price/volume columns to avoid loading unnecessary data.
        """
        cols_lower = {c.lower(): c for c in all_cols}
        keep = []
        for candidates in [_DATE_COLS, _TICKER_COLS, _TEXT_COLS]:
            for cand in candidates:
                if cand.lower() in cols_lower:
                    keep.append(cols_lower[cand.lower()])
                    break
        if len(keep) < 3:
            # Fallback: keep all columns and let _normalise_columns sort it out
            return all_cols
        return keep

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise FNSPID column names to: date, ticker, article.
        Handles column name variations across FNSPID releases.
        """
        cols_lower = {c.lower(): c for c in df.columns}
        col_map = {}

        for canonical, candidates in [
            ("date",    _DATE_COLS),
            ("ticker",  _TICKER_COLS),
            ("article", _TEXT_COLS),
        ]:
            for cand in candidates:
                if cand.lower() in cols_lower:
                    col_map[cols_lower[cand.lower()]] = canonical
                    break
            else:
                raise ValueError(
                    f"Cannot find '{canonical}' column in FNSPID. "
                    f"Available columns: {list(df.columns)}. "
                    f"Add the correct name to _{'_'.join(canonical.upper().split())}_COLS in news_loader.py."
                )

        df = df.rename(columns=col_map)[["date", "ticker", "article"]].copy()
        df["date"] = (
            pd.to_datetime(df["date"], utc=False, errors="coerce")
            .dt.tz_localize(None)   # strip any UTC info → tz-naive for consistent comparison
            .dt.normalize()
        )
        df = df.dropna(subset=["date", "article"]).reset_index(drop=True)
        df = df.sort_values("date")
        return df
