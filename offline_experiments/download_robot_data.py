#!/usr/bin/env python3
"""
download_robot_eval_data.py
---------------------------------
Dump the three core tables used by the distributed-robot-evaluation
service (`policies`, `sessions`, `episodes`) to local Parquet/CSV files.

Example
-------
python download_robot_eval_data.py \
    --db_url postgresql://centralserver:m3lxcf830x20g4@localhost:5432/real_eval \
    --out_dir ./eval_dump \
    --formats parquet csv

The script never touches large artefacts stored in GCS; it only pulls the
relational data, making the dump lightweight and quick.
"""
import os
import argparse
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


TABLES = ["policies", "sessions", "episodes"]
DEFAULT_FORMATS = ["parquet", "csv"]


def fetch_table(engine: Engine, table_name: str) -> pd.DataFrame:
    """Read an entire SQL table into a DataFrame (all columns)."""
    df = pd.read_sql_table(table_name, con=engine)
    return df


def save_dataframe(df: pd.DataFrame, out_dir: str, table_name: str, formats: list[str]) -> None:
    """Write DataFrame to disk in all requested formats."""
    if "parquet" in formats:
        parquet_path = os.path.join(out_dir, f"{table_name}.parquet")
        df.to_parquet(parquet_path, compression="snappy", index=False)
    if "csv" in formats:
        csv_path = os.path.join(out_dir, f"{table_name}.csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
    if "pickle" in formats:
        pkl_path = os.path.join(out_dir, f"{table_name}.pkl")
        df.to_pickle(pkl_path)


def main():
    parser = argparse.ArgumentParser(description="Download evaluation-server tables to local files.")
    parser.add_argument(
        "--db_url",
        required=True,
        help="SQLAlchemy-compatible database URL, e.g. "
             "'postgresql://user:pass@host:5432/real_eval'",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory in which to place the dumped files (created if absent).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FORMATS,
        choices=["parquet", "csv", "pickle"],
        help="One or more output formats to write (default: parquet csv).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-table row-count logging.",
    )
    args = parser.parse_args()

    # 1) Make sure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # 2) Connect (read-only) and dump each table
    engine = create_engine(args.db_url, echo=False)

    timestamp = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
    manifest_path = os.path.join(args.out_dir, f"dump_manifest_{timestamp}.txt")

    with open(manifest_path, "w") as manifest:
        manifest.write(f"Dump generated at {timestamp} UTC\n")
        manifest.write(f"Source DB: {args.db_url}\n")
        manifest.write(f"Formats  : {', '.join(args.formats)}\n\n")

        for tbl in TABLES:
            df = fetch_table(engine, tbl)
            save_dataframe(df, args.out_dir, tbl, args.formats)

            if not args.quiet:
                print(f"{tbl:<10} rows dumped: {len(df):>7}")

            manifest.write(f"{tbl:<10} {len(df):>10} rows\n")

    if not args.quiet:
        print(f"\n✓ Dump complete. Files are in: {args.out_dir}")
        print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

