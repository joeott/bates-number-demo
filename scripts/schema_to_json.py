#!/usr/bin/env python3
"""
schema_to_json.py
Dump an SQL schema as machine-readable JSON.

Usage examples
--------------
# Postgres (URI via env var so your shell history stays clean)
export DB_URI="postgresql+psycopg2://user:secret@localhost:5432/mydb"
python schema_to_json.py             # → prints JSON to stdout
python schema_to_json.py -o schema.json

# MySQL
python schema_to_json.py "mysql+pymysql://user:pass@127.0.0.1/db"

# SQLite
python schema_to_json.py sqlite:///local.db

Docs:  https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from sqlalchemy import create_engine, inspect


def reflect_schema(uri: str) -> dict:
    """Reflect database schema and return as dictionary."""
    eng = create_engine(uri)
    insp = inspect(eng)

    schema: dict[str, dict] = defaultdict(dict)

    for tbl in insp.get_table_names():
        tinfo = {"columns": [], "primary_key": [], "foreign_keys": [], "indexes": []}

        # Columns
        for col in insp.get_columns(tbl):
            tinfo["columns"].append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col["nullable"],
                "default": col.get("default"),
                "autoincrement": col.get("autoincrement"),
                "comment": col.get("comment"),
            })
            if col.get("primary_key"):
                tinfo["primary_key"].append(col["name"])

        # Foreign keys
        for fk in insp.get_foreign_keys(tbl):
            tinfo["foreign_keys"].append({
                "constrained_columns": fk["constrained_columns"],
                "referred_schema": fk.get("referred_schema"),
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"],
            })

        # Indexes (incl. unique constraints)
        for idx in insp.get_indexes(tbl):
            tinfo["indexes"].append({
                "name": idx["name"],
                "unique": idx["unique"],
                "column_names": idx["column_names"],
            })

        schema[tbl] = tinfo

    return schema


def main() -> None:
    """Main entry point."""
    p = argparse.ArgumentParser(description="Dump SQL schema to JSON")
    p.add_argument("uri", nargs="?", default=os.getenv("DB_URI"),
                   help="SQLAlchemy DB URI (or set DB_URI env var)")
    p.add_argument("-o", "--output", metavar="FILE",
                   help="write JSON to file instead of stdout")
    args = p.parse_args()

    if not args.uri:
        sys.exit("Error: supply DB URI via argument or DB_URI env var")

    try:
        schema_json = json.dumps(reflect_schema(args.uri), indent=2)
    except Exception as e:
        sys.exit(f"Error connecting to database: {e}")

    if args.output:
        with open(args.output, "w") as fp:
            fp.write(schema_json + "\n")
        print(f"✅  Wrote schema to {args.output}")
    else:
        print(schema_json)


if __name__ == "__main__":
    main()