# HDF5 to SQLite converter for MTG Card Matcher
# Usage: python h5_to_sqlite.py input.h5 output.sqlite

import json
import sqlite3
import sys

try:
    import h5py
except ImportError:
    print("This script requires h5py. Install with: pip install h5py")
    sys.exit(1)


def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS offsets (
            card_id   TEXT PRIMARY KEY,
            start_idx INTEGER NOT NULL,
            end_idx   INTEGER NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            card_id   TEXT PRIMARY KEY,
            meta_json TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_offsets_range ON offsets(start_idx, end_idx);")
    conn.commit()


def main(inp: str, outp: str):
    with h5py.File(inp, "r") as h5:
        # Read offsets and metadata JSON from HDF5
        # Expectation from C++: datasets named 'offsets' and 'metadata' contain JSON strings
        offsets_json = h5["offsets"][()].decode("utf-8") if isinstance(h5["offsets"][()], (bytes, bytearray)) else h5["offsets"][()]
        metadata_json = h5["metadata"][()].decode("utf-8") if isinstance(h5["metadata"][()], (bytes, bytearray)) else h5["metadata"][()]
        offsets = json.loads(offsets_json)
        metadata = json.loads(metadata_json)

    conn = sqlite3.connect(outp)
    try:
        ensure_schema(conn)
        cur = conn.cursor()

        # Insert offsets
        # offsets is a dict: card_id -> [start, end]
        cur.executemany(
            "INSERT OR REPLACE INTO offsets(card_id, start_idx, end_idx) VALUES (?, ?, ?)",
            [(cid, int(rng[0]), int(rng[1])) for cid, rng in offsets.items()],
        )

        # Insert metadata (store as JSON string)
        cur.executemany(
            "INSERT OR REPLACE INTO metadata(card_id, meta_json) VALUES (?, ?)",
            [(cid, json.dumps(meta)) for cid, meta in metadata.items()],
        )

        conn.commit()
        print(f"Wrote SQLite DB to {outp}")
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python h5_to_sqlite.py input.h5 output.sqlite")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
