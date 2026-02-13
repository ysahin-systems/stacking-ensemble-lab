from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(rows, path: str | Path) -> None:
    
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
