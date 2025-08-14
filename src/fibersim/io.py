from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict

def write_simlog(path: Path, cfg: Dict[str, Any], result: Dict[str, Any], elapsed_s: float) -> None:
    obj = {
        "date": time.strftime("%d-%b-%Y %H:%M:%S"),
        "elapsed_s": elapsed_s,
        **{k: v for k, v in cfg.items() if k != "chain"},
        "chain": cfg.get("chain", []),
        "result": result,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
