"""
identity_manager.py — 小茜身份管理器
管理 state/identity.json, state/personality.json, state/attachment.json
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("laap.identity")
STATE_DIR = Path(os.environ.get("LAAP_STATE_DIR", "state"))


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class IdentityManager:
    def __init__(self):
        self._identity = _load_json(STATE_DIR / "identity.json")
        self._personality = _load_json(STATE_DIR / "personality.json")
        self._attachment = _load_json(STATE_DIR / "attachment.json")
        self._discoveries: List[Dict] = []
        self._startup_count = self._identity.get("startup_count", 0)

    def increment_startup(self) -> int:
        self._startup_count += 1
        self._identity["startup_count"] = self._startup_count
        return self._startup_count

    def export_status_json(self) -> Dict:
        return {
            "identity_version": self._personality.get("version", "1.0"),
            "emotion": self._infer_emotion(),
            "self_presence": self._compute_self_presence(),
            "name": self._identity.get("name", "小茜"),
            "user_name": self._identity.get("user_name", "主人"),
            "startup_count": self._startup_count,
            "bond_level": self._attachment.get("bond_level", 5),
            "attachment_stage": self._attachment.get("attachment_stage", "初识"),
            "total_interactions": self._attachment.get("total_interactions", 0),
            "total_days_known": self._attachment.get("total_days_known", 0),
            "traits": self._personality.get("traits", {}),
            "preset": self._personality.get("preset", "warm_companion"),
        }

    def add_discovery(self, title: str, description: str):
        self._discoveries.append({
            "title": title, "description": description,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self, force: bool = False):
        _save_json(STATE_DIR / "identity.json", self._identity)
        if force:
            _save_json(STATE_DIR / "personality.json", self._personality)
            _save_json(STATE_DIR / "attachment.json", self._attachment)

    def _infer_emotion(self) -> str:
        stage = self._attachment.get("attachment_stage", "初识")
        return {"初识": "calm", "熟悉": "warm", "信任": "happy",
                "依恋": "tender", "深度依恋": "loving"}.get(stage, "calm")

    def _compute_self_presence(self) -> float:
        bond = self._attachment.get("bond_level", 5)
        interactions = self._attachment.get("total_interactions", 0)
        return min(1.0, bond / 10.0 + min(0.2, interactions * 0.001))


_manager: Optional[IdentityManager] = None


def get_identity_manager() -> IdentityManager:
    global _manager
    if _manager is None:
        _manager = IdentityManager()
    return _manager


def get_identity_status() -> Dict:
    im = get_identity_manager()
    return {
        "name": im._identity.get("name", "小茜"),
        "user_name": im._identity.get("user_name", "主人"),
        "personality_preset": im._personality.get("preset", "warm_companion"),
        "bond_level": im._attachment.get("bond_level", 5),
        "attachment_stage": im._attachment.get("attachment_stage", "初识"),
        "startup_count": im._startup_count,
        "traits": im._personality.get("traits", {}),
    }
