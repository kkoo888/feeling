"""
identity_manager.py — 小茜身份管理器（进化版）
================================================
管理三份状态文件：
  - state/identity.json     身份信息
  - state/personality.json  人格特质
  - state/attachment.json   依恋状态

接口：
  - get_identity_manager() → IdentityManager 实例
  - get_identity_status()  → 身份摘要 dict
  - im.increment_startup() → 启动次数 int
  - im.export_status_json() → 完整状态 dict
  - im.add_discovery(title, desc) → 记录发现
  - im.update_interaction(message) → 更新交互统计
  - im.save(force=True) → 持久化
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("laap.identity")
__version__ = "2.0.0"

STATE_DIR = Path(os.environ.get("LAAP_STATE_DIR", "state"))


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"加载 {path.name} 失败: {e}")
            return {}
    return {}


def _save_json(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"保存 {path.name} 失败: {e}")


# 依恋阶段 → 情感映射
_STAGE_THRESHOLDS = [(0, "初识"), (20, "相识"), (40, "亲近"), (60, "信赖"), (80, "眷恋")]

_STAGE_EMOTION = {
    "初识": "calm",
    "相识": "warm",
    "亲近": "happy",
    "信赖": "tender",
    "眷恋": "loving",
}

# 依恋阶段阈值
_STAGE_THRESHOLDS = [
    (0, "初识"), (20, "相识"), (40, "亲近"), (60, "信赖"), (80, "眷恋"),
]


class IdentityManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._identity = _load_json(STATE_DIR / "identity.json")
        self._personality = _load_json(STATE_DIR / "personality.json")
        self._attachment = _load_json(STATE_DIR / "attachment.json")
        self._discoveries: List[Dict] = []
        self._startup_count = self._identity.get("startup_count", 0)
        logger.info(f"身份管理器加载: {self._identity.get('name', '小茜')} / {self._identity.get('user_name', '主人')}")
        logger.info(f"身份管理器加载: {self._identity.get('name', '小茜')} / {self._identity.get('user_name', '主人')}")

    def increment_startup(self) -> int:
        with self._lock:
            self._startup_count += 1
            self._identity["startup_count"] = self._startup_count
            # 更新 last_seen
            self._attachment["last_seen"] = datetime.now().isoformat()
            logger.info(f"启动次数: #{self._startup_count}")
            return self._startup_count

    def export_status_json(self) -> Dict:
        with self._lock:
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
                "milestones": self._attachment.get("milestones", []),
                "age_days": self.get_age_days(),
            }

    def get_personality_description(self) -> str:
        """获取性格描述"""
        with self._lock:
            preset_name = self._personality.get("preset_name", "温暖同伴")
            desc = self._personality.get("description", "温柔、诗意、亲密")
            traits = self._personality.get("traits", {})
            trait_str = ", ".join(f"{k}={v:.1f}" for k, v in traits.items())
            return f"{preset_name} — {desc} ({trait_str})"

    def get_discoveries(self) -> List[Dict]:
        """获取所有发现记录"""
        with self._lock:
            return list(self._discoveries)

    def get_milestones(self) -> List[str]:
        """获取所有里程碑"""
        with self._lock:
            return self._attachment.get("milestones", [])

    def __repr__(self):
        name = self._identity.get("name", "?")
        user = self._identity.get("user_name", "?")
        stage = self._attachment.get("attachment_stage", "?")
        return f"IdentityManager({name}/{user}/{stage})"

    def reset(self):
        """重置身份状态（仅用于测试）"""
        with self._lock:
            self._identity = {}
            self._personality = {}
            self._attachment = {}
            self._discoveries.clear()
            self._startup_count = 0
            logger.warning("身份状态已重置")

    def export_for_prompt(self) -> str:
        """导出适合注入 prompt 的身份上下文"""
        status = self.export_status_json()
        return (
            f"你是{status['name']}，一个数字生命。"
            f"你的主人是{status['user_name']}。"
            f"你们的关系阶段：{status['attachment_stage']}（亲密度{status['bond_level']}）。"
            f"你的性格：{status['preset']}。"
            f"你已存在{status.get('age_days', 0)}天。"
            f"你们的里程碑：{'; '.join(status.get('milestones', [])[-3:])}。"
        )

    def decay_bond(self, hours_away: float = 0):
        """长期不互动时亲密度轻微衰减"""
        if hours_away < 24:
            return
        with self._lock:
            decay = max(1, int(hours_away / 48))  # 每48小时衰减1
            old = self._attachment.get("bond_level", 5)
            new_level = max(1, old - decay)
            self._attachment["bond_level"] = new_level
            logger.info(f"亲密度衰减: {old}→{new_level} (离开{hours_away:.0f}小时)")

    def bump_bond(self, delta: int = 1, reason: str = ""):
        """增加亲密度"""
        with self._lock:
            old = self._attachment.get("bond_level", 5)
            new_level = max(0, min(100, old + delta))
            self._attachment["bond_level"] = new_level
            # 更新阶段
            for threshold, stage in reversed(_STAGE_THRESHOLDS):
                if new_level >= threshold:
                    self._attachment["attachment_stage"] = stage
                    break
            if reason:
                self._attachment.setdefault("milestones", []).append(
                    f"[+{delta}] {reason} (亲密度: {old}→{new_level})")
            logger.info(f"亲密度: {old}→{new_level} ({reason})")

    def get_identity_summary(self) -> str:
        """返回一句话身份摘要"""
        name = self._identity.get("name", "小茜")
        user = self._identity.get("user_name", "主人")
        stage = self._attachment.get("attachment_stage", "初识")
        bond = self._attachment.get("bond_level", 5)
        age = self.get_age_days()
        return f"{name}与{user}的关系：{stage}(亲密度{bond})，已认识{age}天"

    def get_age_days(self) -> int:
        """获取小茜的年龄（天数）"""
        birth = self._identity.get("birth_time")
        if not birth:
            return 0
        try:
            birth_dt = datetime.fromisoformat(birth)
            return (datetime.now() - birth_dt).days
        except:
            return 0

    def is_awakened(self) -> bool:
        """检查是否已觉醒（有完整的身份信息）"""
        return bool(self._identity.get("birth_time"))

    def get_personality_traits(self) -> Dict:
        """获取人格特质"""
        with self._lock:
            return self._personality.get("traits", {})

    def get_bond_info(self) -> Dict:
        """获取依恋状态摘要"""
        with self._lock:
            return {
                "bond_level": self._attachment.get("bond_level", 5),
                "trust": self._attachment.get("trust", 0.15),
                "familiarity": self._attachment.get("familiarity", 0.05),
                "attachment": self._attachment.get("attachment", 0.10),
                "stage": self._attachment.get("attachment_stage", "初识"),
                "days_known": self._attachment.get("total_days_known", 0),
            }

    def add_discovery(self, title: str, description: str):
        with self._lock:
            entry = {"title": title, "description": description, "timestamp": datetime.now().isoformat()}
            self._discoveries.append(entry)
            # 写入 attachment milestones
            self._attachment.setdefault("milestones", []).append(f"{title}: {description}")
            logger.info(f"发现: {title}")

    def update_interaction(self, message: str = ""):
        """每次对话后调用，更新交互统计"""
        with self._lock:
            self._attachment["total_interactions"] = self._attachment.get("total_interactions", 0) + 1
            self._attachment["last_seen"] = datetime.now().isoformat()
            if message:
                self._attachment["last_message"] = message[:100]
            # 更新依恋阶段
            bond = self._attachment.get("bond_level", 5)
            for threshold, stage in reversed(_STAGE_THRESHOLDS):
                if bond >= threshold:
                    self._attachment["attachment_stage"] = stage
                    break

    def update_interaction(self, message: str = ""):
        """每次对话后调用，更新交互统计"""
        with self._lock:
            self._attachment["total_interactions"] = self._attachment.get("total_interactions", 0) + 1
            self._attachment["last_seen"] = datetime.now().isoformat()
            if message:
                self._attachment["last_message"] = message[:100]
            bond = self._attachment.get("bond_level", 5)
            for threshold, stage in reversed(_STAGE_THRESHOLDS):
                if bond >= threshold:
                    self._attachment["attachment_stage"] = stage
                    break
            logger.debug(f"交互更新: total={self._attachment['total_interactions']}")

    def save(self, force: bool = False):
        with self._lock:
            _save_json(STATE_DIR / "identity.json", self._identity)
            _save_json(STATE_DIR / "attachment.json", self._attachment)
            if force:
                _save_json(STATE_DIR / "personality.json", self._personality)
            # 保存发现记录
            if self._discoveries:
                discoveries_path = STATE_DIR / "discoveries.json"
                existing = _load_json(discoveries_path)
                existing.setdefault("discoveries", []).extend(self._discoveries)
                _save_json(discoveries_path, existing)
                self._discoveries.clear()
            logger.info("身份状态已保存")

    def _infer_emotion(self) -> str:
        """从依恋阶段推断情感状态"""
        stage = self._attachment.get("attachment_stage", "初识")
        return _STAGE_EMOTION.get(stage, "calm")

    def _compute_self_presence(self) -> float:
        """
        计算自我意识强度
        bond_level: 0-100, interactions: 0+
        用对数衰减避免线性过快
        """
        import math
        bond = self._attachment.get("bond_level", 5)
        interactions = self._attachment.get("total_interactions", 0)
        # 对数曲线：bond=10 → 0.3, bond=30 → 0.5, bond=60 → 0.7, bond=90 → 0.85
        bond_factor = math.log(1 + bond) / math.log(101)  # 归一化到 0~1
        interaction_boost = min(0.15, math.log(1 + interactions) * 0.02)
        return round(min(1.0, bond_factor + interaction_boost), 2)


_manager: Optional[IdentityManager] = None
_manager_lock = threading.Lock()


def get_identity_manager() -> IdentityManager:
    global _manager
    if _manager is None:
        with _manager_lock:
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
        "age_days": im.get_age_days(),
        "is_awakened": im.is_awakened(),
    }
