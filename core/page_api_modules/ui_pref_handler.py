"""
UI 偏好处理模块
提供 WebUI 轻量偏好（主题模式等）的持久化。

背景：插件页 iframe 由 dashboard 以 sandbox（无 allow-same-origin）嵌入，
页面内 localStorage 读写会抛 SecurityError，主题模式等偏好无法在浏览器侧
持久化；因此经桥接（走父页面鉴权）存到插件数据目录。
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from quart import request

from astrbot.api import logger

if TYPE_CHECKING:
    from .utils import PageApiUtils

ALLOWED_THEME_MODES = {"auto", "light", "dark"}


class UiPrefHandler:
    """UI 偏好处理器（纯文件读写，不依赖记忆引擎就绪）"""

    def __init__(self, utils: "PageApiUtils", data_dir: str = ""):
        self.utils = utils
        self.data_dir = data_dir or ""

    def _prefs_path(self) -> str:
        if not self.data_dir:
            return ""
        return os.path.join(self.data_dir, "ui_prefs.json")

    def _load(self) -> dict[str, Any]:
        path = self._prefs_path()
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"[livingmemory] 读取 ui_prefs 失败: {exc}")
            return {}

    def _save(self, prefs: dict[str, Any]) -> None:
        path = self._prefs_path()
        if not path:
            raise OSError("data_dir 未就绪")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    async def get_ui_pref(self) -> dict[str, Any]:
        """读取 UI 偏好（当前含 theme_mode）"""
        prefs = self._load()
        return self.utils.ok({"theme_mode": prefs.get("theme_mode", "")})

    async def update_ui_pref(self) -> dict[str, Any]:
        """更新 UI 偏好

        Payload:
            - theme_mode: auto / light / dark
        """
        payload = await request.get_json(silent=True) or {}
        mode = str(payload.get("theme_mode", "")).strip().lower()
        if mode not in ALLOWED_THEME_MODES:
            return self.utils.error("theme_mode 必须是 auto/light/dark")

        prefs = self._load()
        prefs["theme_mode"] = mode
        prefs["updated_at"] = time.time()
        try:
            self._save(prefs)
        except OSError as exc:
            logger.warning(f"[livingmemory] 写入 ui_prefs 失败: {exc}")
            return self.utils.error("保存 UI 偏好失败")
        return self.utils.ok({"theme_mode": mode})
