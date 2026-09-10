"""全局配置管理 — 从 data/config.json 加载并持久化

支持:
  - 从 config.template.json 初始化
  - ${ENV_VAR} 环境变量引用解析
  - Schema 校验写回保护
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # pipixia 项目根目录
_DATA_DIR_OVERRIDE = os.getenv("PIPIXIA_DATA_DIR", "").strip()
DATA_DIR = Path(_DATA_DIR_OVERRIDE or str(BASE_DIR / "data")).resolve()
TEMPLATE_PATH = BASE_DIR / "data" / "config.template.json"

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Heartbeat 默认常量
DEFAULT_HEARTBEAT_PROMPT = (
    "Read HEARTBEAT.md if it exists (workspace context). Follow it strictly. "
    "Do not infer or repeat old tasks from prior chats. "
    "If nothing needs attention, reply HEARTBEAT_OK."
)
DEFAULT_HEARTBEAT_EVERY = "30m"
DEFAULT_HEARTBEAT_ACK_MAX_CHARS = 300

DEFAULT_CONFIG: dict[str, Any] = {
    "agents": {
        "defaults": {
            "model": None,
            "user_timezone": "Asia/Shanghai",
            "recursion_limit": 50,
            "contextTokens": 200000,
            "thinkingDefault": "off",
            "contextBudget": {
                "thinking_reserve": 0.20,
                "active_ratio": 0.80,
                "session_summary_ratio": 0.05,
                "memory_injection_ratio": 0.05,
                "jit_tool_output_ratio": 0.036,
                "sliding_ratio": 0.80,
                "forced_ratio": 0.95,
            },
            "compaction": {
                "enabled": True,
                "threshold": 0.54,
                "slidingThreshold": 0.54,
                "forcedThreshold": 0.70,
                "reserveTokens": 20000,
                "keepRecentTokens": 8000,
                "keepRecentTurns": 12,
                "forcedKeepRecentTurns": 4,
                "summaryMaxTokens": 3000,
                "maxHistoryShare": 0.5,
                "softThresholdTokens": 4000,
            },
            "contextPruning": {
                "softTrim": True,
                "toolOutputMaxChars": 15000,
                "recentPreserve": 4,
            },
            "subagents": {
                "allow_agents": ["*"],
                "max_spawn_depth": 2,
                "max_children_per_agent": 5,
                "archive_after_minutes": 60,
                "recent_minutes": 30,
                "run_timeout_seconds": 0,  # 0 = 无超时
            },
            "heartbeat": {
                "enabled": False,
                "every": DEFAULT_HEARTBEAT_EVERY,
                "ackMaxChars": DEFAULT_HEARTBEAT_ACK_MAX_CHARS,
                "activeHours": {"start": "08:00", "end": "24:00"},
                "target": "webchat",
                "maxEvents": 50,
            },
            "statePersist": {
                "enabled": True,
                "autoSaveIntervalMinutes": 5,
            },
        },
        "list": [
            {
                "id": "main",
                "name": "Assistant",
                "description": "Default general-purpose assistant",
            }
        ],
    },
    "models": {
        "providers": {},
    },
    "tools": {
        "fs": {"workspace_only": True, "readonly_dirs": ["docs"]},
        "exec": {
            "apply_patch": {"enabled": False},
            "maxOutputChars": 5000,
            "approval": {
                "security": "full",
                "ask": "on_miss",
                "ask_timeout_seconds": 60,
                "pending_timeout_seconds": 300,
                "allowlist": [],
            },
        },
        "web": {
            "search": {
                "provider": "duckduckgo",
                "apiKey": "",
                "baseUrl": "",
            },
        },
        "knowledge": {
            "projectPath": "",
            "pythonPath": "",
            "configPath": "",
            "collectionName": "pipixia_{agent_id}",
            "timeoutSeconds": 120,
        },
    },
    "chat": {
        "timeoutSeconds": 120,  # 0 = 无超时
    },
    "session": {
        "maintenance": {
            "mode": "warn",
            "pruneAfter": "30d",
            "maxEntries": 500,
            "maxDiskBytes": None,
            "highWaterBytes": None,
        },
        "titleMaxLen": 60,
    },
    "cron": {
        "enabled": False,
        "store": None,
    },
    "app": {
        "locale": "zh-CN",
        "theme": "system",
        "dataDir": None,
        "logLevel": "info",
        "logFile": {
            "enabled": True,
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
        },
        "messageQueue": {
            "debounceMs": 1000,
        },
        "systemEvents": {
            "maxEvents": 20,
        },
        "proxy": None,
    },
    "notifications": {
        "enabled": True,
        "sound": True,
        "badge": True,
        "quietHours": {"start": "23:00", "end": "08:00"},
    },
    "sandbox": {
        "mode": "soft",
        "snapshotBeforeExec": False,
        "undoStackSize": 50,
        "writeApproval": "on_overwrite",
        "loopDetection": {
            "warningThreshold": 10,
            "criticalThreshold": 20,
            "circuitBreaker": 30,
            "historySize": 30,
        },
    },
    "backup": {
        "autoBackup": False,
        "intervalHours": 24,
        "maxSnapshots": 10,
        "backupDir": None,
    },
    "runtime": {
        "maxConcurrentSessions": 5,
        "memoryLimitMB": 0,
        "processTimeoutSeconds": 300,
        "gcIdleMinutes": 30,
    },
    "skills": {
        "autoDiscover": True,
        "trustedSources": [],
        "updateCheck": False,
    },
    "browser": {
        "enabled": False,
        "headless": True,
        "viewport": "1280x720",
        "proxy": None,
    },
    "mem": {
        "enabled": True,
        "storage": {
            "db_path": "{data_dir}/mem/mem.db",
        },
        "llm": {
            "model": "qwen-plus",
            "api_key": "",
            "base_url": "",
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "",
            "base_url": "",
            "dimensions": 1536,
            "batch_size": 32,
            "timeout": 60.0,
        },
        "recall": {
            "max_task_results": 5,
            "min_task_hits": 3,
            "chunks_per_task": 5,
            "max_orphan_chunks": 5,
            "final_chunk_top_k": 5,
            "max_skill_results": 0,
            "budget_chars": 20000,
            "skill_budget_chars": 2000,
            "min_task_score": 0.015,
            "rrf_k": 60,
            "recency_half_life_days": 14,
            "min_inject_score": 0.015,
        },
        "dedup": {
            "similarity_threshold": 0.60,
        },
        "task": {
            "idle_timeout_hours": 2,
        },
        "skill_evolution": {
            "enabled": True,
            "min_chunks_for_eval": 6,
            "min_confidence": 0.7,
            "auto_install": False,
        },
    },
}

_config: dict[str, Any] | None = None
_raw_config: dict[str, Any] | None = None


def _config_path() -> Path:
    """配置文件路径。可通过 PIPIXIA_CONFIG_PATH 覆盖，否则为 data/config.json"""
    override = os.getenv("PIPIXIA_CONFIG_PATH", "").strip()
    if override:
        return Path(override).resolve()
    return DATA_DIR / "config.json"


def get_config_path() -> str:
    """返回配置文件绝对路径，供前端展示"""
    return str(_config_path())


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并：override 中的值覆盖 base，缺失的保留 base 默认值"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def merge_config_defaults(override: dict[str, Any]) -> dict[str, Any]:
    """将局部配置合并到唯一的运行时默认配置。"""
    return _deep_merge(DEFAULT_CONFIG, override)


def _resolve_env_vars(obj: Any) -> Any:
    """递归解析 ${ENV_VAR} 引用，替换为环境变量值（未设置则保留原字符串）"""
    if isinstance(obj, str):
        def _replace(m: re.Match) -> str:
            var_name = m.group(1)
            return os.environ.get(var_name, m.group(0))
        return _ENV_VAR_RE.sub(_replace, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def _init_from_template() -> dict[str, Any]:
    """使用运行时默认配置初始化；模板仅作为受测试约束的发布镜像。"""
    return copy.deepcopy(DEFAULT_CONFIG)


def is_initialized() -> bool:
    """检查配置文件是否已创建"""
    return _config_path().exists()


def load_config() -> dict[str, Any]:
    global _config, _raw_config
    p = _config_path()
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            saved = json.load(f)
        _raw_config = merge_config_defaults(saved)
    else:
        _raw_config = _init_from_template()
        save_config(_raw_config, validate=False)

    # 迁移：若存在顶层 heartbeat（旧版），迁移到 agents.defaults.heartbeat
    legacy_hb = _raw_config.get("heartbeat")
    if legacy_hb and isinstance(legacy_hb, dict):
        agents_cfg = _raw_config.setdefault("agents", {})
        defaults_cfg = agents_cfg.setdefault("defaults", {})
        defaults_cfg["heartbeat"] = {
            "enabled": legacy_hb.get("enabled", True),
            "every": f"{legacy_hb.get('interval_seconds', 1800) // 60}m",
            "prompt": DEFAULT_HEARTBEAT_PROMPT,
            "ackMaxChars": 300,
            "target": "webchat",
        }
        qh = legacy_hb.get("quiet_hours") or {}
        if qh:
            sh, eh = qh.get("start", 23), qh.get("end", 8)
            defaults_cfg["heartbeat"]["activeHours"] = {
                "start": f"{sh:02d}:00",
                "end": f"{eh:02d}:00" if eh < 24 else "24:00",
            }
        else:
            defaults_cfg["heartbeat"]["activeHours"] = {"start": "08:00", "end": "24:00"}
        del _raw_config["heartbeat"]
        save_config(_raw_config, validate=False)

    agents_list = _raw_config.get("agents", {}).get("list", [])
    if not agents_list:
        main_agent = {
            "id": "main",
            "name": "Assistant",
            "description": "Default general-purpose assistant",
        }
        _raw_config.setdefault("agents", {})["list"] = [main_agent]
        save_config(_raw_config, validate=False)

    _config = _resolve_env_vars(copy.deepcopy(_raw_config))

    _init_models_config(_config)
    return _config


def _init_models_config(cfg: dict[str, Any]) -> None:
    """初始化模型配置管理器"""
    try:
        from llm.models_config import models_config
        models_config.initialize(cfg.get("models"))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to initialize models config: {e}")


def get_config() -> dict[str, Any]:
    if _config is None:
        return load_config()
    return _config


def get_raw_config() -> dict[str, Any]:
    global _raw_config
    if _raw_config is None:
        load_config()
    return copy.deepcopy(_raw_config if _raw_config is not None else DEFAULT_CONFIG)


def save_config(cfg: dict[str, Any], *, validate: bool = True) -> None:
    """持久化配置。validate=True 时写前校验，非法配置拒绝落盘并抛异常。"""
    global _config, _raw_config
    if validate:
        from config_schema import validate_config
        result = validate_config(cfg)
        if not result.ok:
            raise ValueError(f"Config validation failed: {'; '.join(result.errors)}")
    next_raw_config = copy.deepcopy(cfg)
    next_config = _resolve_env_vars(copy.deepcopy(cfg))
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        shutil.move(str(tmp), str(p))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    _raw_config = next_raw_config
    _config = next_config


def resolve_mem_config() -> dict:
    """Return the `mem` config section with db_path placeholder resolved."""
    cfg = get_config().get("mem", {})
    storage = dict(cfg.get("storage", {}))
    db_path = storage.get("db_path", "{data_dir}/mem/mem.db")
    storage["db_path"] = db_path.replace("{data_dir}", str(DATA_DIR))
    return {**cfg, "storage": storage}


def resolve_chat_timeout_seconds() -> int:
    """聊天请求超时秒数。0 表示无超时。"""
    raw = get_config().get("chat", {}).get("timeoutSeconds")
    if raw is None:
        return 120
    try:
        v = int(raw)
        return max(0, v)
    except (TypeError, ValueError):
        return 120


# ---------------------------------------------------------------------------
# Agent 配置解析
# ---------------------------------------------------------------------------

def list_agents() -> list[dict[str, Any]]:
    return get_config().get("agents", {}).get("list", [])


def get_agent_defaults() -> dict[str, Any]:
    return get_config().get("agents", {}).get("defaults", {})


def resolve_agent_config(agent_id: str) -> dict[str, Any]:
    """合并 defaults + agent 级别覆盖，返回完整配置"""
    defaults = get_agent_defaults()
    for agent in list_agents():
        if agent["id"] == agent_id:
            override = {k: v for k, v in agent.items() if v is not None}
            return _deep_merge(defaults, override)
    return defaults


def resolve_tool_policy_from_config(
    cfg: dict[str, Any],
    agent_id: str,
) -> tuple[list[str], list[str]]:
    """从指定配置解析 Agent 的工具 allow/deny 策略。"""
    agent_entry = None
    for a in (cfg.get("agents", {}).get("list") or []):
        if a.get("id") == agent_id:
            agent_entry = a
            break
    policy = (agent_entry or {}).get("tools") or {}
    defaults_policy = (cfg.get("agents", {}).get("defaults", {}).get("tools")) or {}
    deny = list(policy.get("deny") or defaults_policy.get("deny") or [])
    allow = list(policy.get("allow") or defaults_policy.get("allow") or [])
    return allow, deny


def resolve_tool_policy(agent_id: str) -> tuple[list[str], list[str]]:
    """解析当前配置中的 Agent 工具 allow/deny 策略。"""
    return resolve_tool_policy_from_config(get_config(), agent_id)


def normalize_tool_name(name: str) -> str:
    return name.replace("-", "_").lower().strip()


def is_tool_name_allowed(
    tool_name: str,
    allow: list[str],
    deny: list[str],
) -> bool:
    """判断工具名是否满足已解析的 allow/deny 策略。"""

    normalized = normalize_tool_name(tool_name)
    deny_set = {normalize_tool_name(name) for name in deny if name}
    allow_set = (
        {normalize_tool_name(name) for name in allow if name}
        if allow
        else None
    )

    if normalized in deny_set:
        return False
    if normalized == "apply_patch" and "exec" in deny_set:
        return False
    if allow_set is None:
        return True
    if normalized in allow_set:
        return True
    if normalized == "apply_patch" and "exec" in allow_set:
        return True
    return False


def is_tool_allowed_by_policy(agent_id: str, tool_name: str) -> bool:
    """判断工具是否满足 Agent 的 allow/deny 策略。"""
    allow, deny = resolve_tool_policy(agent_id)
    return is_tool_name_allowed(tool_name, allow, deny)


def resolve_tool_catalog_signature() -> str:
    """返回影响工具集合的配置签名，用于运行时缓存失效。"""
    tools_config = get_config().get("tools") or {}
    return json.dumps(
        tools_config,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def get_exec_approval_config() -> dict[str, Any]:
    """获取 exec 工具的 approval 配置（security/ask/allowlist）"""
    cfg = get_config()
    exec_cfg = (cfg.get("tools") or {}).get("exec") or {}
    approval = exec_cfg.get("approval")
    if approval is None:
        base = {
            "security": "full",
            "ask": "on_miss",
            "ask_timeout_seconds": 60,
            "allowlist": [],
        }
    else:
        base = {
            "security": approval.get("security", "allowlist"),
            "ask": approval.get("ask", "on_miss"),
            "ask_timeout_seconds": int(approval.get("ask_timeout_seconds", 60)),
            "allowlist": list(approval.get("allowlist") or []),
        }

    # 合并 data/exec-approvals.json 中的 allowlist（若存在）
    approvals_path = DATA_DIR / "exec-approvals.json"
    if approvals_path.exists():
        try:
            with open(approvals_path, "r", encoding="utf-8") as f:
                extra = json.load(f)
            extra_list = extra.get("allowlist") or []
            if extra_list:
                seen = {p for p in base["allowlist"]}
                for p in extra_list:
                    if p and p not in seen:
                        base["allowlist"].append(p)
                        seen.add(p)
        except Exception:
            pass
    return base


def resolve_agent_dir(agent_id: str) -> Path:
    return DATA_DIR / "agents" / agent_id


def resolve_agent_workspace(agent_id: str) -> Path:
    return resolve_agent_dir(agent_id) / "workspace"


def resolve_agent_skills_dir(agent_id: str) -> Path:
    """Skills 位于 workspace 内，Agent 可通过文件工具读写"""
    return resolve_agent_workspace(agent_id) / "skills"


def resolve_agent_sessions_dir(agent_id: str) -> Path:
    return resolve_agent_dir(agent_id) / "sessions"


def resolve_agent_knowledge_dir(agent_id: str) -> Path:
    return resolve_agent_dir(agent_id) / "knowledge"


def resolve_agent_storage_dir(agent_id: str) -> Path:
    return resolve_agent_dir(agent_id) / "storage"


def resolve_global_skills_dir() -> Path:
    return DATA_DIR / "skills"


# ---------------------------------------------------------------------------
# 心跳配置解析（per-agent，merge defaults + list[i].heartbeat）
# ---------------------------------------------------------------------------

def resolve_heartbeat_prompt(raw: str | None = None) -> str:
    """解析心跳 prompt：配置值非空时使用配置值，否则回退到内置默认。"""
    trimmed = raw.strip() if isinstance(raw, str) else ""
    return trimmed or DEFAULT_HEARTBEAT_PROMPT


def _parse_every_to_seconds(raw: str | None) -> int | None:
    """解析 every 字符串（如 30m、1h）为秒数；0m 或空返回 None 表示关闭"""
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    if s in ("0", "0m", "0h", "disabled", "off"):
        return None
    import re
    m = re.match(r"^(\d+)\s*(m|min|h|hr|s)?$", s)
    if not m:
        return None
    num = int(m.group(1))
    unit = (m.group(2) or "m").lower()
    if unit.startswith("h"):
        return num * 3600
    if unit.startswith("s"):
        return num
    return num * 60


def get_heartbeat_config(agent_id: str) -> dict[str, Any]:
    """合并 agents.defaults.heartbeat + agents.list[i].heartbeat，返回该 agent 的心跳配置。

    prompt 字段通过 resolve_heartbeat_prompt() 解析：配置值为空时回退到内置默认常量。
    """
    cfg = get_config()
    defaults = cfg.get("agents", {}).get("defaults", {})
    hb_defaults = defaults.get("heartbeat") or {}

    # 兼容旧版：顶层 heartbeat
    legacy = cfg.get("heartbeat") or {}
    if not hb_defaults and legacy:
        hb_defaults = {
            "enabled": legacy.get("enabled", True),
            "every": f"{legacy.get('interval_seconds', 1800) // 60}m",
            "ackMaxChars": DEFAULT_HEARTBEAT_ACK_MAX_CHARS,
            "activeHours": None,
            "target": "webchat",
        }
        qh = legacy.get("quiet_hours") or {}
        if qh:
            start_h, end_h = qh.get("start", 23), qh.get("end", 8)
            hb_defaults["activeHours"] = {
                "start": f"{start_h:02d}:00",
                "end": f"{end_h:02d}:00" if end_h < 24 else "24:00",
            }
    if not hb_defaults:
        hb_defaults = {
            "enabled": False,
            "every": DEFAULT_HEARTBEAT_EVERY,
            "ackMaxChars": DEFAULT_HEARTBEAT_ACK_MAX_CHARS,
            "activeHours": {"start": "08:00", "end": "24:00"},
            "target": "webchat",
        }

    # per-agent 覆盖
    override = {}
    for agent in list_agents():
        if agent.get("id") == agent_id and agent.get("heartbeat"):
            override = agent["heartbeat"]
            break
    merged = {**hb_defaults, **{k: v for k, v in override.items() if v is not None}}

    # 统一解析 prompt（空值回退到内置默认）
    merged["prompt"] = resolve_heartbeat_prompt(merged.get("prompt"))

    every_raw = merged.get("every", DEFAULT_HEARTBEAT_EVERY)
    interval_sec = _parse_every_to_seconds(every_raw)
    merged["interval_seconds"] = interval_sec
    return merged


def is_cron_enabled() -> bool:
    cfg = get_config()
    cron_cfg = cfg.get("cron") or {}
    return bool(cron_cfg.get("enabled", False))


def add_agent_to_config(agent_cfg: dict[str, Any]) -> None:
    cfg = get_raw_config()
    agents_list = cfg.setdefault("agents", {}).setdefault("list", [])
    for existing in agents_list:
        if existing["id"] == agent_cfg["id"]:
            existing.update(agent_cfg)
            save_config(cfg)
            return
    agents_list.append(agent_cfg)
    save_config(cfg)


def remove_agent_from_config(agent_id: str) -> bool:
    """从配置中移除指定 Agent；至少保留一个时返回 False"""
    cfg = get_raw_config()
    agents_list = cfg.get("agents", {}).get("list", [])
    if len(agents_list) <= 1:
        return False
    new_list = [a for a in agents_list if a.get("id") != agent_id]
    if len(new_list) == len(agents_list):
        return False
    cfg["agents"]["list"] = new_list
    save_config(cfg)
    return True
