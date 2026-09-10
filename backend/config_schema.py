"""配置 Schema 校验 — Pydantic v2 模型

覆盖: agents / models / tools / session / cron / mem / compaction / contextPruning
提供: validate_config(), sanitize_config_for_client()
"""

import copy
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from config import DEFAULT_CONFIG, merge_config_defaults


_HEARTBEAT_DEFAULTS = DEFAULT_CONFIG["agents"]["defaults"]["heartbeat"]
_ACTIVE_HOURS_DEFAULTS = _HEARTBEAT_DEFAULTS["activeHours"]


# ---------------------------------------------------------------------------
# Models 配置
# ---------------------------------------------------------------------------

class ModelEntry(BaseModel):
    id: str
    name: Optional[str] = None
    reasoning: bool = False
    input: List[str] = Field(default_factory=lambda: ["text"])
    context_window: Optional[int] = Field(default=None, alias="contextWindow")
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens")
    cost: Optional[Dict[str, Any]] = None

    model_config = {"populate_by_name": True}


class ProviderConfig(BaseModel):
    baseUrl: Optional[str] = None
    apiKey: Optional[str] = None
    api: Optional[str] = None
    models: List[ModelEntry] = Field(default_factory=list)

    model_config = {"populate_by_name": True, "extra": "allow"}


class ModelsConfig(BaseModel):
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Heartbeat 配置
# ---------------------------------------------------------------------------

class ActiveHoursConfig(BaseModel):
    start: str = _ACTIVE_HOURS_DEFAULTS["start"]
    end: str = _ACTIVE_HOURS_DEFAULTS["end"]


class HeartbeatConfig(BaseModel):
    enabled: bool = _HEARTBEAT_DEFAULTS["enabled"]
    every: str = _HEARTBEAT_DEFAULTS["every"]
    prompt: Optional[str] = None
    ackMaxChars: int = _HEARTBEAT_DEFAULTS["ackMaxChars"]
    activeHours: Optional[ActiveHoursConfig] = Field(
        default_factory=ActiveHoursConfig
    )
    target: str = _HEARTBEAT_DEFAULTS["target"]
    maxEvents: int = Field(
        default=_HEARTBEAT_DEFAULTS["maxEvents"],
        ge=10,
        le=200,
    )

    @field_validator("every")
    @classmethod
    def validate_every(cls, v: str) -> str:
        if not v or not v.strip():
            return _HEARTBEAT_DEFAULTS["every"]
        s = v.strip().lower()
        if s in ("0", "0m", "0h", "disabled", "off"):
            return s
        if not re.match(r"^\d+\s*(m|min|h|hr|s)?$", s):
            raise ValueError(f"Invalid heartbeat interval: {v}")
        return s


# ---------------------------------------------------------------------------
# Compaction 配置
# ---------------------------------------------------------------------------

class CompactionConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.54
    slidingThreshold: float = 0.54
    forcedThreshold: float = 0.70
    reserveTokens: int = 20000
    keepRecentTokens: int = 8000
    keepRecentTurns: int = 12
    forcedKeepRecentTurns: int = 4
    summaryMaxTokens: int = 3000
    maxHistoryShare: float = 0.5
    softThresholdTokens: int = 4000


# ---------------------------------------------------------------------------
# Context Pruning 配置
# ---------------------------------------------------------------------------

class ContextPruningConfig(BaseModel):
    softTrim: bool = True
    toolOutputMaxChars: int = 15000
    recentPreserve: int = 4


# ---------------------------------------------------------------------------
# Subagents 配置
# ---------------------------------------------------------------------------

class SubagentsConfig(BaseModel):
    allow_agents: List[str] = Field(default_factory=lambda: ["*"])
    max_spawn_depth: int = 2
    max_children_per_agent: int = 5
    archive_after_minutes: int = 60
    recent_minutes: int = 30
    run_timeout_seconds: int = 0  # 0 = 无超时


class ChatConfig(BaseModel):
    timeoutSeconds: int = 120  # 0 = 无超时


# ---------------------------------------------------------------------------
# Tools Policy
# ---------------------------------------------------------------------------

class ToolsPolicyConfig(BaseModel):
    allow: Optional[List[str]] = None
    deny: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Agent Defaults
# ---------------------------------------------------------------------------

class AgentStatePersistConfig(BaseModel):
    """Agent 状态持久化配置"""
    enabled: bool = True  # 是否启用状态持久化
    autoSaveIntervalMinutes: int = Field(default=5, ge=1, le=60)  # 自动保存间隔


class ContextBudgetConfig(BaseModel):
    """上下文预算比率 — 从 contextTokens 派生所有分配值"""
    thinking_reserve: float = 0.20
    active_ratio: float = 0.80
    session_summary_ratio: float = 0.05
    memory_injection_ratio: float = 0.05
    jit_tool_output_ratio: float = 0.036
    sliding_ratio: float = 0.80
    forced_ratio: float = 0.95


class AgentDefaultsConfig(BaseModel):
    model: Optional[str] = None
    user_timezone: str = "Asia/Shanghai"
    recursion_limit: int = 50
    contextTokens: int = 200000
    thinkingDefault: str = "off"
    contextBudget: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    contextPruning: ContextPruningConfig = Field(default_factory=ContextPruningConfig)
    subagents: SubagentsConfig = Field(default_factory=SubagentsConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    tools: Optional[ToolsPolicyConfig] = None
    statePersist: AgentStatePersistConfig = Field(default_factory=AgentStatePersistConfig)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Agent Entry
# ---------------------------------------------------------------------------

class AgentEntryConfig(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    model: Optional[str] = None
    subagents: Optional[SubagentsConfig] = None
    heartbeat: Optional[HeartbeatConfig] = None
    tools: Optional[ToolsPolicyConfig] = None

    model_config = {"extra": "allow"}

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(f"Agent ID must be alphanumeric/dash/underscore: {v}")
        return v.strip()


# ---------------------------------------------------------------------------
# Agents 配置
# ---------------------------------------------------------------------------

class AgentsConfig(BaseModel):
    defaults: AgentDefaultsConfig = Field(default_factory=AgentDefaultsConfig)
    list: List[AgentEntryConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_at_least_one_agent(self) -> "AgentsConfig":
        if not self.list:
            self.list = [
                AgentEntryConfig.model_validate(copy.deepcopy(agent))
                for agent in DEFAULT_CONFIG["agents"]["list"]
            ]
        return self


# ---------------------------------------------------------------------------
# Tools 配置
# ---------------------------------------------------------------------------

class FsToolsConfig(BaseModel):
    workspace_only: bool = True
    readonly_dirs: List[str] = Field(default_factory=lambda: ["docs"])


class ExecToolConfig(BaseModel):
    enabled: bool = False


class ExecApprovalConfig(BaseModel):
    security: Literal["deny", "allowlist", "full"] = "allowlist"
    ask: Literal["off", "on_miss", "always"] = "on_miss"
    ask_timeout_seconds: int = 60
    pending_timeout_seconds: int = Field(default=300, ge=60, le=3600)  # 审批请求在队列中的最大存活时间
    allowlist: list[str] = Field(default_factory=list)


class ExecToolsConfig(BaseModel):
    apply_patch: ExecToolConfig = Field(default_factory=ExecToolConfig)
    approval: ExecApprovalConfig = Field(default_factory=ExecApprovalConfig)
    maxOutputChars: int = Field(default=5000, ge=1000, le=50000)  # 执行工具输出最大字符数


class WebSearchConfig(BaseModel):
    provider: str = "duckduckgo"
    apiKey: str = ""
    baseUrl: str = ""


class WebToolsConfig(BaseModel):
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class KnowledgeToolsConfig(BaseModel):
    projectPath: str = ""
    pythonPath: str = ""
    configPath: str = ""
    collectionName: str = "pipixia_{agent_id}"
    timeoutSeconds: int = Field(default=120, ge=1, le=600)


class ToolsConfig(BaseModel):
    fs: FsToolsConfig = Field(default_factory=FsToolsConfig)
    exec: ExecToolsConfig = Field(default_factory=ExecToolsConfig)
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    knowledge: KnowledgeToolsConfig = Field(default_factory=KnowledgeToolsConfig)


# ---------------------------------------------------------------------------
# Session 配置
# ---------------------------------------------------------------------------

class SessionMaintenanceConfig(BaseModel):
    mode: str = "warn"
    pruneAfter: str = "30d"
    maxEntries: int = 500
    maxDiskBytes: Optional[int] = None
    highWaterBytes: Optional[int] = None


class SessionConfig(BaseModel):
    maintenance: SessionMaintenanceConfig = Field(default_factory=SessionMaintenanceConfig)
    titleMaxLen: int = Field(default=60, ge=20, le=120)  # 会话标题最大长度


# ---------------------------------------------------------------------------
# Cron 配置
# ---------------------------------------------------------------------------

class CronConfig(BaseModel):
    enabled: bool = False
    store: Optional[str] = None


# ---------------------------------------------------------------------------
# App Config (locale, theme, data directory, logging, proxy)
# ---------------------------------------------------------------------------

class MessageQueueConfig(BaseModel):
    """消息队列配置"""
    debounceMs: int = Field(default=1000, ge=100, le=10000)


class SystemEventsConfig(BaseModel):
    """系统事件队列配置"""
    maxEvents: int = Field(default=20, ge=5, le=100)  # 系统事件队列大小


class LogFileConfig(BaseModel):
    """日志文件配置"""
    enabled: bool = True  # 是否启用文件日志
    maxBytes: int = Field(default=10 * 1024 * 1024, ge=1024)  # 单个文件最大大小，默认10MB
    backupCount: int = Field(default=5, ge=0)  # 保留的备份文件数量


class AppConfig(BaseModel):
    locale: str = "zh-CN"
    theme: str = "system"
    dataDir: Optional[str] = None
    logLevel: str = "info"
    logFile: LogFileConfig = Field(default_factory=LogFileConfig)
    messageQueue: MessageQueueConfig = Field(default_factory=MessageQueueConfig)
    systemEvents: SystemEventsConfig = Field(default_factory=SystemEventsConfig)
    proxy: Optional[str] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Notifications Config
# ---------------------------------------------------------------------------

class QuietHoursConfig(BaseModel):
    start: str = "23:00"
    end: str = "08:00"


class NotificationsConfig(BaseModel):
    enabled: bool = True
    sound: bool = True
    badge: bool = True
    quietHours: QuietHoursConfig = Field(default_factory=QuietHoursConfig)


# ---------------------------------------------------------------------------
# Sandbox Config
# ---------------------------------------------------------------------------

class LoopDetectionConfig(BaseModel):
    """循环检测配置"""
    warningThreshold: int = Field(default=10, ge=3, le=50)  # 警告阈值
    criticalThreshold: int = Field(default=20, ge=5, le=100)  # 严重警告阈值
    circuitBreaker: int = Field(default=30, ge=10, le=200)  # 熔断阈值
    historySize: int = Field(default=30, ge=10, le=100)  # 历史记录大小


class SandboxConfig(BaseModel):
    mode: Literal["off", "soft", "strict"] = "soft"
    snapshotBeforeExec: bool = False
    undoStackSize: int = 50
    writeApproval: Literal["off", "on_overwrite", "always"] = "on_overwrite"
    loopDetection: LoopDetectionConfig = Field(default_factory=LoopDetectionConfig)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Backup Config
# ---------------------------------------------------------------------------

class BackupConfig(BaseModel):
    autoBackup: bool = False
    intervalHours: int = 24
    maxSnapshots: int = 10
    backupDir: Optional[str] = None


# ---------------------------------------------------------------------------
# Runtime Config
# ---------------------------------------------------------------------------

class RuntimeConfig(BaseModel):
    maxConcurrentSessions: int = 5
    memoryLimitMB: int = 0
    processTimeoutSeconds: int = 300
    gcIdleMinutes: int = 30


# ---------------------------------------------------------------------------
# Skills Config
# ---------------------------------------------------------------------------

class SkillsGlobalConfig(BaseModel):
    autoDiscover: bool = True
    trustedSources: List[str] = Field(default_factory=list)
    updateCheck: bool = False


# ---------------------------------------------------------------------------
# Browser Config
# ---------------------------------------------------------------------------

class BrowserConfig(BaseModel):
    enabled: bool = False
    headless: bool = True
    viewport: Optional[str] = "1280x720"
    proxy: Optional[str] = None


# ---------------------------------------------------------------------------
# Mem (Memory System) Config — docs/memory-system-refactor.md §10
# ---------------------------------------------------------------------------

class MemStorageConfig(BaseModel):
    db_path: str = "{data_dir}/mem/mem.db"


class MemEmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = ""
    dimensions: int = 1536
    batch_size: int = 32
    timeout: float = 60.0


class MemLLMConfig(BaseModel):
    model: str = "qwen-plus"
    api_key: str = ""
    base_url: str = ""


class MemRecallConfig(BaseModel):
    max_task_results: int = 5
    min_task_hits: int = 3
    chunks_per_task: int = 5
    max_orphan_chunks: int = 5
    final_chunk_top_k: int = 5
    max_skill_results: int = 0
    budget_chars: int = 20000
    skill_budget_chars: int = 2000
    min_task_score: float = 0.015
    rrf_k: int = 60
    recency_half_life_days: float = 14
    min_inject_score: float = 0.015


class MemDedupConfig(BaseModel):
    similarity_threshold: float = 0.60


class MemTaskConfig(BaseModel):
    idle_timeout_hours: float = 2


class MemSkillEvolutionConfig(BaseModel):
    enabled: bool = True
    min_chunks_for_eval: int = 6
    min_confidence: float = 0.7
    auto_install: bool = False


class MemConfig(BaseModel):
    enabled: bool = True
    storage: MemStorageConfig = Field(default_factory=MemStorageConfig)
    llm: MemLLMConfig = Field(default_factory=MemLLMConfig)
    embedding: MemEmbeddingConfig = Field(default_factory=MemEmbeddingConfig)
    recall: MemRecallConfig = Field(default_factory=MemRecallConfig)
    dedup: MemDedupConfig = Field(default_factory=MemDedupConfig)
    task: MemTaskConfig = Field(default_factory=MemTaskConfig)
    skill_evolution: MemSkillEvolutionConfig = Field(default_factory=MemSkillEvolutionConfig)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Root Config
# ---------------------------------------------------------------------------

class PIPIXIAConfig(BaseModel):
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    cron: CronConfig = Field(default_factory=CronConfig)
    mem: MemConfig = Field(default_factory=MemConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    skills: SkillsGlobalConfig = Field(default_factory=SkillsGlobalConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)

    model_config = {"populate_by_name": True, "extra": "allow"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ConfigValidationResult:
    def __init__(self, ok: bool, config: Optional[Dict[str, Any]] = None, errors: Optional[List[str]] = None):
        self.ok = ok
        self.config = config
        self.errors = errors or []


def validate_config(raw: Dict[str, Any]) -> ConfigValidationResult:
    try:
        parsed = PIPIXIAConfig.model_validate(merge_config_defaults(raw))
        return ConfigValidationResult(
            ok=True,
            config=parsed.model_dump(
                by_alias=True,
                exclude_unset=True,
            ),
        )
    except Exception as e:
        errors = []
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                errors.append(f"{loc}: {err.get('msg', str(err))}")
        else:
            errors.append(str(e))
        return ConfigValidationResult(ok=False, errors=errors)


# ---------------------------------------------------------------------------
# Sanitization (remove sensitive fields for client)
# ---------------------------------------------------------------------------

SENSITIVE_KEYS = {"apiKey", "api_key", "token", "secret", "password"}


def _mask_value(v: Any) -> str:
    if not v or not isinstance(v, str):
        return ""
    if len(v) <= 8:
        return "***"
    return v[:4] + "***" + v[-4:]


def _sanitize_dict(d: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    if depth > 10:
        return d
    result = {}
    for k, v in d.items():
        k_lower = k.lower().replace("-", "").replace("_", "")
        if any(sk.lower().replace("-", "").replace("_", "") == k_lower for sk in SENSITIVE_KEYS):
            result[k] = _mask_value(v)
        elif isinstance(v, dict):
            result[k] = _sanitize_dict(v, depth + 1)
        elif isinstance(v, list):
            result[k] = [_sanitize_dict(item, depth + 1) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result


def sanitize_config_for_client(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _sanitize_dict(copy.deepcopy(cfg))
