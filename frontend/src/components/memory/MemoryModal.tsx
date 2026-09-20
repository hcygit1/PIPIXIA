"use client";

import { useState, useEffect, useCallback } from "react";
import { useAgentContext } from "@/lib/agentContext";
import { useUi } from "@/lib/uiContext";
import SkillEvolutionWorkspace from "@/components/memory/SkillEvolutionWorkspace";
import * as api from "@/lib/api";
import {
  X, BrainCircuit, BarChart3, ListTodo, Sparkles, Search,
  ChevronRight, ChevronLeft, Loader2, Database, Clock,
  FileText, MessageSquare, Bot, User, RefreshCw,
  CircleHelp, Link, Plus, FolderInput, ArchiveX,
  GitBranch,
} from "lucide-react";

/* ================================================================
   Types
   ================================================================ */

interface MemStats {
  totalChunks: number;
  totalTasks: number;
  completedTasks: number;
  totalSkills: number;
  totalSessions: number;
  roleBreakdown: Record<string, number>;
  dedupBreakdown: Record<string, number>;
  timeRange: { earliest: string | null; latest: string | null };
}

interface TaskItem {
  id: string;
  sessionKey: string;
  title: string;
  summary: string;
  status: string;
  startedAt: number;
  endedAt: number | null;
  chunkCount: number;
}

interface TaskDetail {
  id: string;
  title: string;
  summary: string;
  status: string;
  startedAt: number;
  endedAt: number | null;
  chunks: { id: string; role: string; content: string; summary: string; createdAt: number }[];
}

interface SkillItem {
  id: string;
  name: string;
  description: string;
  version: number;
  status: string;
  qualityScore: number | null;
  createdAt: number;
  updatedAt: number;
}

interface MemoryItem {
  id: string;
  sessionKey: string;
  role: string;
  summary: string;
  excerpt: string;
  taskId: string | null;
  createdAt: number;
}

interface SearchResult {
  id: string;
  score: number;
  role: string;
  summary: string;
  excerpt: string;
  sessionKey: string;
  taskId: string | null;
  createdAt: number;
}

interface BoundaryReviewItem {
  id: string;
  sessionKey: string;
  currentTaskId: string;
  currentTaskTitle: string;
  currentTaskSummary: string;
  turnId: string;
  confidence: number;
  reason: string;
  retryCount: number;
  createdAt: number;
  recentContext: { id: string; role: string; content: string }[];
  pendingChunks: { id: string; role: string; content: string }[];
}

/* ================================================================
   Helpers
   ================================================================ */

function formatTs(ts: number | null) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: "var(--success)", completed: "var(--accent)", merged: "var(--warning)",
    deprecated: "var(--text-tertiary)", superseded: "var(--text-tertiary)",
  };
  const c = colors[status] || "var(--text-secondary)";
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium"
      style={{ background: `color-mix(in srgb, ${c} 15%, transparent)`, color: c }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
      {status}
    </span>
  );
}

function RoleIcon({ role }: { role: string }) {
  if (role === "assistant") return <Bot className="w-3 h-3" style={{ color: "var(--accent)" }} />;
  if (role === "user") return <User className="w-3 h-3" style={{ color: "var(--success)" }} />;
  return <MessageSquare className="w-3 h-3" style={{ color: "var(--text-secondary)" }} />;
}

function StatCard({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) {
  return (
    <div className="glass-card p-3 flex items-center gap-3">
      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
        style={{ background: "var(--accent-bg)" }}>
        {icon}
      </div>
      <div>
        <div className="text-lg font-bold tabular-nums" style={{ color: "var(--text)" }}>{value}</div>
        <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>{label}</div>
      </div>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3" style={{ color: "var(--text-tertiary)" }}>
      <Database className="w-8 h-8" />
      <p className="text-xs">{message}</p>
    </div>
  );
}

function useDeferredLoad(load: () => void) {
  useEffect(() => {
    const timer = window.setTimeout(() => load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);
}

/* ================================================================
   Tab: 概览 (Overview)
   ================================================================ */

function OverviewTab({ agentId }: { agentId: string }) {
  const [stats, setStats] = useState<MemStats | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.memStats(agentId);
      if (r.ok) setStats(r);
    } catch { /* ignore */ }
    setLoading(false);
  }, [agentId]);

  useDeferredLoad(load);

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} /></div>;
  if (!stats) return <EmptyState message="记忆系统尚未初始化" />;

  return (
    <div className="w-full min-w-0 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>数据概览</h3>
        <button onClick={load} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
      </div>

      <div className="grid w-full grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="记忆片段" value={stats.totalChunks} icon={<FileText className="w-4 h-4" style={{ color: "var(--accent)" }} />} />
        <StatCard label="任务" value={stats.totalTasks} icon={<ListTodo className="w-4 h-4" style={{ color: "var(--accent)" }} />} />
        <StatCard label="技能" value={stats.totalSkills} icon={<Sparkles className="w-4 h-4" style={{ color: "var(--accent)" }} />} />
        <StatCard label="会话" value={stats.totalSessions} icon={<MessageSquare className="w-4 h-4" style={{ color: "var(--accent)" }} />} />
      </div>

      {/* Role Breakdown */}
      <div className="glass-card p-3 space-y-2">
        <div className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>角色分布</div>
        <div className="flex gap-2 flex-wrap">
          {Object.entries(stats.roleBreakdown).map(([role, count]) => (
            <div key={role} className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px]"
              style={{ background: "var(--bg-inset)" }}>
              <RoleIcon role={role} />
              <span style={{ color: "var(--text)" }}>{role}</span>
              <span className="font-bold tabular-nums" style={{ color: "var(--accent)" }}>{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Dedup Breakdown */}
      <div className="glass-card p-3 space-y-2">
        <div className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>去重状态</div>
        <div className="flex gap-2 flex-wrap">
          {Object.entries(stats.dedupBreakdown).map(([status, count]) => (
            <div key={status} className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px]"
              style={{ background: "var(--bg-inset)" }}>
              <StatusBadge status={status} />
              <span className="font-bold tabular-nums" style={{ color: "var(--text)" }}>{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Time Range */}
      <div className="glass-card p-3 space-y-1">
        <div className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>时间范围</div>
        <div className="flex items-center gap-2 text-[11px]" style={{ color: "var(--text-secondary)" }}>
          <Clock className="w-3.5 h-3.5" />
          <span>{stats.timeRange.earliest ? formatTs(Number(stats.timeRange.earliest)) : "—"}</span>
          <span>→</span>
          <span>{stats.timeRange.latest ? formatTs(Number(stats.timeRange.latest)) : "—"}</span>
        </div>
      </div>

      {/* Task completion */}
      {stats.totalTasks > 0 && (
        <div className="glass-card p-3 space-y-2">
          <div className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>任务完成率</div>
          <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-inset)" }}>
            <div className="h-full rounded-full transition-all" style={{
              width: `${Math.round((stats.completedTasks / stats.totalTasks) * 100)}%`,
              background: "var(--accent)",
            }} />
          </div>
          <div className="text-[10px] tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {stats.completedTasks} / {stats.totalTasks} 已完成 ({Math.round((stats.completedTasks / stats.totalTasks) * 100)}%)
          </div>
        </div>
      )}
    </div>
  );
}

/* ================================================================
   Tab: 任务 (Tasks)
   ================================================================ */

function TasksTab({ agentId }: { agentId: string }) {
  const [tasks, setTasks] = useState<TaskItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [statusFilter, setStatusFilter] = useState("");
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const PAGE_SIZE = 20;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.memTasks(agentId, { status: statusFilter || undefined, limit: PAGE_SIZE, offset: page * PAGE_SIZE });
      if (r.ok) { setTasks(r.tasks); setTotal(r.total); }
    } catch { /* ignore */ }
    setLoading(false);
  }, [agentId, page, statusFilter]);

  useDeferredLoad(load);

  const openDetail = async (id: string) => {
    setDetailLoading(true);
    try {
      const r = await api.memTaskDetail(agentId, id);
      if (r.ok) setDetail(r);
    } catch { /* ignore */ }
    setDetailLoading(false);
  };

  if (detail) {
    return (
      <div className="w-full min-w-0 space-y-3">
        <button onClick={() => setDetail(null)} className="flex items-center gap-1 text-[11px] btn-ghost px-2 py-1" type="button">
          <ChevronLeft className="w-3.5 h-3.5" /> 返回列表
        </button>
        <div className="glass-card p-4 space-y-3">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>{detail.title || "未命名任务"}</h3>
              <p className="text-[10px] font-mono mt-0.5" style={{ color: "var(--text-tertiary)" }}>{detail.id}</p>
            </div>
            <StatusBadge status={detail.status} />
          </div>
          {detail.summary && (
            <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>{detail.summary}</p>
          )}
          <div className="flex gap-3 text-[10px]" style={{ color: "var(--text-tertiary)" }}>
            <span>开始: {formatTs(detail.startedAt)}</span>
            {detail.endedAt && <span>结束: {formatTs(detail.endedAt)}</span>}
          </div>
        </div>

        <div className="text-[11px] font-semibold px-1" style={{ color: "var(--text-secondary)" }}>
          关联片段 ({detail.chunks.length})
        </div>
        <div className="space-y-1.5 max-h-[60vh] overflow-y-auto">
          {detail.chunks.map((c) => (
            <div key={c.id} className="glass-card p-2.5 space-y-1">
              <div className="flex items-center gap-1.5">
                <RoleIcon role={c.role} />
                <span className="text-[10px] font-medium" style={{ color: "var(--text)" }}>{c.role}</span>
                <span className="text-[9px] ml-auto" style={{ color: "var(--text-tertiary)" }}>{formatTs(c.createdAt)}</span>
              </div>
              {c.summary && <p className="text-[10px]" style={{ color: "var(--accent)" }}>{c.summary}</p>}
              <p className="text-[10px] leading-relaxed whitespace-pre-wrap" style={{ color: "var(--text-secondary)" }}>{c.content}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="w-full min-w-0 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>任务列表</h3>
        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
            className="input text-[10px] py-0.5 px-1.5 w-24"
          >
            <option value="">全部状态</option>
            <option value="active">active</option>
            <option value="completed">completed</option>
          </select>
          <button onClick={load} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-16"><Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} /></div>
      ) : tasks.length === 0 ? (
        <EmptyState message="暂无任务" />
      ) : (
        <>
          <div className="space-y-1.5">
            {tasks.map((t) => (
              <button key={t.id} type="button" onClick={() => openDetail(t.id)}
                className="w-full text-left glass-card p-3 space-y-1.5 transition-all"
                onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")}
                onMouseLeave={e => (e.currentTarget.style.background = "")}>
                <div className="flex items-start justify-between gap-2">
                  <span className="text-[12px] font-medium truncate" style={{ color: "var(--text)" }}>{t.title || "未命名"}</span>
                  <StatusBadge status={t.status} />
                </div>
                {t.summary && <p className="text-[10px] line-clamp-2" style={{ color: "var(--text-secondary)" }}>{t.summary}</p>}
                <div className="flex items-center gap-3 text-[9px]" style={{ color: "var(--text-tertiary)" }}>
                  <span>{formatTs(t.startedAt)}</span>
                  <span>{t.chunkCount} 片段</span>
                </div>
              </button>
            ))}
          </div>

          {/* Pagination */}
          {total > PAGE_SIZE && (
            <div className="flex items-center justify-center gap-2 pt-2">
              <button disabled={page === 0} onClick={() => setPage(p => p - 1)} className="btn-ghost p-1.5 disabled:opacity-30" type="button">
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="text-[10px] tabular-nums" style={{ color: "var(--text-secondary)" }}>
                {page + 1} / {Math.ceil(total / PAGE_SIZE)}
              </span>
              <button disabled={(page + 1) * PAGE_SIZE >= total} onClick={() => setPage(p => p + 1)} className="btn-ghost p-1.5 disabled:opacity-30" type="button">
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </>
      )}

      {detailLoading && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center" style={{ background: "rgba(0,0,0,0.2)" }}>
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: "var(--accent)" }} />
        </div>
      )}
    </div>
  );
}

function BoundaryReviewsTab({ agentId }: { agentId: string }) {
  const [reviews, setReviews] = useState<BoundaryReviewItem[]>([]);
  const [tasks, setTasks] = useState<TaskItem[]>([]);
  const [targets, setTargets] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [resolving, setResolving] = useState<string | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [reviewResult, taskResult] = await Promise.all([
        api.memBoundaryReviews(agentId),
        api.memTasks(agentId, { limit: 200 }),
      ]);
      if (reviewResult.ok) setReviews(reviewResult.reviews);
      if (taskResult.ok) setTasks(taskResult.tasks);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [agentId]);

  useDeferredLoad(load);

  const resolve = async (
    review: BoundaryReviewItem,
    action: "assign_current" | "create_new" | "assign_other" | "orphan",
  ) => {
    const targetTaskId = targets[review.id];
    if (action === "assign_other" && !targetTaskId) {
      setError("请选择目标任务");
      return;
    }
    if (action === "orphan" && !window.confirm("确认将该轮次标记为无任务价值吗？")) return;

    setResolving(review.id);
    setError("");
    try {
      await api.resolveMemBoundaryReview(agentId, review.id, {
        action,
        target_task_id: action === "assign_other" ? targetTaskId : undefined,
      });
      await load();
    } catch (resolveError) {
      setError(resolveError instanceof Error ? resolveError.message : "处理失败");
    } finally {
      setResolving(null);
    }
  };

  if (loading) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} /></div>;
  }

  return (
    <div className="w-full min-w-0 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>待确认边界</h3>
          <span className="text-[10px] tabular-nums" style={{ color: "var(--text-tertiary)" }}>{reviews.length}</span>
        </div>
        <button onClick={load} className="btn-ghost p-1" type="button" title="刷新"><RefreshCw className="w-3.5 h-3.5" /></button>
      </div>

      {error && <div className="text-[11px] px-3 py-2 rounded" style={{ color: "var(--danger)", background: "color-mix(in srgb, var(--danger) 10%, transparent)" }}>{error}</div>}

      {reviews.length === 0 ? <EmptyState message="没有待确认的任务边界" /> : (
        <div className="space-y-3">
          {reviews.map((review) => {
            const busy = resolving === review.id;
            return (
              <div key={review.id} className="glass-card p-3 space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-[12px] font-medium truncate" style={{ color: "var(--text)" }}>{review.currentTaskTitle || "未命名任务"}</div>
                    <div className="text-[9px] font-mono mt-0.5" style={{ color: "var(--text-tertiary)" }}>{review.turnId}</div>
                  </div>
                  <span className="text-[10px] tabular-nums px-1.5 py-0.5 rounded" style={{ color: "var(--warning)", background: "color-mix(in srgb, var(--warning) 12%, transparent)" }}>
                    置信度 {Math.round(review.confidence * 100)}%
                  </span>
                </div>

                {review.currentTaskSummary && (
                  <div className="space-y-1">
                    <div className="text-[10px] font-medium" style={{ color: "var(--text-tertiary)" }}>当前任务</div>
                    <p className="text-[10px] leading-relaxed line-clamp-4" style={{ color: "var(--text-secondary)" }}>{review.currentTaskSummary}</p>
                  </div>
                )}

                {review.recentContext.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[10px] font-medium" style={{ color: "var(--text-tertiary)" }}>最近上下文</div>
                    <div className="space-y-1">
                      {review.recentContext.map((chunk) => (
                        <div key={chunk.id} className="flex gap-2 px-2 py-1.5 rounded" style={{ background: "var(--bg-inset)" }}>
                          <RoleIcon role={chunk.role} />
                          <p className="text-[10px] leading-relaxed line-clamp-3 min-w-0" style={{ color: "var(--text-secondary)" }}>{chunk.content}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="space-y-1">
                  <div className="text-[10px] font-medium" style={{ color: "var(--warning)" }}>待确认轮次</div>
                  {review.pendingChunks.map((chunk) => (
                    <div key={chunk.id} className="flex gap-2 px-2 py-1.5 rounded" style={{ background: "var(--bg-inset)" }}>
                      <RoleIcon role={chunk.role} />
                      <p className="text-[10px] leading-relaxed whitespace-pre-wrap min-w-0" style={{ color: "var(--text)" }}>{chunk.content}</p>
                    </div>
                  ))}
                </div>

                <div className="text-[10px]" style={{ color: "var(--text-tertiary)" }}>
                  {review.reason || "模型未给出有效判断"} · 已重试 {review.retryCount} 次 · {formatTs(review.createdAt)}
                </div>

                <div className="flex flex-wrap items-center gap-1.5 pt-1" style={{ borderTop: "1px solid var(--border)" }}>
                  <button disabled={busy} onClick={() => resolve(review, "assign_current")} className="btn-ghost px-2 py-1.5 text-[10px] flex items-center gap-1 disabled:opacity-50" type="button">
                    <Link className="w-3 h-3" />归入当前
                  </button>
                  <button disabled={busy} onClick={() => resolve(review, "create_new")} className="btn-ghost px-2 py-1.5 text-[10px] flex items-center gap-1 disabled:opacity-50" type="button">
                    <Plus className="w-3 h-3" />创建新任务
                  </button>
                  <div className="flex items-center gap-1 min-w-[190px] flex-1">
                    <select value={targets[review.id] || ""} onChange={(event) => setTargets(current => ({ ...current, [review.id]: event.target.value }))} className="input text-[10px] py-1 px-1.5 min-w-0 flex-1">
                      <option value="">选择其他任务</option>
                      {tasks.filter(task => task.id !== review.currentTaskId).map(task => <option key={task.id} value={task.id}>{task.title || task.id}</option>)}
                    </select>
                    <button disabled={busy || !targets[review.id]} onClick={() => resolve(review, "assign_other")} className="btn-ghost p-1.5 disabled:opacity-40" type="button" title="归入所选任务"><FolderInput className="w-3.5 h-3.5" /></button>
                  </div>
                  <button disabled={busy} onClick={() => resolve(review, "orphan")} className="btn-ghost p-1.5 disabled:opacity-50" type="button" title="标记为无任务价值">
                    {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ArchiveX className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ================================================================
   Tab: 技能 (Skills)
   ================================================================ */

function SkillsTab({ agentId }: { agentId: string }) {
  const [skills, setSkills] = useState<SkillItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.memSkills(agentId, { status: statusFilter || undefined });
      if (r.ok) setSkills(r.skills);
    } catch { /* ignore */ }
    setLoading(false);
  }, [agentId, statusFilter]);

  useDeferredLoad(load);

  return (
    <div className="w-full min-w-0 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>技能列表</h3>
        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="input text-[10px] py-0.5 px-1.5 w-24"
          >
            <option value="">全部</option>
            <option value="active">active</option>
            <option value="deprecated">deprecated</option>
          </select>
          <button onClick={load} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-16"><Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} /></div>
      ) : skills.length === 0 ? (
        <EmptyState message="暂无技能" />
      ) : (
        <div className="space-y-1.5">
          {skills.map((s) => (
            <div key={s.id} className="glass-card p-3 space-y-1.5">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--accent)" }} />
                  <span className="text-[12px] font-medium" style={{ color: "var(--text)" }}>{s.name}</span>
                  <span className="text-[9px] px-1 py-0.5 rounded" style={{ background: "var(--bg-inset)", color: "var(--text-tertiary)" }}>v{s.version}</span>
                </div>
                <StatusBadge status={s.status} />
              </div>
              {s.description && <p className="text-[10px] line-clamp-2" style={{ color: "var(--text-secondary)" }}>{s.description}</p>}
              <div className="flex items-center gap-3 text-[9px]" style={{ color: "var(--text-tertiary)" }}>
                {s.qualityScore != null && <span>质量: {(s.qualityScore * 100).toFixed(0)}%</span>}
                <span>更新: {formatTs(s.updatedAt)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ================================================================
   Tab: 记忆搜索 (Memory Search)
   ================================================================ */

function SearchTab({ agentId }: { agentId: string }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [memTotal, setMemTotal] = useState(0);
  const [memPage, setMemPage] = useState(1);
  const [memLoading, setMemLoading] = useState(true);
  const [roleFilter, setRoleFilter] = useState("");

  const loadMemories = useCallback(async () => {
    setMemLoading(true);
    try {
      const r = await api.memMemories(agentId, { page: memPage, limit: 30, role: roleFilter || undefined });
      if (r.ok) { setMemories(r.memories); setMemTotal(r.totalPages); }
    } catch { /* ignore */ }
    setMemLoading(false);
  }, [agentId, memPage, roleFilter]);

  useDeferredLoad(loadMemories);

  const doSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setSearched(true);
    try {
      const r = await api.memSearch(agentId, query.trim());
      if (r.ok) setResults(r.results || []);
    } catch { /* ignore */ }
    setLoading(false);
  };

  return (
    <div className="w-full min-w-0 space-y-4">
      {/* Search bar */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5" style={{ color: "var(--text-tertiary)" }} />
          <input
            className="input text-xs pl-8 pr-3 py-2 w-full"
            placeholder="搜索记忆片段..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doSearch()}
          />
        </div>
        <button onClick={doSearch} disabled={loading || !query.trim()} className="btn-ghost px-3 py-1.5 text-[11px] font-medium disabled:opacity-40" type="button"
          style={{ background: "var(--accent-bg)", color: "var(--accent)" }}>
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : "搜索"}
        </button>
      </div>

      {/* Search Results */}
      {searched && (
        <div className="space-y-2">
          <div className="text-[11px] font-semibold" style={{ color: "var(--text-secondary)" }}>
            搜索结果 ({results.length})
          </div>
          {results.length === 0 ? (
            <p className="text-[10px] py-4 text-center" style={{ color: "var(--text-tertiary)" }}>未找到相关记忆</p>
          ) : (
            <div className="space-y-1.5 max-h-[40vh] overflow-y-auto">
              {results.map((r) => (
                <div key={r.id} className="glass-card p-2.5 space-y-1">
                  <div className="flex items-center gap-1.5">
                    <RoleIcon role={r.role} />
                    <span className="text-[10px] font-medium" style={{ color: "var(--text)" }}>{r.role}</span>
                    <span className="text-[9px] px-1 rounded tabular-nums" style={{ background: "var(--accent-bg)", color: "var(--accent)" }}>
                      {r.score.toFixed(2)}
                    </span>
                    <span className="text-[9px] ml-auto" style={{ color: "var(--text-tertiary)" }}>{formatTs(r.createdAt)}</span>
                  </div>
                  {r.summary && <p className="text-[10px]" style={{ color: "var(--accent)" }}>{r.summary}</p>}
                  <p className="text-[10px] leading-relaxed line-clamp-3" style={{ color: "var(--text-secondary)" }}>{r.excerpt}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Browse all memories */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-[11px] font-semibold" style={{ color: "var(--text-secondary)" }}>全部记忆</div>
          <div className="flex items-center gap-2">
            <select value={roleFilter} onChange={(e) => { setRoleFilter(e.target.value); setMemPage(1); }}
              className="input text-[10px] py-0.5 px-1.5 w-24">
              <option value="">全部角色</option>
              <option value="user">user</option>
              <option value="assistant">assistant</option>
              <option value="tool">tool</option>
            </select>
            <button onClick={loadMemories} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
          </div>
        </div>

        {memLoading ? (
          <div className="flex items-center justify-center py-8"><Loader2 className="w-4 h-4 animate-spin" style={{ color: "var(--accent)" }} /></div>
        ) : memories.length === 0 ? (
          <EmptyState message="暂无记忆片段" />
        ) : (
          <>
            <div className="space-y-1 max-h-[40vh] overflow-y-auto">
              {memories.map((m) => (
                <div key={m.id} className="glass-card p-2.5 space-y-1">
                  <div className="flex items-center gap-1.5">
                    <RoleIcon role={m.role} />
                    <span className="text-[10px] font-medium" style={{ color: "var(--text)" }}>{m.role}</span>
                    {m.taskId && (
                      <span className="text-[9px] px-1 rounded truncate max-w-[100px]"
                        style={{ background: "var(--bg-inset)", color: "var(--text-tertiary)" }}
                        title={m.taskId}>
                        任务关联
                      </span>
                    )}
                    <span className="text-[9px] ml-auto flex-shrink-0" style={{ color: "var(--text-tertiary)" }}>{formatTs(m.createdAt)}</span>
                  </div>
                  {m.summary && <p className="text-[10px]" style={{ color: "var(--accent)" }}>{m.summary}</p>}
                  <p className="text-[10px] leading-relaxed line-clamp-2" style={{ color: "var(--text-secondary)" }}>{m.excerpt}</p>
                </div>
              ))}
            </div>

            {memTotal > 1 && (
              <div className="flex items-center justify-center gap-2 pt-1">
                <button disabled={memPage <= 1} onClick={() => setMemPage(p => p - 1)} className="btn-ghost p-1.5 disabled:opacity-30" type="button">
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <span className="text-[10px] tabular-nums" style={{ color: "var(--text-secondary)" }}>
                  {memPage} / {memTotal}
                </span>
                <button disabled={memPage >= memTotal} onClick={() => setMemPage(p => p + 1)} className="btn-ghost p-1.5 disabled:opacity-30" type="button">
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/* ================================================================
   Tab: Skill 进化 (human-controlled stages)
   ================================================================ */

const EVOLUTION_STAGES = [
  ["created", "待导出"], ["data_exported", "数据已导出"],
  ["dataset_confirmed", "数据集已确认"], ["candidate_generated", "Candidate 已生成"],
  ["candidate_confirmed", "Candidate 已确认"], ["dev_evaluated", "开发集已评估"],
  ["revision_pending", "等待修改"], ["regression_verified", "回归已验证"],
  ["holdout_verified", "独立集已验证"], ["approved", "已批准"],
  ["published", "已发布"], ["abandoned", "已放弃"],
] as const;

function EvolutionTab({ agentId }: { agentId: string }) {
  const [items, setItems] = useState<any[]>([]);
  const [family, setFamily] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [traceRows, setTraceRows] = useState<any[]>([]);
  const [traceTaskId, setTraceTaskId] = useState<string | null>(null);
  const [message, setMessage] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await api.memEvolutionTasks(agentId);
      if (result.ok) setItems(result.tasks || []);
    } catch { /* keep the last visible state */ }
    setLoading(false);
  }, [agentId]);

  useDeferredLoad(load);

  const create = async () => {
    const value = family.trim();
    if (!value) return;
    setBusy(true);
    try {
      await api.createMemEvolutionTask(agentId, value);
      setFamily("");
      await load();
    } finally { setBusy(false); }
  };

  const advance = async (item: any, stage: string) => {
    setBusy(true);
    try {
      await api.updateMemEvolutionStage(agentId, item.id, stage);
      await load();
    } finally { setBusy(false); }
  };

  const advanceStage = async (item: any, stage: string) => {
    if (item.stage === "created" && stage === "data_exported") {
      setBusy(true);
      try { await api.exportMemEvolutionTask(agentId, item.id, selected); setSelected([]); setTraceTaskId(null); setMessage(`已导出 ${selected.length} 条轨迹`); await load(); }
      finally { setBusy(false); }
      return;
    }
    await advance(item, stage);
  };

  const openTracePicker = async (item: any) => {
    setBusy(true); setMessage("");
    try {
      const result = await api.memEvolutionTraces(agentId, item.id);
      setTraceRows(result.traces || []); setTraceTaskId(item.id); setSelected([]);
    } catch (error) { setMessage(error instanceof Error ? error.message : "加载轨迹失败"); }
    finally { setBusy(false); }
  };

  return (
    <div className="w-full min-w-0 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>Skill 进化</h3>
          <p className="text-[10px] mt-1" style={{ color: "var(--text-tertiary)" }}>每个阶段单独确认，不会自动跑完整流程</p>
        </div>
        <button onClick={load} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
      </div>
      <div className="flex gap-2">
        <input className="input text-xs flex-1" value={family} onChange={(e) => setFamily(e.target.value)} placeholder="输入任务族名称" />
        <button className="btn-ghost px-2" type="button" onClick={create} disabled={busy || !family.trim()}><Plus className="w-3.5 h-3.5" /> 新建</button>
      </div>
      {message && <div className="text-[10px] px-2 py-1 rounded" style={{ background: "var(--accent-bg)", color: "var(--accent)" }}>{message}</div>}
      {loading ? <div className="flex justify-center py-12"><Loader2 className="w-4 h-4 animate-spin" style={{ color: "var(--accent)" }} /></div> : items.length === 0 ? <EmptyState message="暂无进化任务" /> : (
        <div className="space-y-2">
          {items.map((item) => {
            const currentIndex = EVOLUTION_STAGES.findIndex(([key]) => key === item.stage);
            const next = EVOLUTION_STAGES[currentIndex + 1];
            return (
              <div key={item.id} className="glass-card p-3 space-y-2">
                <div className="flex items-center gap-2">
                  <GitBranch className="w-3.5 h-3.5" style={{ color: "var(--accent)" }} />
                  <span className="text-xs font-medium" style={{ color: "var(--text)" }}>{item.title}</span>
                  <span className="ml-auto text-[10px]" style={{ color: "var(--accent)" }}>{EVOLUTION_STAGES[currentIndex]?.[1] || item.stage}</span>
                </div>
                <div className="flex items-center gap-1 overflow-x-auto">
                  {EVOLUTION_STAGES.slice(0, -1).map(([key, label], index) => <span key={key} className="text-[9px] whitespace-nowrap" style={{ color: index <= currentIndex ? "var(--accent)" : "var(--text-tertiary)" }}>{index > 0 && " → "}{label}</span>)}
                </div>
                {item.stage === "created" && <button className="btn-ghost text-[10px] px-2 py-1" type="button" disabled={busy} onClick={() => openTracePicker(item)}>选择轨迹</button>}
                {item.stage !== "created" && item.stage !== "abandoned" && <span className="text-[10px]" style={{ color: "var(--text-tertiary)" }}>该阶段执行器尚未接入</span>}
                {traceTaskId === item.id && item.stage === "created" && (
                  <div className="border-t pt-2 space-y-2" style={{ borderColor: "var(--border)" }}>
                    <div className="flex items-center justify-between"><span className="text-[10px] font-medium" style={{ color: "var(--text-secondary)" }}>选择要导出的轨迹</span><span className="text-[10px]" style={{ color: "var(--accent)" }}>已选 {selected.length} 条</span></div>
                    <div className="max-h-48 overflow-y-auto space-y-1">
                      {traceRows.length === 0 ? <div className="text-[10px] py-3" style={{ color: "var(--text-tertiary)" }}>暂无可选轨迹</div> : traceRows.map((row) => <label key={row.trace_id} className="flex gap-2 p-2 rounded cursor-pointer" style={{ background: "var(--bg-inset)" }}><input type="checkbox" checked={selected.includes(row.trace_id)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, row.trace_id] : current.filter((id) => id !== row.trace_id))} /><span className="min-w-0 text-[10px]" style={{ color: "var(--text-secondary)" }}><b style={{ color: "var(--text)" }}>{row.title || row.trace_id}</b><br />{row.input_preview}<br /><span style={{ color: "var(--text-tertiary)" }}>{row.category || "未分类"} · {row.created_at || "无时间"}</span></span></label>)}
                    </div>
                    <button className="btn-ghost text-[10px] px-2 py-1" type="button" disabled={busy || selected.length === 0} onClick={() => advanceStage(item, "data_exported")}>确认导出并进入：数据已导出</button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ================================================================
   Main Modal
   ================================================================ */

const TABS = [
  { key: "overview", label: "概览", icon: BarChart3 },
  { key: "tasks", label: "任务", icon: ListTodo },
  { key: "reviews", label: "待确认", icon: CircleHelp },
  { key: "skills", label: "技能", icon: Sparkles },
  { key: "evolution", label: "进化", icon: GitBranch },
  { key: "search", label: "记忆", icon: Search },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export default function MemoryModal() {
  const { currentAgentId } = useAgentContext();
  const { showMemoryModal, setShowMemoryModal } = useUi();
  const [tab, setTab] = useState<TabKey>("overview");

  useEffect(() => {
    if (!showMemoryModal) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setShowMemoryModal(false); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [showMemoryModal, setShowMemoryModal]);

  if (!showMemoryModal) return null;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-[60] transition-opacity"
        style={{ background: "rgba(0,0,0,0.30)" }}
        onClick={() => setShowMemoryModal(false)} aria-hidden />

      {/* Modal */}
      <div className="fixed inset-4 sm:inset-8 z-[61] flex w-auto min-w-0 flex-col rounded-2xl overflow-hidden animate-scale-in"
        style={{
          background: "var(--bg-elevated)",
          border: "1px solid var(--border)",
          boxShadow: "var(--shadow-xl)",
        }}>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 flex-shrink-0"
          style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2.5">
            <BrainCircuit className="w-4.5 h-4.5" style={{ color: "var(--accent)" }} />
            <h2 className="text-sm font-semibold" style={{ color: "var(--text)" }}>记忆看板</h2>
          </div>
          <button onClick={() => setShowMemoryModal(false)} className="btn-ghost p-1.5" type="button">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tab Bar */}
        <div className="flex w-full min-w-0 items-center px-5 gap-1 flex-shrink-0 overflow-x-auto"
          style={{ borderBottom: "1px solid var(--border)" }}>
          {TABS.map((t) => (
            <button key={t.key} type="button" onClick={() => setTab(t.key)}
              className="flex items-center gap-1.5 px-3 py-2.5 text-[11px] font-medium transition-all relative"
              style={{ color: tab === t.key ? "var(--accent)" : "var(--text-secondary)" }}>
              <t.icon className="w-3.5 h-3.5" />
              <span>{t.label}</span>
              {tab === t.key && (
                <div className="absolute bottom-0 left-1 right-1 h-[2px] rounded-full" style={{ background: "var(--accent)" }} />
              )}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className={`flex w-full min-w-0 flex-1 overflow-y-auto ${tab === "evolution" ? "p-0" : "p-5"}`}>
          {tab === "overview" && <OverviewTab key={currentAgentId} agentId={currentAgentId} />}
          {tab === "tasks" && <TasksTab key={currentAgentId} agentId={currentAgentId} />}
          {tab === "reviews" && <BoundaryReviewsTab key={currentAgentId} agentId={currentAgentId} />}
          {tab === "skills" && <SkillsTab key={currentAgentId} agentId={currentAgentId} />}
          {tab === "evolution" && <SkillEvolutionWorkspace key={currentAgentId} agentId={currentAgentId} />}
          {tab === "search" && <SearchTab key={currentAgentId} agentId={currentAgentId} />}
        </div>
      </div>
    </>
  );
}
