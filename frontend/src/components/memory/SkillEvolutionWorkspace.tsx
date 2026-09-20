"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle, Check, CheckCircle2, ChevronLeft, ChevronRight,
  Database, Eye, FileCheck2, FileText, GitBranch, Loader2, Plus,
  RefreshCw, Search, ShieldCheck, Sparkles, TestTube2, XCircle,
} from "lucide-react";
import * as api from "@/lib/api";

type Execution = {
  operation: string | null;
  status: "idle" | "running" | "succeeded" | "failed" | "cancelled";
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
};

type EvolutionTask = {
  id: string;
  title: string;
  task_family: string;
  data_source?: "langfuse" | "skilllearnbench";
  stage: string;
  artifacts: Record<string, string>;
  execution?: Execution;
  created_at: string;
  updated_at: string;
};

type TraceRow = {
  trace_id: string;
  title: string;
  category: string | null;
  session_id: string;
  created_at: string | null;
  input_preview: string;
  output_preview: string;
  skill_names: string[];
};

const PHASES = [
  { key: "data", label: "数据选择", icon: Database },
  { key: "dataset", label: "数据集整理", icon: FileText },
  { key: "candidate", label: "候选生成", icon: Sparkles },
  { key: "dev", label: "开发评估", icon: TestTube2 },
  { key: "revision", label: "局部优化", icon: GitBranch },
  { key: "regression", label: "回归验证", icon: FileCheck2 },
  { key: "release", label: "验收发布", icon: ShieldCheck },
] as const;

const STAGE_LABELS: Record<string, string> = {
  created: "等待选择数据", data_exported: "轨迹已导出",
  dataset_confirmed: "数据集已确认", candidate_generated: "候选已生成",
  candidate_confirmed: "候选已确认", dev_evaluated: "开发评估完成",
  revision_pending: "等待局部优化", regression_verified: "回归验证完成",
  regression_pending: "等待回归验证", holdout_pending: "等待独立验收",
  holdout_verified: "独立验收完成", approved: "等待发布",
  published: "已发布", abandoned: "已放弃",
};

function phaseIndex(stage: string): number {
  if (stage === "created") return 0;
  if (stage === "data_exported") return 1;
  if (stage === "dataset_confirmed" || stage === "candidate_generated") return 2;
  if (stage === "candidate_confirmed" || stage === "dev_evaluated") return 3;
  if (stage === "revision_pending") return 4;
  if (stage === "regression_pending" || stage === "regression_verified") return 5;
  return 6;
}

function timeLabel(value?: string | null) {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function elapsedLabel(startedAt?: string | null, now = Date.now()) {
  if (!startedAt) return "0 秒";
  const seconds = Math.max(0, Math.floor((now - new Date(startedAt).getTime()) / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} 分 ${seconds % 60} 秒`;
}

function ExecutionResult({ execution }: { execution?: Execution }) {
  if (!execution || execution.status === "idle") return null;
  const failed = execution.status === "failed";
  const cancelled = execution.status === "cancelled";
  const interrupted = failed || cancelled;
  const running = execution.status === "running";
  return (
    <div className="flex items-start gap-2 border px-3 py-2 text-[11px]" style={{ borderColor: interrupted ? "var(--error)" : "var(--border)", background: interrupted ? "var(--error-bg)" : running ? "var(--accent-bg)" : "var(--success-bg)" }}>
      {running ? <Loader2 className="mt-0.5 h-4 w-4 animate-spin" style={{ color: "var(--accent)" }} /> : interrupted ? <XCircle className="mt-0.5 h-4 w-4" style={{ color: "var(--error)" }} /> : <CheckCircle2 className="mt-0.5 h-4 w-4" style={{ color: "var(--success)" }} />}
      <div className="min-w-0">
        <div className="font-medium" style={{ color: interrupted ? "var(--error)" : "var(--text)" }}>{running ? "正在执行" : failed ? "执行失败" : cancelled ? "评估已取消" : "执行成功"}</div>
        <div className="mt-0.5 break-words" style={{ color: "var(--text-secondary)" }}>{interrupted ? execution.error : execution.result ? JSON.stringify(execution.result, null, 2) : "操作已完成"}</div>
      </div>
    </div>
  );
}

function ExecutionMonitor({ task, agentId, onChanged }: { task: EvolutionTask; agentId: string; onChanged: () => Promise<void> }) {
  const [data, setData] = useState<any>(null); const [error, setError] = useState(""); const [cancelling, setCancelling] = useState(false); const [now, setNow] = useState(Date.now());
  const load = useCallback(async () => { try { const next = await api.memEvolutionExecution(agentId, task.id); setData(next); if (task.execution?.status === "running" && next.execution?.status !== "running") await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "加载执行状态失败"); } }, [agentId, task.id, task.execution?.status, onChanged]);
  useEffect(() => { void load(); if (task.execution?.status !== "running") return; const timer = window.setInterval(() => void load(), 1500); return () => window.clearInterval(timer); }, [load, task.execution?.status]);
  useEffect(() => { if (task.execution?.status !== "running") return; const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(timer); }, [task.execution?.status]);
  if (task.execution?.status !== "running" && !data?.logs?.length) return <ExecutionResult execution={task.execution} />;
  const cancel = async () => { setCancelling(true); setError(""); try { await api.cancelMemEvolutionExecution(agentId, task.id); await load(); } catch (e) { setError(e instanceof Error ? e.message : "取消失败"); } finally { setCancelling(false); } };
  return <div className="space-y-2 border p-3" style={{ borderColor: "var(--border)", background: "var(--bg-inset)" }}><div className="flex items-center justify-between gap-3"><div><div className="text-[11px] font-medium" style={{ color: "var(--text)" }}>{data?.phase || "正在准备评估"}</div><div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[10px]" style={{ color: "var(--text-tertiary)" }}><span>已运行 {elapsedLabel(task.execution?.started_at, now)}</span><span>日志更新 {data?.log_updated_at ? timeLabel(data.log_updated_at) : "等待中"}</span><span>{data?.process_alive === false && task.execution?.status === "running" ? "进程状态待确认" : "后台进程运行中"}</span></div></div>{task.execution?.status === "running" && <button className="btn-ghost flex-shrink-0 px-2 py-1 text-[10px]" style={{ color: "var(--error)" }} disabled={cancelling} onClick={cancel} type="button">{cancelling ? "取消中..." : "取消评估"}</button>}</div>{error && <div className="text-[10px]" style={{ color: "var(--error)" }}>{error}</div>}<pre className="max-h-48 overflow-auto whitespace-pre-wrap border p-2 text-[10px]" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>{(data?.logs || []).join("\n") || "等待日志输出..."}</pre></div>;
}

function ArtifactResult({ task }: { task: EvolutionTask }) {
  const entries = Object.entries(task.artifacts || {});
  if (!entries.length) return <div className="py-10 text-center text-[11px]" style={{ color: "var(--text-tertiary)" }}>该阶段还没有产物</div>;
  return (
    <div className="divide-y border" style={{ borderColor: "var(--border)" }}>
      {entries.map(([key, value]) => <div key={key} className="grid grid-cols-[130px_minmax(0,1fr)] gap-3 px-3 py-2 text-[11px]" style={{ borderColor: "var(--border)" }}><span style={{ color: "var(--text-tertiary)" }}>{key}</span><span className="break-all" style={{ color: "var(--text)" }}>{value}</span></div>)}
    </div>
  );
}

function ValidationStage({ task, agentId, split, onChanged }: { task: EvolutionTask; agentId: string; split: "regression" | "holdout"; onChanged: () => Promise<void> }) {
  const [report, setReport] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const pendingStage = split === "regression" ? "regression_pending" : "holdout_pending";
  const title = split === "regression" ? "历史失败样本回归验证" : "独立留出集验收";
  const load = useCallback(async () => {
    try { const result = await api.memEvolutionValidation(agentId, task.id, split); setReport(result.report); }
    catch (e) { setError(e instanceof Error ? e.message : "加载验证报告失败"); }
  }, [agentId, task.id, split]);
  useEffect(() => { void load(); }, [load]);
  const run = async () => {
    setBusy(true); setError(""); setMessage("");
    try { const result = await api.runMemEvolutionValidation(agentId, task.id, split); if (result.report) setReport(result.report); setMessage(result.report?.status === "not_applicable" ? "本轮没有历史失败样本，可直接确认" : "验证已在后台启动"); await onChanged(); }
    catch (e) { setError(e instanceof Error ? e.message : "验证执行失败"); }
    finally { setBusy(false); }
  };
  const confirm = async () => {
    setBusy(true); setError("");
    try { await api.confirmMemEvolutionValidation(agentId, task.id, split); setMessage(split === "regression" ? "回归结果已确认，进入独立验收" : "独立验收已确认"); await onChanged(); }
    catch (e) { setError(e instanceof Error ? e.message : "确认失败"); }
    finally { setBusy(false); }
  };
  const label = (system: string) => system === "without_skill" ? "无 Skill" : system === "active_skill" ? "当前 Active Skill" : system === "human_authored" ? "官方人工 Skill" : "Candidate";
  return <div className="flex min-h-[480px] min-w-0 flex-col gap-3"><div className="border-b pb-3"><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>{title}</h3><p className="mt-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>{split === "regression" ? "只重跑开发阶段沉淀的历史失败样本；没有回归样本时会明确跳过。" : "使用从未参与生成和修改的 Holdout 样本，并加入官方人工 Skill 作为最终参考基线。"}</p></div>{error && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}>{error}</div>}{message && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--success)", color: "var(--success)", background: "var(--success-bg)" }}>{message}</div>}{report?.status === "not_applicable" ? <div className="border px-3 py-4 text-[11px]" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>{report.reason}</div> : <><div className="grid gap-2 md:grid-cols-4">{(report?.systems || []).map((row: any) => <div key={row.system} className="border px-3 py-3" style={{ borderColor: "var(--border)" }}><div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>{label(row.system)}</div><div className="mt-1 text-xl font-semibold" style={{ color: "var(--text)" }}>{Math.round((row.pass_rate || 0) * 100)}%</div><div className="mt-1 text-[10px]" style={{ color: "var(--text-tertiary)" }}>通过 {row.passed}/{row.evaluated_cases} · 平均 {Math.round(row.avg_tokens || 0)} tokens</div></div>)}</div><div className="min-h-0 flex-1 overflow-auto border" style={{ borderColor: "var(--border)" }}>{report?.cases?.length ? <table className="w-full text-left text-[10px]"><thead className="sticky top-0" style={{ background: "var(--bg-inset)", color: "var(--text-tertiary)" }}><tr><th className="px-3 py-2">样本</th><th className="px-3 py-2">版本</th><th className="px-3 py-2">结果</th><th className="px-3 py-2">错误</th></tr></thead><tbody>{report.cases.map((row: any, index: number) => <tr key={`${row.sample_id}-${row.variant}-${index}`} className="border-t" style={{ borderColor: "var(--border)" }}><td className="break-all px-3 py-2">{row.sample_id}</td><td className="px-3 py-2">{label(row.variant)}</td><td className="px-3 py-2" style={{ color: row.external_failure ? "var(--error)" : row.passed ? "var(--success)" : "var(--text-secondary)" }}>{row.external_failure ? "外部失败" : row.passed ? "通过" : "失败"}</td><td className="break-all px-3 py-2" style={{ color: "var(--text-tertiary)" }}>{row.error || "-"}</td></tr>)}</tbody></table> : <div className="py-16 text-center text-[11px]" style={{ color: "var(--text-tertiary)" }}>尚未生成验证报告</div>}</div></>}<ExecutionMonitor task={task} agentId={agentId} onChanged={onChanged} /><div className="flex justify-end gap-2 border-t pt-3" style={{ borderColor: "var(--border)" }}><button type="button" className="btn-ghost px-3 py-1.5 text-[11px]" disabled={busy || task.stage !== pendingStage || task.execution?.status === "running"} onClick={run}>{busy ? "启动中..." : "开始验证"}</button><button type="button" className="btn-primary px-3 py-1.5 text-[11px]" disabled={busy || task.stage !== pendingStage || task.execution?.status === "running" || !report} onClick={confirm}>确认验证结果</button></div></div>;
}

function ReleaseStage({ task, agentId, onChanged }: { task: EvolutionTask; agentId: string; onChanged: () => Promise<void> }) {
  const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const [message, setMessage] = useState("");
  const approve = async () => { setBusy(true); setError(""); try { await api.approveMemEvolutionTask(agentId, task.id); setMessage("人工审核已通过，可以发布"); await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "审核失败"); } finally { setBusy(false); } };
  const publish = async () => { setBusy(true); setError(""); try { await api.publishMemEvolutionTask(agentId, task.id); setMessage("Candidate 已发布为 Active Skill"); await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "发布失败"); } finally { setBusy(false); } };
  return <div className="space-y-4"><div className="border-b pb-3"><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>发布门禁与人工审核</h3><p className="mt-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>留出集报告通过后，先人工审核，再将 Candidate 发布为 Active Skill。</p></div>{error && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}>{error}</div>}{message && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--success)", color: "var(--success)", background: "var(--success-bg)" }}>{message}</div>}<ArtifactResult task={task} /><div className="flex justify-end gap-2 border-t pt-3" style={{ borderColor: "var(--border)" }}><button className="btn-ghost px-3 py-1.5 text-[11px]" disabled={busy || task.stage !== "holdout_verified"} onClick={approve} type="button">人工审核通过</button><button className="btn-primary px-3 py-1.5 text-[11px]" disabled={busy || task.stage !== "approved"} onClick={publish} type="button">发布为 Active Skill</button></div></div>;
}

function DataStage({ task, agentId, onChanged }: { task: EvolutionTask; agentId: string; onChanged: () => Promise<void> }) {
  const [rows, setRows] = useState<TraceRow[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [preview, setPreview] = useState<TraceRow | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [name, setName] = useState("");
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [families, setFamilies] = useState<any[]>([]);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);

  const load = useCallback(async (targetPage = 1) => {
    setLoading(true); setError("");
    try {
      const result = await api.memEvolutionTraces(agentId, task.id, { page: targetPage, limit: 50, sessionId, name });
      setRows(result.traces || []); setHasMore(Boolean(result.has_more)); setPage(targetPage);
    } catch (loadError) { setError(loadError instanceof Error ? loadError.message : "加载轨迹失败"); }
    finally { setLoading(false); }
  }, [agentId, task.id, sessionId, name]);

  useEffect(() => { if (task.stage === "created" && task.data_source !== "skilllearnbench") void load(1); }, [task.id, task.data_source]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (task.data_source === "skilllearnbench") {
      void api.memEvolutionSkillLearnBenchFamilies().then((result) => setFamilies(result.families || [])).catch((e) => setError(e instanceof Error ? e.message : "加载标准任务集失败"));
    }
  }, [task.data_source]);

  const prepareBenchmark = async () => {
    setBenchmarkBusy(true); setError("");
    try { await api.prepareMemEvolutionSkillLearnBench(agentId, task.id); await onChanged(); }
    catch (e) { setError(e instanceof Error ? e.message : "准备标准任务集失败"); }
    finally { setBenchmarkBusy(false); }
  };

  const exportSelected = async () => {
    setLoading(true); setError("");
    try { await api.exportMemEvolutionTask(agentId, task.id, [...selected]); await onChanged(); }
    catch (exportError) { setError(exportError instanceof Error ? exportError.message : "导出失败"); }
    finally { setLoading(false); }
  };

  if (task.stage !== "created") return <div className="space-y-4"><div><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>数据准备结果</h3><p className="mt-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>{task.data_source === "skilllearnbench" ? "已准备 SkillLearnBench 标准任务实例和种子执行轨迹。" : "只有人工勾选的 Langfuse 轨迹被写入本次进化任务。"}</p></div><ExecutionResult execution={task.execution} /><ArtifactResult task={task} /></div>;

  if (task.data_source === "skilllearnbench") {
    const family = families.find((row) => row.family === task.task_family);
    return <div className="space-y-4"><div className="border-b pb-3"><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>SkillLearnBench 数据准备</h3><p className="mt-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>读取第一个实例的公开任务材料生成 Candidate，再将同任务族实例划分为 Dev 和 Holdout；不会把 verifier 或 solution 提供给生成器。</p></div>{error && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}>{error}</div>}<div className="border" style={{ borderColor: "var(--border)" }}><div className="border-b px-3 py-2 text-[11px] font-medium" style={{ borderColor: "var(--border)" }}>{task.task_family} · 版本 {family?.revision || "-"}</div><div className="divide-y" style={{ borderColor: "var(--border)" }}>{(family?.instances || []).map((row: any) => <div key={row.instance_id} className="grid gap-2 px-3 py-3 sm:grid-cols-[220px_90px_minmax(0,1fr)]"><span className="text-[11px] font-medium" style={{ color: "var(--text)" }}>{row.instance_id}</span><span className="text-[10px]" style={{ color: "var(--accent)" }}>{row.role === "seed" ? "生成实例" : "评估实例"}</span><span className="truncate text-[10px]" style={{ color: "var(--text-secondary)" }}>{row.instruction_preview || "无任务说明"}</span></div>)}</div></div><ExecutionResult execution={task.execution} /><div className="flex justify-end border-t pt-3" style={{ borderColor: "var(--border)" }}><button type="button" className="btn-primary flex items-center gap-1.5 px-3 py-1.5 text-[11px]" disabled={benchmarkBusy} onClick={prepareBenchmark}>{benchmarkBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}准备任务族数据</button></div></div>;
  }

  const allChecked = rows.length > 0 && rows.every((row) => selected.has(row.trace_id));
  return (
    <div className="flex h-full min-h-[480px] min-w-0 flex-col">
      <div className="flex flex-wrap items-end gap-2 border-b pb-3" style={{ borderColor: "var(--border)" }}>
        <label className="min-w-[180px] flex-1 text-[10px]" style={{ color: "var(--text-secondary)" }}>会话 ID<input className="input mt-1 w-full text-[11px]" value={sessionId} onChange={(event) => setSessionId(event.target.value)} placeholder="可选" /></label>
        <label className="min-w-[180px] flex-1 text-[10px]" style={{ color: "var(--text-secondary)" }}>轨迹名称<input className="input mt-1 w-full text-[11px]" value={name} onChange={(event) => setName(event.target.value)} placeholder="可选" /></label>
        <button className="btn-ghost flex items-center gap-1 px-3 py-1.5 text-[11px]" onClick={() => load(1)} disabled={loading} type="button"><Search className="h-3.5 w-3.5" />查询</button>
      </div>
      {error && <div className="mt-3 flex items-center gap-2 border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}><AlertCircle className="h-4 w-4" />{error}</div>}
      <div className="mt-3 grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0 overflow-hidden border" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--border)", background: "var(--bg-inset)" }}>
            <label className="flex items-center gap-2 text-[11px]" style={{ color: "var(--text)" }}><input type="checkbox" checked={allChecked} onChange={(event) => setSelected((current) => { const next = new Set(current); rows.forEach((row) => event.target.checked ? next.add(row.trace_id) : next.delete(row.trace_id)); return next; })} />全选当前页</label>
            <span className="text-[10px]" style={{ color: "var(--accent)" }}>已选 {selected.size} 条</span>
          </div>
          <div className="max-h-[360px] overflow-auto">
            {loading && !rows.length ? <div className="flex justify-center py-16"><Loader2 className="h-5 w-5 animate-spin" style={{ color: "var(--accent)" }} /></div> : rows.length === 0 ? <div className="py-16 text-center text-[11px]" style={{ color: "var(--text-tertiary)" }}>没有符合条件的轨迹</div> : (
              <table className="w-full table-fixed text-left text-[10px]"><thead className="sticky top-0" style={{ background: "var(--bg-elevated)", color: "var(--text-tertiary)" }}><tr><th className="w-9 px-2 py-2"></th><th className="px-2 py-2">任务内容</th><th className="w-24 px-2 py-2">状态</th><th className="w-28 px-2 py-2">时间</th><th className="w-10 px-2 py-2"></th></tr></thead><tbody>{rows.map((row) => <tr key={row.trace_id} className="border-t" style={{ borderColor: "var(--border)" }}><td className="px-2 py-2"><input type="checkbox" checked={selected.has(row.trace_id)} onChange={(event) => setSelected((current) => { const next = new Set(current); event.target.checked ? next.add(row.trace_id) : next.delete(row.trace_id); return next; })} /></td><td className="px-2 py-2"><div className="truncate font-medium" style={{ color: "var(--text)" }}>{row.title || row.input_preview || row.trace_id}</div><div className="mt-0.5 truncate" style={{ color: "var(--text-tertiary)" }}>{row.session_id || row.trace_id}</div></td><td className="px-2 py-2" style={{ color: row.category === "success" ? "var(--success)" : "var(--text-secondary)" }}>{row.category || "未分类"}</td><td className="px-2 py-2" style={{ color: "var(--text-secondary)" }}>{timeLabel(row.created_at)}</td><td className="px-2 py-2"><button className="btn-ghost p-1" onClick={() => setPreview(row)} title="查看轨迹" type="button"><Eye className="h-3.5 w-3.5" /></button></td></tr>)}</tbody></table>
            )}
          </div>
          <div className="flex items-center justify-center gap-2 border-t px-3 py-2" style={{ borderColor: "var(--border)" }}><button className="btn-ghost p-1" disabled={page <= 1 || loading} onClick={() => load(page - 1)} type="button"><ChevronLeft className="h-3.5 w-3.5" /></button><span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>第 {page} 页</span><button className="btn-ghost p-1" disabled={!hasMore || loading} onClick={() => load(page + 1)} type="button"><ChevronRight className="h-3.5 w-3.5" /></button></div>
        </div>
        <div className="min-w-0 border p-3" style={{ borderColor: "var(--border)" }}>
          <div className="mb-3 text-[11px] font-medium" style={{ color: "var(--text)" }}>轨迹预览</div>
          {!preview ? <div className="py-16 text-center text-[10px]" style={{ color: "var(--text-tertiary)" }}>点击查看按钮检查输入和输出</div> : <div className="space-y-3 text-[10px]"><div><div className="mb-1" style={{ color: "var(--text-tertiary)" }}>轨迹 ID</div><div className="break-all" style={{ color: "var(--text)" }}>{preview.trace_id}</div></div><div><div className="mb-1" style={{ color: "var(--text-tertiary)" }}>输入</div><pre className="max-h-32 overflow-auto whitespace-pre-wrap border p-2" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>{preview.input_preview}</pre></div><div><div className="mb-1" style={{ color: "var(--text-tertiary)" }}>输出</div><pre className="max-h-32 overflow-auto whitespace-pre-wrap border p-2" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>{preview.output_preview}</pre></div></div>}
        </div>
      </div>
      <div className="mt-3 flex items-center justify-between border-t pt-3" style={{ borderColor: "var(--border)" }}><span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>导出前请逐项确认内容属于同一任务族</span><button className="btn-primary flex items-center gap-1.5 px-3 py-1.5 text-[11px]" disabled={!selected.size || loading} onClick={exportSelected} type="button">{loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}确认导出 {selected.size} 条</button></div>
    </div>
  );
}

function DatasetStage({ task, agentId, onChanged }: { task: EvolutionTask; agentId: string; onChanged: () => Promise<void> }) {
  const [draft, setDraft] = useState<any>(null);
  const [activeSplit, setActiveSplit] = useState("seed");
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    try { const result = await api.memEvolutionDataset(agentId, task.id); setDraft({ ...result.draft, validation: result.validation || [] }); setError(""); }
    catch (loadError) { setError(loadError instanceof Error ? loadError.message : "加载数据集失败"); }
  }, [agentId, task.id]);
  useEffect(() => { void load(); }, [load]);
  if (task.stage === "dataset_confirmed") return <div className="space-y-4"><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>数据集已冻结</h3><p className="text-[11px]" style={{ color: "var(--text-secondary)" }}>当前版本为 v1，后续评估只读取该版本。</p><ArtifactResult task={task} /></div>;
  const assignments = draft?.assignments || { seed: [], dev: [], holdout: [], excluded: [] };
  const samples = draft?.samples || {};
  const rows = Object.values(samples).filter((row: any) => (assignments[activeSplit] || []).includes(row.sample_id)) as any[];
  const move = (sampleId: string, split: string) => setDraft((current: any) => ({ ...current, assignments: Object.fromEntries(Object.entries(current.assignments).map(([key, ids]: [string, any]) => [key, key === split ? [...ids.filter((id: string) => id !== sampleId), sampleId] : ids.filter((id: string) => id !== sampleId)])) }));
  const save = async (confirm = false) => {
    setBusy(true); setMessage(""); setError("");
    try { await api.saveMemEvolutionDataset(agentId, task.id, draft.assignments); if (confirm) { await api.confirmMemEvolutionDataset(agentId, task.id); await onChanged(); setMessage("数据集已确认并冻结为 v1"); } else setMessage("草稿已保存"); }
    catch (saveError) { setError(saveError instanceof Error ? saveError.message : "保存失败"); }
    finally { setBusy(false); }
  };
  return <div className="flex min-h-[480px] min-w-0 flex-col gap-3">
    <div className="border-b pb-3"><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>数据集整理</h3><p className="mt-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>系统已给出初始建议，请逐条确认样本归属。Seed 只能放成功且有完整轨迹的样本。</p></div>
    {error && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}>{error}</div>}
    {message && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--success)", color: "var(--success)", background: "var(--success-bg)" }}>{message}</div>}
    <div className="grid grid-cols-4 gap-2">{(["seed", "dev", "holdout", "excluded"] as const).map((split) => <button key={split} type="button" onClick={() => setActiveSplit(split)} className="border px-3 py-2 text-left" style={{ borderColor: activeSplit === split ? "var(--accent)" : "var(--border)", background: activeSplit === split ? "var(--accent-bg)" : "transparent" }}><div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>{split === "seed" ? "Seed 种子" : split === "dev" ? "Dev 开发" : split === "holdout" ? "Holdout 验收" : "排除"}</div><div className="mt-1 text-lg font-semibold" style={{ color: "var(--text)" }}>{(assignments[split] || []).length}</div></button>)}</div>
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_300px]">
      <div className="min-h-0 overflow-auto border" style={{ borderColor: "var(--border)" }}>{rows.length ? rows.map((row) => <button key={row.sample_id} type="button" onClick={() => setSelected(row.sample_id)} className="block w-full border-b px-3 py-2 text-left" style={{ borderColor: "var(--border)", background: selected === row.sample_id ? "var(--accent-bg)" : "transparent" }}><div className="truncate text-[11px] font-medium" style={{ color: "var(--text)" }}>{row.task_snapshot?.title || row.sample_id}</div><div className="mt-1 truncate text-[10px]" style={{ color: "var(--text-tertiary)" }}>{row.category || "未分类"} · {row.sample_id}</div></button>) : <div className="py-12 text-center text-[11px]" style={{ color: "var(--text-tertiary)" }}>该分组暂无样本</div>}</div>
      <div className="border p-3" style={{ borderColor: "var(--border)" }}>{selected && samples[selected] ? <div className="space-y-3 text-[10px]"><div className="break-all font-medium" style={{ color: "var(--text)" }}>{selected}</div><pre className="max-h-36 overflow-auto whitespace-pre-wrap border p-2" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>{JSON.stringify(samples[selected], null, 2)}</pre><div className="flex flex-wrap gap-1">{["seed", "dev", "holdout", "excluded"].map((split) => <button key={split} type="button" onClick={() => move(selected, split)} className="btn-ghost px-2 py-1 text-[10px]">移到 {split}</button>)}</div></div> : <div className="py-12 text-center text-[10px]" style={{ color: "var(--text-tertiary)" }}>选择样本查看详情</div>}</div>
    </div>
    <div className="flex items-center justify-between border-t pt-3" style={{ borderColor: "var(--border)" }}><span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>{draft?.validation?.length ? `有 ${draft.validation.length} 项待修正` : "建议分组可直接确认"}</span><div className="flex gap-2"><button type="button" className="btn-ghost px-3 py-1.5 text-[11px]" onClick={() => save(false)} disabled={busy}>保存草稿</button><button type="button" className="btn-primary px-3 py-1.5 text-[11px]" onClick={() => save(true)} disabled={busy}>确认并冻结 v1</button></div></div>
  </div>;
}

function CandidateStage({ task, agentId, onChanged }: { task: EvolutionTask; agentId: string; onChanged: () => Promise<void> }) {
  const [candidate, setCandidate] = useState<any>(null); const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const [message, setMessage] = useState("");
  const load = useCallback(async () => { try { const result = await api.memEvolutionCandidate(agentId, task.id); setCandidate(result.candidate); } catch (e) { setError(e instanceof Error ? e.message : "加载 Candidate 失败"); } }, [agentId, task.id]);
  useEffect(() => { void load(); }, [load]);
  const generate = async () => { setBusy(true); setError(""); setMessage(""); try { const result = await api.generateMemEvolutionCandidate(agentId, task.id); setCandidate(result.candidate); setMessage("Candidate 已生成，等待人工确认"); await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "生成失败"); } finally { setBusy(false); } };
  const confirm = async () => { setBusy(true); setError(""); try { await api.confirmMemEvolutionCandidate(agentId, task.id); setMessage("Candidate 已确认，可进入开发评估"); await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "确认失败"); } finally { setBusy(false); } };
  const canGenerate = task.stage === "dataset_confirmed" || task.stage === "candidate_generated";
  return <div className="flex min-h-[480px] min-w-0 flex-col gap-3"><div className="border-b pb-3"><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>Candidate 生成与静态检查</h3><p className="mt-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>Candidate 只使用已冻结的 Seed 样本生成，检查通过后仍需人工确认。</p></div>{error && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}>{error}</div>}{message && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--success)", color: "var(--success)", background: "var(--success-bg)" }}>{message}</div>}<div className="grid gap-2 sm:grid-cols-3"><div className="border px-3 py-2 text-[10px]" style={{ borderColor: "var(--border)" }}>版本<div className="mt-1 text-sm font-semibold" style={{ color: "var(--text)" }}>{candidate?.version ? `v${candidate.version}` : "未生成"}</div></div><div className="border px-3 py-2 text-[10px]" style={{ borderColor: "var(--border)" }}>静态检查<div className="mt-1 text-sm font-semibold" style={{ color: candidate?.checks?.passed ? "var(--success)" : "var(--text-secondary)" }}>{candidate ? (candidate.checks?.passed ? "通过" : "未通过") : "待执行"}</div></div><div className="border px-3 py-2 text-[10px]" style={{ borderColor: "var(--border)" }}>来源 Seed<div className="mt-1 text-sm font-semibold" style={{ color: "var(--text)" }}>{candidate?.source_sample_ids?.length || 0} 条</div></div></div><div className="grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_280px]"><pre className="min-h-[280px] overflow-auto whitespace-pre-wrap border p-3 text-[10px]" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>{candidate?.content || "点击生成后在这里查看完整 SKILL.md"}</pre><div className="border p-3 text-[10px]" style={{ borderColor: "var(--border)" }}><div className="font-medium" style={{ color: "var(--text)" }}>检查结果</div>{candidate?.checks?.reasons?.length ? <ul className="mt-2 list-disc space-y-1 pl-4" style={{ color: "var(--error)" }}>{candidate.checks.reasons.map((reason: string) => <li key={reason}>{reason}</li>)}</ul> : <div className="mt-2" style={{ color: "var(--text-secondary)" }}>暂无问题</div>}<div className="mt-4 font-medium" style={{ color: "var(--text)" }}>来源样本</div><div className="mt-2 space-y-1" style={{ color: "var(--text-secondary)" }}>{(candidate?.source_sample_ids || []).map((id: string) => <div key={id} className="break-all">{id}</div>)}</div></div></div><div className="flex justify-end gap-2 border-t pt-3" style={{ borderColor: "var(--border)" }}><button type="button" className="btn-ghost px-3 py-1.5 text-[11px]" disabled={!canGenerate || busy} onClick={generate}>{busy ? "处理中..." : "生成 / 重新生成"}</button><button type="button" className="btn-primary px-3 py-1.5 text-[11px]" disabled={task.stage !== "candidate_generated" || !candidate?.checks?.passed || busy} onClick={confirm}>确认 Candidate</button></div></div>;
}

function DevStage({ task, agentId, onChanged }: { task: EvolutionTask; agentId: string; onChanged: () => Promise<void> }) {
  const [report, setReport] = useState<any>(null); const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const [message, setMessage] = useState("");
  const load = useCallback(async () => { try { const result = await api.memEvolutionDev(agentId, task.id); setReport(result.report); } catch (e) { setError(e instanceof Error ? e.message : "加载评估报告失败"); } }, [agentId, task.id]);
  useEffect(() => { void load(); }, [load]);
  const run = async () => { setBusy(true); setError(""); try { await api.runMemEvolutionDev(agentId, task.id); setMessage("开发集评估已在后台启动"); await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "评估失败"); } finally { setBusy(false); } };
  const confirm = async () => { setBusy(true); setError(""); try { await api.confirmMemEvolutionDev(agentId, task.id); setMessage("评估已确认，进入失败归因"); await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "确认失败"); } finally { setBusy(false); } };
  return <div className="flex min-h-[480px] min-w-0 flex-col gap-3"><div className="border-b pb-3"><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>开发集三版本对照评估</h3><p className="mt-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>三种版本使用同一批 Dev 样本；缺少验证结果的样本会标记为外部失败。</p></div>{error && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}>{error}</div>}{message && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--success)", color: "var(--success)", background: "var(--success-bg)" }}>{message}</div>}<ExecutionMonitor task={task} agentId={agentId} onChanged={onChanged} /><div className="grid gap-2 md:grid-cols-3">{(report?.systems || []).map((row: any) => <div key={row.system} className="border px-3 py-3" style={{ borderColor: "var(--border)" }}><div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>{row.system === "without_skill" ? "无 Skill" : row.system === "active_skill" ? "当前 Active Skill" : "Candidate"}</div><div className="mt-1 text-xl font-semibold" style={{ color: "var(--text)" }}>{Math.round((row.pass_rate || 0) * 100)}%</div><div className="mt-1 text-[10px]" style={{ color: "var(--text-tertiary)" }}>通过 {row.passed}/{row.evaluated_cases} · 外部失败 {row.external_failures}</div></div>)}</div><div className="min-h-0 flex-1 overflow-auto border" style={{ borderColor: "var(--border)" }}>{report?.cases?.length ? <table className="w-full text-left text-[10px]"><thead className="sticky top-0" style={{ background: "var(--bg-inset)", color: "var(--text-tertiary)" }}><tr><th className="px-3 py-2">样本</th><th className="px-3 py-2">版本</th><th className="px-3 py-2">结果</th><th className="px-3 py-2">错误</th></tr></thead><tbody>{report.cases.map((row: any, index: number) => <tr key={`${row.sample_id}-${row.variant}-${index}`} className="border-t" style={{ borderColor: "var(--border)" }}><td className="break-all px-3 py-2">{row.sample_id}</td><td className="px-3 py-2">{row.variant}</td><td className="px-3 py-2" style={{ color: row.external_failure ? "var(--error)" : row.passed ? "var(--success)" : "var(--text-secondary)" }}>{row.external_failure ? "外部失败" : row.passed ? "通过" : "失败"}</td><td className="break-all px-3 py-2" style={{ color: "var(--text-tertiary)" }}>{row.error || "-"}</td></tr>)}</tbody></table> : <div className="py-16 text-center text-[11px]" style={{ color: "var(--text-tertiary)" }}>尚未生成评估报告</div>}</div><div className="flex justify-end gap-2 border-t pt-3" style={{ borderColor: "var(--border)" }}><button type="button" className="btn-ghost px-3 py-1.5 text-[11px]" disabled={busy || task.stage !== "candidate_confirmed" || task.execution?.status === "running"} onClick={run}>{busy ? "启动中..." : "开始评估"}</button><button type="button" className="btn-primary px-3 py-1.5 text-[11px]" disabled={busy || task.stage !== "dev_evaluated" || task.execution?.status === "running" || !report} onClick={confirm}>确认评估结果</button></div></div>;
}

function RevisionStage({ task, agentId, onChanged }: { task: EvolutionTask; agentId: string; onChanged: () => Promise<void> }) {
  const [data, setData] = useState<any>({}); const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const [message, setMessage] = useState("");
  const load = useCallback(async () => { try { setData(await api.memEvolutionRevision(agentId, task.id)); } catch (e) { setError(e instanceof Error ? e.message : "加载归因结果失败"); } }, [agentId, task.id]);
  useEffect(() => { void load(); }, [load]);
  const analyze = async () => { setBusy(true); setError(""); try { const result = await api.analyzeMemEvolutionRevision(agentId, task.id); setData((current: any) => ({ ...current, attribution: result.attribution })); setMessage("失败归因已完成"); } catch (e) { setError(e instanceof Error ? e.message : "归因失败"); } finally { setBusy(false); } };
  const revise = async () => { setBusy(true); setError(""); try { const result = await api.runMemEvolutionRevision(agentId, task.id); setData((current: any) => ({ ...current, revision: result.revision })); setMessage("局部修改 Candidate 已生成，请返回候选阶段确认"); await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "局部修改失败"); } finally { setBusy(false); } };
  const iterate = async () => { setBusy(true); setError(""); try { const result = await api.iterateMemEvolutionRevision(agentId, task.id); setData((current: any) => ({ ...current, iteration: result.iteration })); setMessage(`自动迭代结束：${result.iteration?.status || "已停止"}`); await onChanged(); } catch (e) { setError(e instanceof Error ? e.message : "自动迭代失败"); } finally { setBusy(false); } };
  const attribution = data.attribution;
  return <div className="flex min-h-[480px] min-w-0 flex-col gap-3"><div className="border-b pb-3"><h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>失败归因与局部优化</h3><p className="mt-1 text-[11px]" style={{ color: "var(--text-secondary)" }}>先分析失败来源，只有确认属于 Skill 缺陷时才允许局部修改。自动迭代最多 3 轮，连续 2 轮无提升会停止。</p></div>{error && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}>{error}</div>}{message && <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--success)", color: "var(--success)", background: "var(--success-bg)" }}>{message}</div>}<div className="grid gap-2 md:grid-cols-3"><div className="border px-3 py-3" style={{ borderColor: "var(--border)" }}><div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>归因结论</div><div className="mt-1 text-sm font-semibold" style={{ color: "var(--text)" }}>{attribution?.classification || "待分析"}</div></div><div className="border px-3 py-3" style={{ borderColor: "var(--border)" }}><div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>下一步</div><div className="mt-1 text-[11px]" style={{ color: "var(--text)" }}>{attribution?.next_action || "-"}</div></div><div className="border px-3 py-3" style={{ borderColor: "var(--border)" }}><div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>证据样本</div><div className="mt-1 text-sm font-semibold" style={{ color: "var(--text)" }}>{attribution?.evidence_sample_ids?.length || 0} 条</div></div></div>{attribution?.failure_types?.length ? <div className="border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)" }}>失败类型：{attribution.failure_types.join("、")}</div> : null}<div className="flex-1 overflow-auto border p-3 text-[11px]" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>{attribution ? <>{`系统建议：${attribution.next_action}`}{data.iteration?.iterations?.length ? <div className="mt-3 space-y-1">{data.iteration.iterations.map((row: any) => <div key={row.iteration}>第 {row.iteration} 轮：{Math.round((row.before_pass_rate || 0) * 100)}% → {Math.round((row.after_pass_rate || 0) * 100)}%，提升 {((row.improvement || 0) * 100).toFixed(1)}%</div>)}</div> : null}</> : "点击“开始归因”查看失败属于 Skill、环境还是评估问题。"}</div><div className="flex flex-wrap justify-end gap-2 border-t pt-3" style={{ borderColor: "var(--border)" }}><button type="button" className="btn-ghost px-3 py-1.5 text-[11px]" disabled={busy} onClick={analyze}>{busy ? "处理中..." : "开始归因"}</button><button type="button" className="btn-ghost px-3 py-1.5 text-[11px]" disabled={busy || attribution?.classification !== "skill_defect"} onClick={revise}>单轮局部修改</button><button type="button" className="btn-primary px-3 py-1.5 text-[11px]" disabled={busy || attribution?.classification !== "skill_defect"} onClick={iterate}>自动迭代（最多 3 轮）</button></div></div>;
}

export default function SkillEvolutionWorkspace({ agentId }: { agentId: string }) {
  const [tasks, setTasks] = useState<EvolutionTask[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [family, setFamily] = useState("");
  const [dataSource, setDataSource] = useState<"langfuse" | "skilllearnbench">("skilllearnbench");
  const [benchmarkFamilies, setBenchmarkFamilies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await api.memEvolutionTasks(agentId);
      const next = result.tasks || [];
      setTasks(next); setActiveId((current) => current && next.some((item: EvolutionTask) => item.id === current) ? current : next[0]?.id || null);
    } catch (loadError) { setError(loadError instanceof Error ? loadError.message : "加载进化任务失败"); }
    finally { setLoading(false); }
  }, [agentId]);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (dataSource !== "skilllearnbench") return;
    void api.memEvolutionSkillLearnBenchFamilies().then((result) => {
      const next = result.families || [];
      setBenchmarkFamilies(next);
      setFamily((current) => current || next[0]?.family || "");
    }).catch((loadError) => setError(loadError instanceof Error ? loadError.message : "加载标准任务集失败"));
  }, [dataSource]);
  const active = useMemo(() => tasks.find((task) => task.id === activeId) || null, [tasks, activeId]);
  const activePhase = active ? phaseIndex(active.stage) : 0;

  const create = async () => {
    if (!family.trim()) return;
    setCreating(true); setError("");
    try { const result = await api.createMemEvolutionTask(agentId, family.trim(), "", dataSource); if (dataSource === "langfuse") setFamily(""); await load(); setActiveId(result.task.id); }
    catch (createError) { setError(createError instanceof Error ? createError.message : "创建失败"); }
    finally { setCreating(false); }
  };

  const abandon = async () => {
    if (!active) return;
    try { await api.updateMemEvolutionStage(agentId, active.id, "abandoned", "用户在工作台放弃本次进化"); await load(); }
    catch (abandonError) { setError(abandonError instanceof Error ? abandonError.message : "操作失败"); }
  };

  const content = !active ? <div className="flex h-full items-center justify-center text-xs" style={{ color: "var(--text-tertiary)" }}>新建或选择一个进化任务</div> : activePhase === 0 ? <DataStage task={active} agentId={agentId} onChanged={load} /> : activePhase === 1 ? <DatasetStage task={active} agentId={agentId} onChanged={load} /> : activePhase === 2 ? <CandidateStage task={active} agentId={agentId} onChanged={load} /> : activePhase === 3 ? <DevStage task={active} agentId={agentId} onChanged={load} /> : activePhase === 4 ? <RevisionStage task={active} agentId={agentId} onChanged={load} /> : activePhase === 5 ? <ValidationStage task={active} agentId={agentId} split="regression" onChanged={load} /> : active.stage === "holdout_verified" || active.stage === "approved" || active.stage === "published" ? <ReleaseStage task={active} agentId={agentId} onChanged={load} /> : <ValidationStage task={active} agentId={agentId} split="holdout" onChanged={load} />;

  return (
    <div className="grid h-full min-h-0 w-full grid-cols-1 overflow-hidden border lg:grid-cols-[240px_minmax(0,1fr)]" style={{ borderColor: "var(--border)" }}>
      <aside className="flex min-h-0 flex-col border-b lg:border-b-0 lg:border-r" style={{ borderColor: "var(--border)", background: "var(--bg-inset)" }}>
        <div className="border-b p-3" style={{ borderColor: "var(--border)" }}><div className="mb-2 flex items-center justify-between"><span className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>进化任务</span><button className="btn-ghost p-1" onClick={load} title="刷新" type="button"><RefreshCw className="h-3.5 w-3.5" /></button></div><div className="space-y-1.5"><select className="input w-full text-[10px]" value={dataSource} onChange={(event) => { const source = event.target.value as "langfuse" | "skilllearnbench"; setDataSource(source); setFamily(source === "langfuse" ? "" : benchmarkFamilies[0]?.family || ""); }}><option value="skilllearnbench">SkillLearnBench 标准任务</option><option value="langfuse">Langfuse 真实轨迹</option></select><div className="flex gap-1">{dataSource === "skilllearnbench" ? <select className="input min-w-0 flex-1 text-[10px]" value={family} onChange={(event) => setFamily(event.target.value)}>{benchmarkFamilies.map((row) => <option key={row.family} value={row.family}>{row.family}</option>)}</select> : <input className="input min-w-0 flex-1 text-[10px]" value={family} onChange={(event) => setFamily(event.target.value)} placeholder="任务族名称" />}<button className="btn-primary p-1.5" disabled={creating || !family.trim()} onClick={create} title="新建进化任务" type="button">{creating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}</button></div></div></div>
        <div className="max-h-40 overflow-y-auto lg:max-h-none lg:flex-1">{loading ? <div className="flex justify-center py-8"><Loader2 className="h-4 w-4 animate-spin" /></div> : tasks.map((task) => <button key={task.id} className="w-full border-b px-3 py-3 text-left transition-colors" style={{ borderColor: "var(--border)", background: task.id === activeId ? "var(--active)" : "transparent" }} onClick={() => setActiveId(task.id)} type="button"><div className="truncate text-[11px] font-medium" style={{ color: "var(--text)" }}>{task.title}</div><div className="mt-1 flex items-center justify-between text-[9px]"><span style={{ color: task.stage === "abandoned" ? "var(--error)" : "var(--accent)" }}>{STAGE_LABELS[task.stage] || task.stage}</span><span style={{ color: "var(--text-tertiary)" }}>{timeLabel(task.updated_at)}</span></div></button>)}</div>
      </aside>
      <section className="flex min-h-0 min-w-0 flex-col bg-[var(--bg-elevated)]">
        {active && <><header className="flex items-center justify-between border-b px-4 py-3" style={{ borderColor: "var(--border)" }}><div className="min-w-0"><div className="truncate text-sm font-semibold" style={{ color: "var(--text)" }}>{active.title}</div><div className="mt-0.5 text-[10px]" style={{ color: "var(--text-secondary)" }}>{active.task_family} · {active.data_source === "skilllearnbench" ? "SkillLearnBench" : "Langfuse"} · {STAGE_LABELS[active.stage] || active.stage}</div></div>{active.stage !== "published" && active.stage !== "abandoned" && <button className="btn-ghost px-2 py-1 text-[10px]" style={{ color: "var(--error)" }} onClick={abandon} type="button">放弃本次进化</button>}</header><nav className="grid grid-cols-4 border-b xl:grid-cols-7" style={{ borderColor: "var(--border)" }}>{PHASES.map((phase, index) => { const Icon = phase.icon; const done = index < activePhase; const current = index === activePhase; return <div key={phase.key} className="flex min-w-0 items-center gap-2 border-r px-2 py-2" style={{ borderColor: "var(--border)", background: current ? "var(--accent-bg)" : "transparent", color: done || current ? "var(--accent)" : "var(--text-tertiary)" }}>{done ? <CheckCircle2 className="h-3.5 w-3.5 flex-shrink-0" /> : <Icon className="h-3.5 w-3.5 flex-shrink-0" />}<span className="truncate text-[10px] font-medium">{phase.label}</span></div>; })}</nav></>}
        {error && <div className="m-4 border px-3 py-2 text-[11px]" style={{ borderColor: "var(--error)", color: "var(--error)", background: "var(--error-bg)" }}>{error}</div>}
        <main className="min-h-0 flex-1 overflow-auto p-4">{content}</main>
      </section>
    </div>
  );
}
