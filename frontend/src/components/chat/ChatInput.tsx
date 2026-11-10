"use client";

import { useState, useRef, useCallback, useMemo } from "react";
import { useAgentContext } from "@/lib/agentContext";
import { useChatState } from "@/lib/chatContext";
import { useUi } from "@/lib/uiContext";
import { StopCircle, ArrowUp, ChevronDown, Sparkles } from "lucide-react";

function formatUsageCount(n: number | undefined | null): string {
  if (!n || n <= 0) return "0";
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function ContextMiniRing({ utilization }: { utilization: number | null }) {
  const pct = utilization == null ? null : Math.min(Math.round(utilization * 100), 100);
  const ratio = utilization == null ? 0 : Math.max(0, Math.min(utilization, 1));
  const r = 7;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - ratio);
  const color =
    pct == null
      ? "var(--text-tertiary)"
      : pct >= 80
        ? "var(--error)"
        : pct >= 60
          ? "var(--warning)"
          : "var(--accent)";

  return (
    <div className="flex items-center gap-1" title={pct == null ? "上下文使用量暂不可用" : `上下文使用量 ${pct}%`}>
      <svg width="16" height="16" viewBox="0 0 18 18" className="flex-shrink-0">
        <circle cx="9" cy="9" r={r} fill="none" stroke="var(--border)" strokeWidth="2" />
        <circle
          cx="9"
          cy="9"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 9 9)"
        />
      </svg>
      <span className="text-[10px] tabular-nums" style={{ color }}>
        {pct == null ? "--" : `${pct}%`}
      </span>
    </div>
  );
}

export default function ChatInput() {
  const { currentModel } = useAgentContext();
  const {
    sendMessage,
    isStreaming,
    stopStreaming,
    lastUsage,
    contextUtilization,
  } = useChatState();
  const { setShowConfigModal, t } = useUi();
  const [text, setText] = useState("");
  const [showCommands, setShowCommands] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // 选中命令后会把焦点交回输入框，此时文本以 "/" 开头，
  // onFocus 会误判为用户正在输入斜杠，导致面板立即重新弹出挡住输入框。
  // 用一个一次性标记跳过这次由代码触发的 focus。
  const skipFocusOpen = useRef(false);
  const commands = useMemo(() => ([
    { cmd: "/new", desc: t.cmdNewDesc },
    { cmd: "/reset", desc: t.cmdResetDesc },
    { cmd: "/compact", desc: t.cmdCompactDesc },
    { cmd: "/help", desc: t.cmdHelpDesc },
    { cmd: "/status", desc: t.cmdStatusDesc },
    { cmd: "/context", desc: t.cmdContextDesc },
    { cmd: "/usage", desc: t.cmdUsageDesc },
    { cmd: "/model", desc: t.cmdModelDesc },
    { cmd: "/subagents", desc: t.cmdSubagentsDesc },
    { cmd: "/whoami", desc: t.cmdWhoamiDesc },
  ]), [t]);

  const filteredCommands = useMemo(() => {
    if (!text.startsWith("/")) return [];
    const query = text.toLowerCase();
    return commands.filter(c => c.cmd.startsWith(query));
  }, [text, commands]);

  const handleSubmit = useCallback(() => {
    if (!text.trim() || isStreaming) return;
    sendMessage(text.trim());
    setText("");
    setShowCommands(false);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [text, isStreaming, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
    if (e.key === "Escape") {
      setShowCommands(false);
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setText(val);
    setShowCommands(val.startsWith("/") && val.length > 0);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  const selectCommand = (cmd: string) => {
    setText(cmd);
    setShowCommands(false);
    skipFocusOpen.current = true;
    textareaRef.current?.focus();
  };

  return (
    <div
      className="relative border-t px-4 py-3"
      style={{ background: "var(--bg)", borderColor: "var(--border)" }}
    >
      {/* Command palette */}
      {showCommands && filteredCommands.length > 0 && (
        <div
          className="absolute bottom-full left-4 right-4 z-10 mb-2 max-h-52 overflow-y-auto rounded-2xl animate-scale-in dropdown-menu"
          style={{ boxShadow: "var(--shadow-md)" }}
        >
          <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider font-medium" style={{ color: "var(--text-tertiary)" }}>
            {t.commandPaletteTitle}
          </div>
          {filteredCommands.map((c) => (
            <button
              key={c.cmd}
              onClick={() => selectCommand(c.cmd)}
              className="dropdown-item"
            >
              <code className="font-mono text-[11px] px-1.5 py-0.5 rounded"
                style={{ background: "var(--accent-muted)", color: "var(--accent)" }}>
                {c.cmd}
              </code>
              <span className="text-xs" style={{ color: "var(--text-secondary)" }}>{c.desc}</span>
            </button>
          ))}
        </div>
      )}

      <div className="mx-auto max-w-[860px]">
        <div
          className="relative overflow-hidden rounded-[22px]"
          style={{
            background: "color-mix(in srgb, var(--bg-elevated) 94%, var(--bg) 6%)",
            border: "1px solid var(--border)",
            boxShadow: "0 1px 0 rgba(15, 23, 42, 0.02)",
          }}
        >
          <div className="relative">
            <textarea
              ref={textareaRef}
              data-testid="chat-input"
              value={text}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              onFocus={() => {
                if (skipFocusOpen.current) {
                  skipFocusOpen.current = false;
                  return;
                }
                if (text.startsWith("/")) setShowCommands(true);
              }}
              onBlur={() => setTimeout(() => setShowCommands(false), 200)}
              placeholder={t.inputPlaceholder}
              rows={1}
              className="w-full resize-none bg-transparent pl-4 pr-14 py-3.5 text-sm focus:outline-none"
              style={{ color: "var(--text)", border: "none" }}
            />
            <div className="absolute bottom-2 right-2">
              {isStreaming ? (
                <button
                  onClick={stopStreaming}
                  className="rounded-xl p-2 text-white transition-all"
                  style={{ background: "var(--error)", boxShadow: "0 1px 2px rgba(15, 23, 42, 0.08)" }}
                >
                  <StopCircle className="w-4 h-4" />
                </button>
              ) : (
                <button
                  data-testid="chat-send-btn"
                  onClick={handleSubmit}
                  disabled={!text.trim()}
                  className="rounded-xl p-2 text-white transition-all disabled:cursor-not-allowed disabled:opacity-20"
                  style={{
                    background: text.trim() ? "var(--accent)" : "var(--text-tertiary)",
                    boxShadow: text.trim() ? "0 1px 2px rgba(15, 23, 42, 0.08)" : "none",
                  }}
                >
                  <ArrowUp className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
          <div
            className="flex items-center justify-between gap-2 border-t px-3 py-1.5"
            style={{ borderColor: "var(--border)", background: "color-mix(in srgb, var(--bg-elevated) 96%, var(--bg) 4%)" }}
          >
            <span className="truncate text-[9px] uppercase tracking-[0.12em]" style={{ color: "var(--text-tertiary)" }}>
              {t.sendHint}
            </span>
            <div className="flex items-center gap-1.5 overflow-x-auto">
              <button
                type="button"
                onClick={() => setShowConfigModal(true)}
                className="flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] transition-all"
                style={{
                  background: "transparent",
                  color: "var(--text)",
                  border: "1px solid var(--border)",
                }}
                title="当前模型"
              >
                <Sparkles className="h-3 w-3" style={{ color: "var(--accent)" }} />
                <span className="whitespace-nowrap">{currentModel?.name || "未设置模型"}</span>
                <ChevronDown className="h-3 w-3" style={{ color: "var(--text-tertiary)" }} />
              </button>
              <div
                className="flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px]"
                style={{
                  background: "transparent",
                  color: "var(--text-secondary)",
                  border: "1px solid var(--border)",
                }}
                title={
                  lastUsage
                    ? `输入 ${lastUsage.input_tokens.toLocaleString()} / 输出 ${lastUsage.output_tokens.toLocaleString()}${lastUsage.duration_ms ? ` / ${(lastUsage.duration_ms / 1000).toFixed(1)}s` : ""}`
                    : "暂无本轮用量统计"
                }
              >
                <span className="whitespace-nowrap tabular-nums">
                  {lastUsage
                    ? `${formatUsageCount(lastUsage.input_tokens)} in / ${formatUsageCount(lastUsage.output_tokens)} out`
                    : "暂无用量"}
                </span>
                <ContextMiniRing utilization={contextUtilization} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
