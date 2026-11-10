import { describe, expect, test, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { ChatMessage as ChatMsgType } from "@/lib/chatState";

// ChatMessage 通过 useUi() 读取文案。这里用一个可控的最小 t 对象替换，
// 避免为每个用例启动完整的 UiProvider（它内部会拉取主题、侧栏状态等）。
vi.mock("@/lib/uiContext", () => {
  const t = new Proxy(
    {
      noReply: "（无回复）",
      parsingTools: "正在解析工具调用",
      commandNoticeTitle: "命令",
      helpListTitle: "可用命令",
    } as Record<string, string>,
    {
      get(target, prop: string) {
        return target[prop] ?? prop;
      },
    },
  );
  return { useUi: () => ({ t }) };
});

// SubagentInlineCard 需要 subagent context，且与本组件的断言无关，直接置空。
vi.mock("./SubagentInlineCard", () => ({
  default: () => null,
}));

import ChatMessage from "./ChatMessage";

function makeMessage(overrides: Partial<ChatMsgType> = {}): ChatMsgType {
  return {
    id: "m1",
    role: "assistant",
    content: "hello",
    createdAt: Date.parse("2026-01-01T10:00:00Z"),
    ...overrides,
  };
}

describe("ChatMessage", () => {
  test("tags the root node with role so callers can style by author", () => {
    const { rerender } = render(<ChatMessage message={makeMessage()} />);
    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-role",
      "assistant",
    );

    rerender(<ChatMessage message={makeMessage({ role: "user" })} />);
    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-role",
      "user",
    );
  });

  test("renders user text as plain wrapped text", () => {
    render(<ChatMessage message={makeMessage({ role: "user", content: "hi there" })} />);
    const paragraph = screen.getByText("hi there");
    expect(paragraph.tagName).toBe("P");
    expect(paragraph.className).toContain("whitespace-pre-wrap");
  });

  test("renders assistant markdown through ReactMarkdown", () => {
    render(
      <ChatMessage message={makeMessage({ content: "**bold text**" })} />,
    );
    // markdown 被解析成 <strong>，而不是原样输出星号
    const strong = screen.getByText("bold text");
    expect(strong.tagName).toBe("STRONG");
  });

  test("shows the no-reply placeholder for an empty finished message", () => {
    render(
      <ChatMessage
        message={makeMessage({ content: "", isStreaming: false })}
      />,
    );
    // 文案被包裹在括号里，用正则做子串匹配
    expect(screen.getByText(/无回复/)).toBeInTheDocument();
  });

  test("shows the parsing indicator for a text-form tool call", () => {
    render(
      <ChatMessage
        message={makeMessage({ content: "functions.grep", isStreaming: true })}
      />,
    );
    expect(screen.getByText("正在解析工具调用")).toBeInTheDocument();
  });

  test("renders the tool chain for non-spawn tool calls", () => {
    render(
      <ChatMessage
        message={makeMessage({
          content: "",
          toolCalls: [{ tool: "grep_search", input: { q: "x" }, output: "ok" }],
        })}
      />,
    );
    expect(screen.getByTestId("tool-call")).toBeInTheDocument();
    expect(screen.getByTestId("tool-call")).toHaveAttribute(
      "data-tool-name",
      "grep_search",
    );
  });

  test("renders a command notice instead of a bubble for command role", () => {
    render(
      <ChatMessage
        message={makeMessage({ role: "command", content: "自定义命令输出" })}
      />,
    );
    expect(screen.getByTestId("chat-message")).toHaveAttribute(
      "data-role",
      "command",
    );
    expect(screen.getByText("命令")).toBeInTheDocument();
    expect(screen.getByText("自定义命令输出")).toBeInTheDocument();
  });

  test("localizes a known help-command block", () => {
    render(
      <ChatMessage
        message={makeMessage({
          role: "command",
          content: "## 可用命令\n- `/new`",
        })}
      />,
    );
    // 命中帮助模板分支后，会输出完整的命令清单
    expect(screen.getByText(/\/compact/)).toBeInTheDocument();
    expect(screen.getByText(/\/status/)).toBeInTheDocument();
  });

  test("only shows metadata on the last, non-streaming assistant message", () => {
    const base = makeMessage({
      content: "done",
      finishedAt: Date.parse("2026-01-01T10:00:02Z"),
      usage: {
        input_tokens: 1200,
        output_tokens: 340,
        total_tokens: 1540,
        duration_ms: 2500,
      },
    });

    const { rerender } = render(<ChatMessage message={base} isLast />);
    expect(screen.getByText("1,200 in / 340 out")).toBeInTheDocument();
    expect(screen.getByText("2.5s")).toBeInTheDocument();

    // isLast=false 时不显示元信息
    rerender(<ChatMessage message={base} isLast={false} />);
    expect(screen.queryByText("1,200 in / 340 out")).not.toBeInTheDocument();
  });

  test("hides metadata while still streaming", () => {
    render(
      <ChatMessage
        message={makeMessage({
          content: "typing",
          isStreaming: true,
          usage: {
            input_tokens: 10,
            output_tokens: 2,
            total_tokens: 12,
            duration_ms: 500,
          },
        })}
        isLast
      />,
    );
    expect(screen.queryByText(/in \/ .* out/)).not.toBeInTheDocument();
  });

  test("copies the message content and confirms visually", async () => {
    // 注意：必须在 userEvent.setup() 之后再装 clipboard mock，
    // 因为 setup() 会用自己带 spy 的 clipboard 覆盖 navigator.clipboard。
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value: { writeText },
    });

    render(<ChatMessage message={makeMessage({ content: "copy me" })} />);

    await user.click(screen.getByRole("button", { name: "复制" }));
    expect(writeText).toHaveBeenCalledWith("copy me");

    // 复制后按钮的无障碍名称切换，作为“已复制”反馈
    expect(screen.getByRole("button", { name: "已复制" })).toBeInTheDocument();
  });

  test("does not render a copy button for user messages", () => {
    render(
      <ChatMessage message={makeMessage({ role: "user", content: "mine" })} />,
    );
    expect(screen.queryByRole("button", { name: "复制" })).not.toBeInTheDocument();
  });
});
