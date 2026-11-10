import { describe, expect, test, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// ChatInput 从三个 context 取值，这里用轻量替身，聚焦组件自身行为。
const sendMessage = vi.fn();
const stopStreaming = vi.fn();
const setShowConfigModal = vi.fn();

const chatState = {
  sendMessage,
  stopStreaming,
  isStreaming: false,
  lastUsage: null as unknown,
  contextUtilization: null as unknown,
};

vi.mock("@/lib/agentContext", () => ({
  useAgentContext: () => ({ currentModel: "test-model" }),
}));

vi.mock("@/lib/chatContext", () => ({
  useChatState: () => chatState,
}));

vi.mock("@/lib/uiContext", () => {
  const t = new Proxy(
    {
      cmdNewDesc: "新建会话",
      cmdResetDesc: "重置会话",
      cmdCompactDesc: "压缩上下文",
      cmdHelpDesc: "查看帮助",
      cmdStatusDesc: "查看状态",
      cmdContextDesc: "查看上下文",
      cmdUsageDesc: "查看用量",
      cmdModelDesc: "查看模型",
      cmdSubagentsDesc: "查看子代理",
      cmdWhoamiDesc: "查看身份",
      commandPaletteTitle: "可用命令",
      inputPlaceholder: "说点什么",
    } as Record<string, string>,
    {
      get(target, prop: string) {
        return target[prop] ?? prop;
      },
    },
  );
  return { useUi: () => ({ setShowConfigModal, t }) };
});

import ChatInput from "./ChatInput";

function input() {
  return screen.getByTestId("chat-input") as HTMLTextAreaElement;
}

describe("ChatInput", () => {
  beforeEach(() => {
    chatState.isStreaming = false;
    sendMessage.mockClear();
    stopStreaming.mockClear();
  });

  test("renders the placeholder from i18n", () => {
    render(<ChatInput />);
    expect(input()).toHaveAttribute("placeholder", "说点什么");
  });

  test("submits trimmed text on Enter and clears the box", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "  hello world  ");
    await user.keyboard("{Enter}");

    expect(sendMessage).toHaveBeenCalledWith("hello world");
    expect(input().value).toBe("");
  });

  test("does not submit whitespace-only input", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "   ");
    await user.keyboard("{Enter}");

    expect(sendMessage).not.toHaveBeenCalled();
  });

  test("inserts a newline instead of submitting when Shift is held", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "line1");
    await user.keyboard("{Shift>}{Enter}{/Shift}");

    expect(sendMessage).not.toHaveBeenCalled();
    expect(input().value).toContain("line1");
  });

  test("does not submit while a response is streaming", async () => {
    chatState.isStreaming = true;
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "blocked");
    await user.keyboard("{Enter}");

    expect(sendMessage).not.toHaveBeenCalled();
  });

  test("opens the command palette when typing a slash", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    expect(screen.queryByText("可用命令")).not.toBeInTheDocument();

    await user.type(input(), "/");
    expect(screen.getByText("可用命令")).toBeInTheDocument();
    expect(screen.getByText("/new")).toBeInTheDocument();
    expect(screen.getByText("/compact")).toBeInTheDocument();
  });

  test("filters the palette by the typed prefix", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "/co");

    expect(screen.getByText("/compact")).toBeInTheDocument();
    expect(screen.getByText("/context")).toBeInTheDocument();
    expect(screen.queryByText("/new")).not.toBeInTheDocument();
  });

  test("does not open the palette for normal text", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "hello");
    expect(screen.queryByText("可用命令")).not.toBeInTheDocument();
  });

  test("fills the box after picking a command", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "/new");
    // 面板里的命令是 <code>，输入框里是 textarea 的 value，
    // 用 selector 限定在面板按钮内点击，避免匹配到多个节点。
    const paletteButton = screen
      .getAllByRole("button")
      .find((btn) => btn.textContent?.includes("/new"));
    expect(paletteButton).toBeDefined();
    await user.click(paletteButton as HTMLElement);

    expect(input().value).toBe("/new");
  });

  test("keeps the palette closed after selecting a command", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "/new");
    const paletteButton = screen
      .getAllByRole("button")
      .find((btn) => btn.textContent?.includes("/new"));
    await user.click(paletteButton as HTMLElement);

    // 选中命令后焦点会交回输入框，此时文本以 "/" 开头。
    // 早期实现里 onFocus 会把面板重新打开，挡住输入框；
    // 组件已用 skipFocusOpen 标记跳过这次由代码触发的 focus，这里守住该行为。
    expect(screen.queryByText("可用命令")).not.toBeInTheDocument();
  });

  test("reopens the palette when the user focuses again and types a slash", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "/new");
    const paletteButton = screen
      .getAllByRole("button")
      .find((btn) => btn.textContent?.includes("/new"));
    await user.click(paletteButton as HTMLElement);
    expect(screen.queryByText("可用命令")).not.toBeInTheDocument();

    // 修复不能顺手把正常路径也关掉：用户清空后重新输入斜杠，面板应该再次出现。
    await user.clear(input());
    await user.type(input(), "/co");
    expect(screen.getByText("可用命令")).toBeInTheDocument();
  });

  test("closes the palette on Escape", async () => {
    const user = userEvent.setup();
    render(<ChatInput />);

    await user.type(input(), "/");
    expect(screen.getByText("可用命令")).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByText("可用命令")).not.toBeInTheDocument();
  });

  test("swaps the send button for a stop button while streaming", () => {
    chatState.isStreaming = true;
    render(<ChatInput />);

    // 流式期间发送按钮不存在，取而代之的是一个无 testid 的停止按钮
    expect(screen.queryByTestId("chat-send-btn")).not.toBeInTheDocument();
    expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
  });

  test("stop button triggers stopStreaming and never sends", async () => {
    chatState.isStreaming = true;
    const user = userEvent.setup();
    render(<ChatInput />);

    // 停止按钮是输入框右侧那个（没有 chat-send-btn 标记），
    // 用容器定位：它包含 StopCircle 图标的 svg
    const stopButton = screen
      .getAllByRole("button")
      .find((btn) => btn.className.includes("rounded-xl") && !btn.hasAttribute("data-testid"));
    expect(stopButton).toBeDefined();

    await user.click(stopButton as HTMLElement);

    expect(stopStreaming).toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  });
});
