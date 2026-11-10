import { describe, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import ThoughtChain from "./ThoughtChain";

// 折叠是纯 CSS 实现的（外层 div 在 grid-rows-[1fr] 与 grid-rows-[0fr] 之间切换），
// 详情节点始终挂在 DOM 上。因此断言折叠状态要看那个容器的类名，而不是查节点是否存在。
function collapseContainer(rowIndex: number): HTMLElement {
  const row = screen.getAllByTestId("tool-call")[rowIndex];
  const container = Array.from(row.children).find((el) =>
    el.className.includes("grid-rows-"),
  ) as HTMLElement;
  if (!container) throw new Error("collapse container not found");
  return container;
}

function isExpanded(rowIndex: number): boolean {
  return collapseContainer(rowIndex).className.includes("grid-rows-[1fr]");
}

describe("ThoughtChain", () => {
  test("renders nothing when there are no tool calls", () => {
    const { container } = render(<ThoughtChain toolCalls={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  test("renders one row per tool call and exposes the tool name", () => {
    render(
      <ThoughtChain
        toolCalls={[
          { tool: "grep_search", input: { q: "foo" }, output: "found" },
          { name: "read_file", input: { path: "a.ts" }, result: "content" },
        ]}
      />,
    );

    const rows = screen.getAllByTestId("tool-call");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveAttribute("data-tool-name", "grep_search");
    // 没有 tool 字段时回退到 name 字段
    expect(rows[1]).toHaveAttribute("data-tool-name", "read_file");
    expect(screen.getByText("grep_search")).toBeInTheDocument();
    expect(screen.getByText("read_file")).toBeInTheDocument();
  });

  test("marks a tool call as running until an output exists", () => {
    render(
      <ThoughtChain
        toolCalls={[
          { tool: "pending_tool", input: {} },
          { tool: "done_tool", input: {}, output: "ok" },
        ]}
      />,
    );

    const rows = screen.getAllByTestId("tool-call");
    expect(rows[0]).toHaveAttribute("data-has-output", "false");
    expect(rows[1]).toHaveAttribute("data-has-output", "true");
  });

  test("treats result field as an output just like output field", () => {
    render(
      <ThoughtChain toolCalls={[{ tool: "t", input: {}, result: "from-result" }]} />,
    );
    expect(screen.getByTestId("tool-call")).toHaveAttribute(
      "data-has-output",
      "true",
    );
  });

  test("expands a row on click and collapses it again", async () => {
    const user = userEvent.setup();
    render(
      <ThoughtChain
        toolCalls={[{ tool: "grep_search", input: { q: "needle" }, output: "haystack" }]}
      />,
    );

    // 默认折叠
    expect(isExpanded(0)).toBe(false);

    await user.click(screen.getByRole("button"));
    expect(isExpanded(0)).toBe(true);
    // 展开后能看到输出内容
    expect(screen.getByText("haystack")).toBeInTheDocument();

    await user.click(screen.getByRole("button"));
    expect(isExpanded(0)).toBe(false);
  });

  test("expands rows independently", async () => {
    const user = userEvent.setup();
    render(
      <ThoughtChain
        toolCalls={[
          { tool: "first", input: {}, output: "one" },
          { tool: "second", input: {}, output: "two" },
        ]}
      />,
    );

    await user.click(screen.getAllByRole("button")[1]);

    // 只有第二行被展开
    expect(isExpanded(0)).toBe(false);
    expect(isExpanded(1)).toBe(true);

    // 再展开第一行，两行互不影响
    await user.click(screen.getAllByRole("button")[0]);
    expect(isExpanded(0)).toBe(true);
    expect(isExpanded(1)).toBe(true);
  });

  test("shows a placeholder while the tool is still executing", async () => {
    const user = userEvent.setup();
    render(<ThoughtChain toolCalls={[{ tool: "slow_tool", input: {} }]} />);

    await user.click(screen.getByRole("button"));
    expect(screen.getByText("Executing...")).toBeInTheDocument();
  });

  test("renders string input verbatim and object input as pretty JSON", async () => {
    const user = userEvent.setup();
    render(
      <ThoughtChain
        toolCalls={[
          { tool: "raw", input: "plain string", output: "ok" },
          { tool: "obj", input: { key: "value" }, output: "ok" },
        ]}
      />,
    );

    await user.click(screen.getAllByRole("button")[0]);
    expect(screen.getByText("plain string")).toBeInTheDocument();
  });
});
