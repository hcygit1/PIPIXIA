import { describe, expect, test, vi } from "vitest";
import { render, fireEvent } from "@testing-library/react";

import ResizeHandle from "./ResizeHandle";

describe("ResizeHandle", () => {
  test("renders a draggable handle", () => {
    const { container } = render(<ResizeHandle onResize={() => {}} />);
    const handle = container.firstElementChild as HTMLElement;
    expect(handle).toBeInTheDocument();
    expect(handle.className).toContain("cursor-col-resize");
  });

  test("reports the horizontal delta while dragging", () => {
    const onResize = vi.fn();
    const { container } = render(<ResizeHandle onResize={onResize} />);
    const handle = container.firstElementChild as HTMLElement;

    fireEvent.mouseDown(handle, { clientX: 100 });
    // 第一次移动：位移 30
    fireEvent.mouseMove(document, { clientX: 130 });
    expect(onResize).toHaveBeenCalledTimes(1);
    expect(onResize).toHaveBeenLastCalledWith(30);

    // 第二次移动：从 130 到 150，位移 20（相对上一次，而非起点）
    fireEvent.mouseMove(document, { clientX: 150 });
    expect(onResize).toHaveBeenCalledTimes(2);
    expect(onResize).toHaveBeenLastCalledWith(20);
  });

  test("supports negative deltas when dragging left", () => {
    const onResize = vi.fn();
    const { container } = render(<ResizeHandle onResize={onResize} />);
    const handle = container.firstElementChild as HTMLElement;

    fireEvent.mouseDown(handle, { clientX: 200 });
    fireEvent.mouseMove(document, { clientX: 150 });

    expect(onResize).toHaveBeenLastCalledWith(-50);
  });

  test("stops reporting deltas after mouse up", () => {
    const onResize = vi.fn();
    const { container } = render(<ResizeHandle onResize={onResize} />);
    const handle = container.firstElementChild as HTMLElement;

    fireEvent.mouseDown(handle, { clientX: 100 });
    fireEvent.mouseMove(document, { clientX: 120 });
    expect(onResize).toHaveBeenCalledTimes(1);

    fireEvent.mouseUp(document);
    fireEvent.mouseMove(document, { clientX: 300 });

    // 松手后的移动不应再触发回调
    expect(onResize).toHaveBeenCalledTimes(1);
  });

  test("sets body cursor and user-select while dragging, then restores them", () => {
    const { container } = render(<ResizeHandle onResize={() => {}} />);
    const handle = container.firstElementChild as HTMLElement;

    fireEvent.mouseDown(handle, { clientX: 100 });
    expect(document.body.style.cursor).toBe("col-resize");
    expect(document.body.style.userSelect).toBe("none");

    fireEvent.mouseUp(document);
    expect(document.body.style.cursor).toBe("");
    expect(document.body.style.userSelect).toBe("");
  });

  test("removes document listeners after mouse up so they do not leak", () => {
    const removeSpy = vi.spyOn(document, "removeEventListener");
    const { container } = render(<ResizeHandle onResize={() => {}} />);
    const handle = container.firstElementChild as HTMLElement;

    fireEvent.mouseDown(handle, { clientX: 100 });
    fireEvent.mouseUp(document);

    const removed = removeSpy.mock.calls.map((call) => call[0]);
    expect(removed).toContain("mousemove");
    expect(removed).toContain("mouseup");

    removeSpy.mockRestore();
  });
});
