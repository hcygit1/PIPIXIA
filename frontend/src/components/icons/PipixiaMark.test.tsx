import { describe, expect, test } from "vitest";
import { render } from "@testing-library/react";

import PIPIXIAMark from "./PipixiaMark";

describe("PIPIXIAMark", () => {
  test("renders an svg that is hidden from assistive tech", () => {
    const { container } = render(<PIPIXIAMark />);
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(svg).toHaveAttribute("viewBox", "0 0 24 24");
  });

  test("applies the default size class", () => {
    const { container } = render(<PIPIXIAMark />);
    expect(container.querySelector("svg")).toHaveAttribute("class", "h-4 w-4");
  });

  test("accepts a custom class name", () => {
    const { container } = render(<PIPIXIAMark className="h-8 w-8 text-red-500" />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("class", "h-8 w-8 text-red-500");
  });

  test("uses the default stroke width on every stroked path", () => {
    const { container } = render(<PIPIXIAMark />);
    const stroked = Array.from(container.querySelectorAll("path[stroke]"));
    expect(stroked.length).toBeGreaterThan(0);
    for (const path of stroked) {
      expect(path).toHaveAttribute("stroke-width", "1.8");
    }
  });

  test("honours a custom stroke width", () => {
    const { container } = render(<PIPIXIAMark strokeWidth={3} />);
    const stroked = Array.from(container.querySelectorAll("path[stroke]"));
    for (const path of stroked) {
      expect(path).toHaveAttribute("stroke-width", "3");
    }
  });

  test("draws the accent dot as a filled circle", () => {
    const { container } = render(<PIPIXIAMark />);
    const dot = container.querySelector("circle");
    expect(dot).not.toBeNull();
    expect(dot).toHaveAttribute("fill", "currentColor");
  });
});
