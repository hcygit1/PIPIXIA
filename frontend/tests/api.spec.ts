import { expect, test } from "@playwright/test";

import {
  fetchCurrentModel,
  getTurnStatus,
  memBoundaryReviews,
  resolveApiBase,
  resolveMemBoundaryReview,
  streamTurn,
} from "../src/lib/api";

const originalFetch = globalThis.fetch;

test.afterEach(() => {
  globalThis.fetch = originalFetch;
});

test.describe("resolveApiBase", () => {
  test("prefers and normalizes the public API URL", () => {
    expect(
      resolveApiBase("  http://localhost:9123/api/  ", "ignored-host"),
    ).toBe("http://localhost:9123/api");
  });

  test("falls back to the browser hostname on port 8002", () => {
    expect(resolveApiBase(undefined, "devbox.local")).toBe(
      "http://devbox.local:8002/api",
    );
  });

  test("uses the configured API port with the browser hostname", () => {
    expect(resolveApiBase(undefined, "devbox.local", "9123")).toBe(
      "http://devbox.local:9123/api",
    );
  });
});

test.describe("getTurnStatus", () => {
  test("returns a terminal status reported by the backend", async () => {
    globalThis.fetch = async () =>
      new Response(
        JSON.stringify({
          turn_id: "turn-1",
          status: "done",
          position: 0,
          session_id: "main-main",
          agent_id: "main",
          error: null,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      );

    await expect(getTurnStatus("turn-1")).resolves.toMatchObject({
      turn_id: "turn-1",
      status: "done",
    });
  });

  test("does not treat an unknown turn as completed", async () => {
    globalThis.fetch = async () =>
      new Response(
        JSON.stringify({ detail: "unknown turn_id" }),
        {
          status: 404,
          headers: { "Content-Type": "application/json" },
        },
      );

    await expect(getTurnStatus("missing-turn")).rejects.toThrow(
      "unknown turn_id",
    );
  });
});

test.describe("streamTurn", () => {
  test("treats timeout as stream inactivity instead of total turn duration", async () => {
    globalThis.fetch = async (_input, init) => {
      const signal = init?.signal;
      const encoder = new TextEncoder();
      const timers: Array<ReturnType<typeof setTimeout>> = [];
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          const enqueue = (delay: number, payload: string) => {
            timers.push(setTimeout(() => {
              controller.enqueue(encoder.encode(`data: ${payload}\n`));
            }, delay));
          };
          enqueue(0, JSON.stringify({ type: "content", content: "a" }));
          enqueue(50, JSON.stringify({ type: "content", content: "b" }));
          timers.push(setTimeout(() => {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "done" })}\n`));
            controller.close();
          }, 100));
          signal?.addEventListener("abort", () => {
            for (const timer of timers) clearTimeout(timer);
            controller.error(new DOMException("Aborted", "AbortError"));
          });
        },
      });
      return new Response(body, { status: 200 });
    };

    const eventTypes: string[] = [];
    await expect(
      streamTurn("turn-long", event => eventTypes.push(event.type), { timeoutMs: 80 }),
    ).resolves.toBeUndefined();
    expect(eventTypes).toEqual(["content", "content", "done"]);
  });

  test("reconnects when the stream closes while the turn is still running", async () => {
    let streamAttempts = 0;
    globalThis.fetch = async (input) => {
      const url = String(input);
      if (url.endsWith("/status")) {
        return new Response(JSON.stringify({
          turn_id: "turn-reconnect",
          status: "running",
          position: 0,
          session_id: "main-main",
          agent_id: "main",
          error: null,
        }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      streamAttempts += 1;
      const payload = streamAttempts === 1
        ? { type: "content", content: "partial" }
        : { type: "done" };
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(payload)}\n`));
          controller.close();
        },
      });
      return new Response(body, { status: 200 });
    };

    const eventTypes: string[] = [];
    await streamTurn("turn-reconnect", event => eventTypes.push(event.type));

    expect(streamAttempts).toBe(2);
    expect(eventTypes).toEqual(["content", "done"]);
  });
});

test.describe("fetchCurrentModel", () => {
  test("returns null while no model is configured", async () => {
    globalThis.fetch = async () =>
      new Response(
        JSON.stringify({ detail: "未配置可用模型" }),
        {
          status: 404,
          headers: { "Content-Type": "application/json" },
        },
      );

    await expect(fetchCurrentModel("main")).resolves.toBeNull();
  });
});

test.describe("memory boundary reviews", () => {
  test("loads pending reviews for the selected agent", async () => {
    let requestedUrl = "";
    globalThis.fetch = async (input) => {
      requestedUrl = String(input);
      return new Response(JSON.stringify({ ok: true, reviews: [], total: 0 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };

    await expect(memBoundaryReviews("main", 25)).resolves.toMatchObject({ ok: true });
    expect(requestedUrl).toContain("/mem/boundary-reviews?");
    expect(requestedUrl).toContain("agent_id=main");
    expect(requestedUrl).toContain("limit=25");
  });

  test("submits a human boundary resolution", async () => {
    let body = "";
    globalThis.fetch = async (_input, init) => {
      body = String(init?.body);
      return new Response(JSON.stringify({
        ok: true,
        id: "review-1",
        status: "resolved",
        resolution: "create_new",
        targetTaskId: "task-2",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };

    await resolveMemBoundaryReview("main", "review-1", {
      action: "create_new",
      note: "目标不同",
    });

    expect(JSON.parse(body)).toEqual({ action: "create_new", note: "目标不同" });
  });
});
