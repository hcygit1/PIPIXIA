import { defineConfig } from "@playwright/test";

// 这个项目里的 Playwright 用来跑 frontend/tests 下的集成用例（纯 Node 断言，
// 不启动浏览器）。组件级单元测试归 vitest 管，因此这里必须把 testMatch
// 限定在 tests/ 目录，否则默认规则会把 src/**/*.test.tsx 也当成 E2E 用例。
export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "list" : "line",
});
