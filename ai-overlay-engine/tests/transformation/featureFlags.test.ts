import { describe, expect, it } from "vitest";
import { SHARPEN_ENABLED } from "../../src/transformation/FeatureFlags";

describe("transformation feature flags", () => {
  it("keeps sharpening enabled by default", () => {
    expect(SHARPEN_ENABLED).toBe(true);
  });
});
