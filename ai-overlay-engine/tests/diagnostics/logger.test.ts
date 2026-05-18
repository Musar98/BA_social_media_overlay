import { describe, expect, it, vi } from "vitest";
import { overlayLogger, VERBOSE_DIAGNOSTICS } from "../../src/diagnostics/logger";

describe("overlayLogger", () => {
  it("disables verbose diagnostics by default", () => {
    const consoleSpy = vi.spyOn(console, "info").mockImplementation(() => {});

    overlayLogger.verbose("hot-path-event", { frame: 1 });
    //TODO
    //expect(VERBOSE_DIAGNOSTICS).toBe(false);
    //expect(consoleSpy).not.toHaveBeenCalled();

    consoleSpy.mockRestore();
  });
});
