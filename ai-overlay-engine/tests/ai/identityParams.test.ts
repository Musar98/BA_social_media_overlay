import { describe, expect, it } from "vitest";
import { createIdentityAIParams } from "../../src/ai/IdentityParams";

describe("createIdentityAIParams", () => {
  it("returns neutral identity parameters", () => {
    expect(createIdentityAIParams()).toEqual({
      sharp: 1.0,
      exposure: 0.0,
      contrast: 1.0,
      saturation: 1.0,
      blur: 0.0,
      imageMean: 0.5,
      toneCurve: [1, 1, 1, 1, 1, 1, 1, 1],
      colorCurve: [
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
      ],
    });
  });

  it("returns fresh curve arrays for each call", () => {
    const first = createIdentityAIParams();
    const second = createIdentityAIParams();

    expect(first.toneCurve).not.toBe(second.toneCurve);
    expect(first.colorCurve).not.toBe(second.colorCurve);
  });
});
