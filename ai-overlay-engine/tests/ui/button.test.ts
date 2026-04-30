// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";
import { filterButton } from "../../src/ui/button";
import { UIState } from "../../src/state/state";

describe("FilterButton", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    UIState.filterEnabled = false;
  });

  it("creates the button in the DOM", () => {
    filterButton.create();
    const btn = document.getElementById("filter-toggle-btn");
    expect(btn).not.toBeNull();
    expect(btn?.innerText).toBe("Filter: Off");
  });

  it("toggles filter state on click", () => {
    filterButton.create();
    const btn = document.getElementById(
      "filter-toggle-btn",
    ) as HTMLButtonElement;

    btn.click();
    expect(UIState.filterEnabled).toBe(true);
    expect(btn.innerText).toBe("Filter: On");

    btn.click();
    expect(UIState.filterEnabled).toBe(false);
    expect(btn.innerText).toBe("Filter: Off");
  });

  it("does not create duplicate buttons", () => {
    filterButton.create();
    filterButton.create();
    const buttons = document.querySelectorAll("#filter-toggle-btn");
    expect(buttons.length).toBe(1);
  });
});
