import { UIState } from "../state/state";

export function createButton() {
  if (document.getElementById("filter-toggle-btn")) {
    return;
  }

  const btn = document.createElement("button");
  btn.id = "filter-toggle-btn";
  btn.style.cssText = `
        position: fixed;
        top: 15px;
        left: 70%;
        transform: translateX(-50%);
        z-index: 999999;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
        background: rgba(0, 120, 255, 0.8);
        color: white;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    `;

  btn.onclick = () => {
    UIState.filterEnabled = !UIState.filterEnabled;
    updateButton();
  };

  document.body.appendChild(btn);
  updateButton();
}

export function updateButton() {
  const btn = document.getElementById("filter-toggle-btn");
  if (!btn) {
    return;
  }

  btn.innerText = UIState.filterEnabled ? "Filter: On" : "Filter: Off";
  btn.style.background = UIState.filterEnabled
    ? "rgba(0, 120, 255, 0.8)"
    : "rgba(0,0,0,0.7)";
}
