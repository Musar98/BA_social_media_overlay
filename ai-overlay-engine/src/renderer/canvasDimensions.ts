const FALLBACK_DIMENSION_REFRESH_MS = 500;

export type CanvasDimensions = {
  width: number;
  height: number;
};

export class SourceCanvasDimensionCache {
  private sourceElement: Element | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private cachedDimensions: CanvasDimensions | null = null;
  private lastFallbackRefreshMs = 0;

  get(
    source: TexImageSource,
    fallbackWidth: number,
    fallbackHeight: number,
  ): CanvasDimensions {
    if (!this.isElementSource(source) || typeof window === "undefined") {
      return { width: fallbackWidth, height: fallbackHeight };
    }

    if (source !== this.sourceElement) {
      this.observe(source);
    }

    if (!this.resizeObserver && this.shouldRefreshFallback()) {
      this.refreshFromElement(source);
    }

    return this.cachedDimensions ?? {
      width: fallbackWidth,
      height: fallbackHeight,
    };
  }

  dispose(): void {
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.sourceElement = null;
    this.cachedDimensions = null;
    this.lastFallbackRefreshMs = 0;
  }

  private observe(source: Element): void {
    this.dispose();
    this.sourceElement = source;
    this.refreshFromElement(source);

    if (typeof ResizeObserver === "undefined") {
      return;
    }

    this.resizeObserver = new ResizeObserver(() => {
      this.refreshFromElement(source);
    });

    this.resizeObserver.observe(source);
  }

  private refreshFromElement(source: Element): void {
    const rect = source.getBoundingClientRect();
    const devicePixelRatio = window.devicePixelRatio || 1;
    const width = Math.round(rect.width * devicePixelRatio);
    const height = Math.round(rect.height * devicePixelRatio);

    this.lastFallbackRefreshMs = this.now();

    if (width > 0 && height > 0) {
      this.cachedDimensions = { width, height };
    }
  }

  private shouldRefreshFallback(): boolean {
    return this.now() - this.lastFallbackRefreshMs >= FALLBACK_DIMENSION_REFRESH_MS;
  }

  private now(): number {
    return typeof performance !== "undefined" ? performance.now() : Date.now();
  }

  private isElementSource(source: TexImageSource): source is TexImageSource & Element {
    return "getBoundingClientRect" in source;
  }
}
