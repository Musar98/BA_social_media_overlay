type LogPayload = Record<string, unknown>;

class OverlayLogger {
  info(event: string, payload?: LogPayload): void {
    console.info(this.format(event, payload));
  }

  error(event: string, payload?: LogPayload): void {
    console.error(this.format(event, payload));
  }

  private format(event: string, payload?: LogPayload): string {
    if (!payload) {
      return `[ai-overlay] ${event}`;
    }

    return `[ai-overlay] ${event} ${JSON.stringify(payload)}`;
  }
}

export const overlayLogger = new OverlayLogger();
