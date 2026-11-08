import type { PoseFrame } from './gestures';

const SAMPLE_WINDOW = 120;

function p95(samples: readonly number[]): number | undefined {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.ceil(sorted.length * 0.95) - 1];
}

export class HandPerformance {
  private readonly receipts: { at: number; skipped: number }[] = [];
  private readonly receiptToSubmit: number[] = [];
  private readonly poseToSubmit: number[] = [];
  private sequence: number | undefined;

  receive(sequence: number, at: number) {
    const skipped = this.sequence === undefined ? 0 : Math.max(0, sequence - this.sequence - 1);
    this.sequence = sequence;
    this.receipts.push({ at, skipped });
    if (this.receipts.length > SAMPLE_WINDOW) this.receipts.shift();
  }

  submitted(frame: PoseFrame, receivedAt: number, submittedAt: number) {
    this.receiptToSubmit.push(submittedAt - receivedAt);
    this.poseToSubmit.push(submittedAt - frame.capturedAt);
    if (this.receiptToSubmit.length > SAMPLE_WINDOW) {
      this.receiptToSubmit.shift();
      this.poseToSubmit.shift();
    }
  }

  reset() {
    this.sequence = undefined;
    this.receipts.length = 0;
    this.receiptToSubmit.length = 0;
    this.poseToSubmit.length = 0;
  }

  read() {
    const latest = this.receipts.at(-1);
    const span = latest ? latest.at - this.receipts[0].at : 0;
    return {
      deliveryFps: latest && span > 0 && performance.now() - latest.at < 1000 ? (this.receipts.length - 1) * 1000 / span : undefined,
      skipped: this.receipts.length ? this.receipts.reduce((sum, sample) => sum + sample.skipped, 0) - this.receipts[0].skipped : undefined,
      receiptToSubmitMs: p95(this.receiptToSubmit),
      poseToSubmitMs: p95(this.poseToSubmit),
    };
  }
}
