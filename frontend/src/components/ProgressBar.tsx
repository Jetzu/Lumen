import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { useStore, type Job } from "../store";

/** Jobs younger than this are not shown: a 30 ms Gaussian must not flash a progress chip. */
const SHOW_AFTER_MS = 300;

function fmtElapsed(sec: number): string {
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)} s`;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec - m * 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * The running long operation (top bar): name, percentage, message, elapsed time and a Cancel
 * button.  Driven by the `progress` WebSocket events dispatched into `store.jobs`.
 */
export default function ProgressBar() {
  const jobs = useStore((s) => s.jobs);
  const cancelJob = useStore((s) => s.cancelJob);
  const list: Job[] = Object.values(jobs).sort((a, b) => a.seenAt - b.seenAt);
  const [now, setNow] = useState(() => performance.now());
  const running = list.length > 0;
  useEffect(() => {
    if (!running) return;
    setNow(performance.now());
    const t = window.setInterval(() => setNow(performance.now()), 200);
    return () => window.clearInterval(t);
  }, [running]);
  const job = list.find((j) => now - j.seenAt >= SHOW_AFTER_MS);
  if (!job) return null;
  const indeterminate = !(job.fraction > 0);
  const cancelling = !!job.cancelling || !!job.cancelled; // requested here, or by another client (the server flag)
  const pct = Math.max(0, Math.min(100, Math.round(job.fraction * 100)));
  const elapsed = Math.max(0, (now - job.seenAt) / 1000);
  const others = list.length - 1;
  return (
    <div
      className={"progress-chip" + (cancelling ? " cancelling" : "")}
      role="progressbar"
      aria-label={job.name}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : pct}
      title={`${job.name}${job.message ? " — " + job.message : ""}${others > 0 ? ` (+${others} more)` : ""}`}
    >
      <span className="progress-name">{job.name}</span>
      {!indeterminate && <span className="progress-pct">{pct}%</span>}
      <span className="progress-msg">{job.message}</span>
      {others > 0 && <span className="progress-more">+{others}</span>}
      <span className="progress-elapsed">{fmtElapsed(elapsed)}</span>
      <button
        className="progress-cancel"
        title={cancelling ? "Cancelling…" : `Cancel ${job.name}`}
        disabled={cancelling}
        onClick={() => void cancelJob(job.id)}
      >
        {cancelling ? "Cancelling…" : <X size={12} />}
      </button>
      <span className="progress-track">
        <span className={"progress-fill" + (indeterminate ? " indeterminate" : "")} style={indeterminate ? undefined : { width: `${pct}%` }} />
      </span>
    </div>
  );
}
