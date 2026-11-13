import { useStore } from "../store";
import ConfirmDialog from "./ConfirmDialog";

/**
 * Toast notifications – and the modal confirm dialog (store.askConfirm), which lives here so it is always mounted.
 * The stack sits at the bottom-left of the viewport (never over an operation sheet's buttons, E5-5 / E6-2); each
 * toast is its own live region (`role=status` / `alert`), so a screen reader announces the new one only (E5-10),
 * and a toast may carry action buttons ("Undo clear", "Auto-threshold and run").
 */
export default function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);
  const confirm = useStore((s) => s.confirm);
  return (
    <>
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={"toast " + t.kind + (t.actions?.length ? " with-actions" : "")} role={t.kind === "error" ? "alert" : "status"} aria-atomic="true" tabIndex={0} title="Dismiss" onClick={() => dismiss(t.id)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " " || e.key === "Escape") { e.preventDefault(); e.stopPropagation(); dismiss(t.id); } }}>
            <span className="toast-msg">{t.msg}</span>
            {t.actions && t.actions.length > 0 && (
              <span className="toast-actions">
                {t.actions.map((a) => (
                  <button key={a.label} type="button" className="btn sm primary" onClick={(e) => { e.stopPropagation(); dismiss(t.id); a.run(); }} onKeyDown={(e) => e.stopPropagation()}>{a.label}</button>
                ))}
              </span>
            )}
          </div>
        ))}
      </div>
      {confirm && <ConfirmDialog />}
    </>
  );
}
