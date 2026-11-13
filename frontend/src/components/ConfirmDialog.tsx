import { useRef, useState } from "react";
import { TriangleAlert } from "lucide-react";
import { useStore } from "../store";
import { useFocusTrap } from "../focusTrap";

/**
 * Modal question (store.askConfirm): "Close “x”? It has unsaved changes." with Cancel / optional
 * alternative ("Save…") / confirm buttons and a "Don't ask again" switch when the request names a
 * preference.  Focus starts on Cancel (the safe answer), Tab cycles inside the dialog, Esc cancels and
 * the element the question came from (menu trigger, tab, canvas) gets the focus back (useFocusTrap).
 * Mounted from Toasts.tsx so it is always available (App.tsx is not owned by this package).
 */
export default function ConfirmDialog() {
  const req = useStore((s) => s.confirm)!;
  const [dontAsk, setDontAsk] = useState(false);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const answer = (a: "confirm" | "alt" | "cancel") => req.resolve(a, dontAsk);
  // isolate: keep single-letter shortcuts and the global Esc chain out of it
  const trap = useFocusTrap(dialogRef, { onEscape: () => answer("cancel"), initialFocus: () => cancelRef.current, isolate: true });
  return (
    <div className="modal-backdrop confirm-backdrop" onMouseDown={() => answer("cancel")}>
      <div className="dialog confirm" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" aria-describedby="confirm-message" tabIndex={-1} ref={dialogRef} onMouseDown={(e) => e.stopPropagation()} onKeyDown={trap.onKeyDown}>
        <div className="dialog-head">
          {req.danger && <TriangleAlert size={16} className="confirm-icon" aria-hidden="true" />}
          <h3 id="confirm-title">{req.title}</h3>
        </div>
        <div className="dialog-body">
          <p id="confirm-message" className="confirm-message">{req.message}</p>
          {req.dontAskPref && (
            <label className="switch confirm-dont-ask">
              <input type="checkbox" checked={dontAsk} onChange={(e) => setDontAsk(e.target.checked)} /> Don't ask again (Preferences ▸ Saving &amp; editing)
            </label>
          )}
        </div>
        <div className="dialog-foot">
          <span className="grow" />
          <button ref={cancelRef} className="btn" onClick={() => answer("cancel")}>Cancel</button>
          {req.altLabel && <button className="btn" onClick={() => answer("alt")}>{req.altLabel}</button>}
          <button className={"btn " + (req.danger ? "danger-primary" : "primary")} onClick={() => answer("confirm")}>{req.confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}
