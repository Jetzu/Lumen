/**
 * NumberInput – the one numeric field of the UI (GUI fix round 3, E7-26).
 *
 * `<input type="number">` formats its value with the BROWSER UI locale (a German Chrome shows "2,5") whatever `lang`
 * says, so the recorder, Results and the status bar ("2.5") disagreed with the fields.  This is a plain text field
 * (`inputMode="decimal"` brings up the numeric keypad on touch devices) that
 *   - shows the committed value with a decimal POINT (`String(v)`, trimmed, no exponent for ordinary magnitudes),
 *   - accepts "," as a decimal separator while typing ("2,5" → 2.5) and validates with a numeric regex,
 *   - keeps min / max / step: ↑ / ↓ step the value (⇧ ×10), clamped to the range; `aria-valuemin/max/now` are set
 *     (role=spinbutton, like a native number input), `aria-invalid` marks a non-numeric text or an external problem,
 *   - commits only finite numbers (the parent keeps its own range validation, inline message and disabled Apply),
 *   - restores the committed value on blur when the text is empty / not a number (or calls `onInvalidBlur`).
 *
 * The element carries `data-kind="number"` for styling (styles.css gives it the number-field width and monospace
 * font) and for tests: the old `input[type="number"]` selector becomes `input[inputmode="decimal"]` (or
 * `input[data-kind="number"]`).
 */
import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

/** A decimal number as typed: optional sign, digits with an optional point, optional exponent. */
export const NUMBER_RE = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

/** The number in a field's text ("2,5" → 2.5); null for an empty / partial / non-numeric text (or a non-integer when `integer`). */
export function parseNumberText(s: string, integer = false): number | null {
  const t = s.trim().replace(/\s+/g, "").replace(",", ".");
  if (!NUMBER_RE.test(t)) return null;
  const n = Number(t);
  if (!Number.isFinite(n)) return null;
  if (integer && !Number.isInteger(n)) return null;
  return n;
}

/** The text a committed value is shown as: always a decimal point, no exponent between 1e-6 and 1e15. */
export function formatNumber(v: number | null | undefined): string {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) return "";
  let s = String(v);
  if (/e/i.test(s) && Math.abs(v) < 1e15 && Math.abs(v) >= 1e-6) s = v.toFixed(12).replace(/\.?0+$/, "");
  return s;
}

function decimalsOf(n: number): number {
  const s = String(n);
  if (/e/i.test(s)) return 0;
  const i = s.indexOf(".");
  return i < 0 ? 0 : s.length - i - 1;
}

export interface NumberInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "type" | "min" | "max" | "step" | "inputMode"> {
  /** The committed value; null / undefined / NaN show an empty field. */
  value: number | null | undefined;
  /** Called with every finite number typed or stepped (never with partial text). */
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  /** ↑ / ↓ increment (⇧ ×10); "any" or undefined = 1. */
  step?: number | "any";
  /** Only whole numbers are committed. */
  integer?: boolean;
  /** Blur with an empty / non-numeric text: restore the committed value (default) or let the parent decide. */
  onInvalidBlur?: () => void;
  /** A problem the parent found (a display maximum below the minimum …): aria-invalid + the `.invalid` look. */
  invalid?: boolean;
  /** Called whenever the typed text starts or stops parsing as a number, so a form can refuse to submit "abc" (E10-3). */
  onValidity?: (ok: boolean) => void;
  /** The raw text and its parsed value on every change (for a parent that wants to validate partial input). */
  onText?: (text: string, parsed: number | null) => void;
}

const NumberInput = forwardRef<HTMLInputElement, NumberInputProps>(function NumberInput(
  { value, onChange, min, max, step, integer, onInvalidBlur, invalid, onText, onValidity, className, onBlur, onKeyDown, ...rest },
  ref,
) {
  const inputRef = useRef<HTMLInputElement>(null);
  useImperativeHandle(ref, () => inputRef.current as HTMLInputElement);
  const committed = typeof value === "number" && Number.isFinite(value) ? value : null;
  const [text, setText] = useState(() => formatNumber(committed));
  // follow external changes (a slider, Reset, a re-seeded sheet) without clobbering the text being typed ("1.0" vs 1)
  useEffect(() => {
    const cur = parseNumberText(text, integer);
    if (committed === null ? text.trim() !== "" && cur !== null : cur !== committed) setText(formatNumber(committed));
  }, [committed]); // eslint-disable-line react-hooks/exhaustive-deps
  const parsed = parseNumberText(text, integer);
  const textBad = text.trim() !== "" && parsed === null;
  const bad = !!invalid || textBad;
  const validityRef = useRef(onValidity);
  validityRef.current = onValidity;
  useEffect(() => {
    validityRef.current?.(!textBad);
  }, [textBad]);

  const stepBy = (dir: 1 | -1, big: boolean) => {
    const base = typeof step === "number" && step > 0 ? step : 1;
    const from = parsed ?? committed ?? (min !== undefined && min > 0 ? min : 0);
    let n = from + base * (big ? 10 : 1) * dir;
    // snap to the step's decimals so 0.1 + 0.2 never shows as 0.30000000000000004
    n = Number(n.toFixed(Math.min(12, Math.max(decimalsOf(base), decimalsOf(from)))));
    if (min !== undefined) n = Math.max(min, n);
    if (max !== undefined) n = Math.min(max, n);
    if (integer) n = Math.round(n);
    const s = formatNumber(n);
    setText(s);
    onText?.(s, n);
    onChange(n);
  };

  return (
    <input
      {...rest}
      ref={inputRef}
      type="text"
      inputMode="decimal"
      data-kind="number"
      role="spinbutton"
      autoComplete="off"
      spellCheck={false}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={parsed ?? committed ?? undefined}
      aria-invalid={bad || undefined}
      className={[className, bad ? "invalid" : ""].filter(Boolean).join(" ") || undefined}
      value={text}
      onChange={(e) => {
        const s = e.target.value;
        setText(s);
        const n = parseNumberText(s, integer);
        onText?.(s, n);
        if (n !== null) onChange(n);
      }}
      onKeyDown={(e) => {
        if ((e.key === "ArrowUp" || e.key === "ArrowDown") && !e.metaKey && !e.ctrlKey && !e.altKey) {
          e.preventDefault();
          e.stopPropagation(); // the viewport's arrow keys nudge the selection; a focused field keeps them
          stepBy(e.key === "ArrowUp" ? 1 : -1, e.shiftKey);
          return;
        }
        onKeyDown?.(e);
      }}
      onBlur={(e) => {
        if (parseNumberText(text, integer) === null) {
          if (onInvalidBlur) onInvalidBlur();
          else setText(formatNumber(committed));
        }
        onBlur?.(e);
      }}
    />
  );
});

export default NumberInput;
