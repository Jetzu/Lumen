import { useEffect, useState } from "react";
import { Play, Pause } from "lucide-react";
import { useStore } from "../store";
import { lutColor } from "../render/compositor";
import type { ImageSummary } from "../types";

export default function DimensionBar({ img }: { img: ImageSummary }) {
  const setPosition = useStore((s) => s.setPosition);
  const setDisplay = useStore((s) => s.setDisplay);
  const luts = useStore((s) => s.luts);
  const d = img.display;
  const { z: nz, t: nt, c: nc } = img.shape;
  // Playback is bound to the image it was started on.  <DimensionBar> is mounted once per
  // <Viewport> / <SplitView> pane (no key), so this state outlives a tab switch; deriving the
  // effective axis here makes playback inert on the very render that shows another image (or an
  // image whose axis collapsed to a single plane, where there is no Pause button to reach), so
  // no interval is ever armed for it.
  const [play, setPlay] = useState<{ axis: "z" | "t"; id: string } | null>(null);
  const playing: "z" | "t" | null = play && play.id === img.id && (play.axis === "z" ? nz > 1 : nt > 1) ? play.axis : null;
  const setPlaying = (axis: "z" | "t" | null) => setPlay(axis ? { axis, id: img.id } : null);

  useEffect(() => {
    // image changed or the played axis is gone: drop the stale entry so switching back does not resume
    if (play && !playing) setPlay(null);
  }, [play, playing]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      const cur = useStore.getState().image(img.id);
      if (!cur) return;
      if (playing === "z") setPosition(img.id, { z: (cur.display.z + 1) % nz });
      else setPosition(img.id, { t: (cur.display.t + 1) % nt });
    }, 120);
    return () => window.clearInterval(id);
  }, [playing, img.id, nz, nt, setPosition]);

  if (nz <= 1 && nt <= 1 && (nc <= 1 || img.rgb)) return null;
  const cal = img.calibration;
  return (
    <div className="dimbar">
      {nc > 1 && !img.rgb && (
        <div className="dim-row">
          <span className="dim-label">C</span>
          <div className="chips" role="group" aria-label="Channels">
            {d.channels.map((ch, i) => {
              const on = d.mode === "composite" ? ch.visible : i === d.c;
              return (
                <button key={i} className={"chip " + (on ? "on" : "off") + (i === d.c ? " current" : "")} aria-pressed={on} title={d.mode === "composite" ? "Click: toggle visibility · Alt-click: make current" : "Click: show this channel"}
                  onClick={(e) => {
                    if (d.mode === "composite" && !e.altKey) {
                      const chans = d.channels.map((c, k) => (k === i ? { visible: !c.visible } : {}));
                      setDisplay(img.id, { channels: chans, c: i });
                    } else setDisplay(img.id, { c: i });
                  }}>
                  <span className="sw" style={{ background: lutColor(luts, ch.lut) }} />{ch.name || `C${i + 1}`}
                </button>
              );
            })}
            <span className="seg" role="group" aria-label="Channel display mode" style={{ marginLeft: 6 }}>
              {(["composite", "single", "grayscale"] as const).map((m) => (
                <button key={m} className={d.mode === m ? "on" : ""} aria-pressed={d.mode === m} onClick={() => setDisplay(img.id, { mode: m })} title={m === "composite" ? "Overlay all visible channels" : m === "single" ? "One channel with its LUT" : "One channel in gray"}>
                  {m === "composite" ? "Composite" : m === "single" ? "Color" : "Gray"}
                </button>
              ))}
            </span>
          </div>
        </div>
      )}
      {nz > 1 && (
        <div className="dim-row">
          <span className="dim-label">Z</span>
          <button className="icon-btn" style={{ width: 22, height: 22 }} title={playing === "z" ? "Pause" : "Play through slices"} aria-label={playing === "z" ? "Pause" : "Play through slices"} aria-pressed={playing === "z"} onClick={() => setPlaying(playing === "z" ? null : "z")}>{playing === "z" ? <Pause size={13} /> : <Play size={13} />}</button>
          <input type="range" aria-label="Z slice" aria-valuetext={`${d.z + 1} of ${nz}`} min={0} max={nz - 1} value={d.z} onChange={(e) => setPosition(img.id, { z: Number(e.target.value) })} />
          <span className="dim-val">{d.z + 1}/{nz}{cal.unit !== "pixel" ? ` · ${(d.z * cal.pixel_depth).toFixed(2)} ${cal.unit}` : ""}</span>
        </div>
      )}
      {nt > 1 && (
        <div className="dim-row">
          <span className="dim-label">T</span>
          <button className="icon-btn" style={{ width: 22, height: 22 }} title={playing === "t" ? "Pause" : "Play through frames"} aria-label={playing === "t" ? "Pause" : "Play through frames"} aria-pressed={playing === "t"} onClick={() => setPlaying(playing === "t" ? null : "t")}>{playing === "t" ? <Pause size={13} /> : <Play size={13} />}</button>
          <input type="range" aria-label="Time frame" aria-valuetext={`${d.t + 1} of ${nt}`} min={0} max={nt - 1} value={d.t} onChange={(e) => setPosition(img.id, { t: Number(e.target.value) })} />
          <span className="dim-val">{d.t + 1}/{nt}{cal.frame_interval ? ` · ${(d.t * cal.frame_interval).toFixed(1)} ${cal.time_unit}` : ""}</span>
        </div>
      )}
    </div>
  );
}
