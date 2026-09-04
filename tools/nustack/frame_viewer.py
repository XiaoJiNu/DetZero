#!/usr/bin/env python3
"""Frame-by-frame image flipbook viewer: Space/Right = next, Left/BackSpace =
prev, Home/End jump, digits 0-9 jump to 0-90%, q/Esc quit. Shows any dir of
sorted PNGs (visuals/, visuals_id_check/, ...)."""

import argparse
import sys
from pathlib import Path
import tkinter as tk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    pngs = sorted(args.dir.glob("*.png"))
    if not pngs:
        print(f"no pngs in {args.dir}", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title(str(pngs[0]))
    label = tk.Label(root, bg="black")
    label.pack()
    state = {"i": min(max(args.start, 0), len(pngs) - 1)}
    # preload keeps frames cached; images are ~450KB so full set stays RAM-able
    def show(i):
        state["i"] = i % len(pngs)
        img = tk.PhotoImage(file=str(pngs[state["i"]]))
        label.configure(image=img, width=img.width(), height=img.height())
        label.image = img
        root.title(f"{pngs[state['i']].name}  [{state['i'] + 1}/{len(pngs)}]  Space=next")

    root.bind("<space>", lambda e: show(state["i"] + 1))
    root.bind("<Right>", lambda e: show(state["i"] + 1))
    root.bind("<Left>", lambda e: show(state["i"] - 1))
    root.bind("<BackSpace>", lambda e: show(state["i"] - 1))
    root.bind("<Home>", lambda e: show(0))
    root.bind("<End>", lambda e: show(len(pngs) - 1))
    for row in ("<KP_0>", "<KP_1>", "<KP_2>", "<KP_3>", "<KP_4>",
                "<KP_5>", "<KP_6>", "<KP_7>", "<KP_8>", "<KP_9>"):
        root.bind(row, lambda e, n=int(row[-2]): show(n * len(pngs) // 10))
    root.bind("q", lambda e: root.destroy())
    root.bind("<Escape>", lambda e: root.destroy())
    show(state["i"])
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
