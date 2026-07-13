#!/usr/bin/env python3
"""
Simple Image Labeler  -  a LabelImg-style bounding box annotation tool.

Features
--------
- Pick an input image folder and an output (labels) folder.
- Draw / move / resize bounding boxes with the mouse.
- Label list with history: labels you create persist in a dropdown and
  carry over. Editing a box's label adds the new label to history.
- "Carry over" mode: boxes from the previous image are auto-copied onto a
  new (un-annotated) image so you keep the same boxes across a sequence
  until you edit them.
- Copy / paste boxes across images (Ctrl+C / Ctrl+V). If exactly one box is
  selected, Ctrl+C copies only that box; otherwise it copies all of them.
  The copy buffer PERSISTS — copy once, then paste as many times and across
  as many images as you like; it only changes when you copy again. Repeated
  pastes onto the same image are nudged slightly so they don't stack unseen.
- Drawing always wins over a bigger box: dragging with a label set creates a
  new box even on top of a larger one. Click a box once to select it, then
  drag its interior to move (or grab a corner to resize). Esc deselects.
- Classes never leak between folders: choosing a new input folder starts a
  fresh class list, read only from that folder's classes.txt (if present).
- Fast navigation: A / D or arrow keys, Space = next.
- Save formats: YOLO (default), Pascal VOC (.xml), and a plain VOC-style
  txt is folded into YOLO. Label file name == image name.
- Compare mode: open a second folder of image+label pairs and view them
  side-by-side, matched by file name.

Run:  python labeler.py
Needs: Python 3.8+  and  Pillow  (pip install pillow)
"""

import os
import json
import shutil
import threading
from collections import OrderedDict
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Pillow is required. Install with:  pip install pillow")


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
HANDLE = 7          # pixel radius of resize handles
MIN_BOX = 5         # smallest box in image pixels
CONFIG = os.path.join(os.path.expanduser("~"), ".simple_labeler.json")

PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]


def color_for(label, labels):
    """Stable color per label."""
    if label not in labels:
        labels.append(label)
    return PALETTE[labels.index(label) % len(PALETTE)]


class Box:
    """A bounding box in IMAGE pixel coordinates."""
    __slots__ = ("x1", "y1", "x2", "y2", "label", "group")

    def __init__(self, x1, y1, x2, y2, label, group=None):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.label = label
        self.group = group          # shared int id -> grouped; None -> standalone

    def norm(self):
        return (min(self.x1, self.x2), min(self.y1, self.y2),
                max(self.x1, self.x2), max(self.y1, self.y2))

    def copy(self):
        return Box(self.x1, self.y1, self.x2, self.y2, self.label, self.group)


class Labeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple Image Labeler")
        self.geometry("1280x820")

        # state
        self.input_dir = ""
        self.output_dir = ""
        self.output_explicit = False  # True once user picks a distinct output
        self.compare_dir = ""
        self.images = []
        self.index = -1
        self.img = None             # PIL image
        self.tkimg = None
        self.scale = 1.0            # effective scale = fit_scale * zoom
        self.fit_scale = 1.0        # scale that fits image to canvas
        self.zoom = 1.0             # user zoom multiplier (1.0 = fit)
        self.offx = self.offy = 0
        self.panning = False
        self.pan_start = (0, 0)
        self.space_held = False     # spacebar -> temporary hand/pan tool
        self.space_pan_used = False # True once a space-drag actually pans
        self.boxes = []             # list[Box]
        self.label_history = []     # persists across images
        self.current_label = ""
        self.fmt = tk.StringVar(value="YOLO")
        self.carry = tk.BooleanVar(value=False)  # default: no auto-copy of boxes
        self.autosave = tk.BooleanVar(value=True)  # save on image change
        self.clipboard = []         # copied boxes — PERSISTS until you copy again
        self.paste_count = 0        # how many times pasted onto the current image
        self.paste_last_index = None
        self.annotated = set()      # image names that have been saved
        # --- navigation speed: decoded-image cache + background prefetch ---
        self._img_cache = OrderedDict()   # path -> PIL.Image (RGB), LRU
        self._cache_cap = 12
        self._cache_lock = threading.Lock()
        self._snapshot = None       # box-state at load; used to skip no-op saves

        # interaction
        self.action = None          # 'draw' | 'move' | 'resize'
        self.sel = None             # primary selected box index (for handles)
        self.selected = set()       # ALL selected box indices (multi-select)
        self._next_group = 1        # next group id to hand out
        self.mouse_img = None       # last cursor position in image coords
        self.handle = None          # which handle when resizing
        self.start = (0, 0)
        self.temp = None
        self._pending_box = None    # box under an undecided press (click vs drag)
        self._pending_shift = False
        self._press_screen = (0, 0) # screen point where the press began

        self._load_config()
        self._build_ui()
        self._bind_keys()
        if self.output_dir:
            self.output_lbl.config(text="Output: " + self.output_dir)

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=4)
        top.pack(side="top", fill="x")

        ttk.Button(top, text="Input Folder", command=self.pick_input).pack(side="left", padx=2)
        ttk.Button(top, text="Output Folder", command=self.pick_output).pack(side="left", padx=2)
        ttk.Label(top, text="Format:").pack(side="left", padx=(12, 2))
        fmt = ttk.Combobox(top, textvariable=self.fmt, width=10, state="readonly",
                           values=["YOLO", "Pascal VOC"])
        fmt.pack(side="left")
        ttk.Checkbutton(top, text="Auto-copy prev boxes", variable=self.carry).pack(side="left", padx=12)
        ttk.Checkbutton(top, text="Auto-save on switch", variable=self.autosave).pack(side="left", padx=4)
        ttk.Button(top, text="◀ Prev (A)", command=self.prev_img).pack(side="left", padx=2)
        ttk.Button(top, text="Next ▶ (D)", command=self.next_img).pack(side="left", padx=2)
        ttk.Button(top, text="Save (Ctrl+S)", command=self.save).pack(side="left", padx=8)
        ttk.Button(top, text="Compare Folder", command=self.pick_compare).pack(side="left", padx=2)
        ttk.Label(top, text="Zoom:").pack(side="left", padx=(12, 2))
        ttk.Button(top, text="−", width=3, command=self.zoom_out).pack(side="left")
        ttk.Button(top, text="+", width=3, command=self.zoom_in).pack(side="left")
        ttk.Button(top, text="Fit", width=4, command=self.zoom_reset).pack(side="left", padx=2)
        self.pan_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="✋ Pan (H)", variable=self.pan_mode,
                        command=self._update_cursor).pack(side="left", padx=6)

        # second row: show where labels are being saved
        row2 = ttk.Frame(self, padding=(6, 0, 6, 4))
        row2.pack(side="top", fill="x")
        self.output_lbl = ttk.Label(row2, text="Output: (choose an input folder)",
                                    foreground="#555")
        self.output_lbl.pack(side="left")

        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)

        # left: canvas wrapped with scrollbars (a reliable way to move around
        # a zoomed image on any platform/trackpad)
        cwrap = ttk.Frame(body)
        cwrap.pack(side="left", fill="both", expand=True)
        self.hbar = ttk.Scrollbar(cwrap, orient="horizontal", command=self._xscroll)
        self.vbar = ttk.Scrollbar(cwrap, orient="vertical", command=self._yscroll)
        self.hbar.pack(side="bottom", fill="x")
        self.vbar.pack(side="right", fill="y")
        self.canvas = tk.Canvas(cwrap, bg="#222", cursor="cross", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        # grab keyboard focus when the pointer is over the canvas, so the
        # space/arrow pan shortcuts always reach it
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        self.canvas.bind("<ButtonPress-1>", lambda e: self.on_down(e, False))
        self.canvas.bind("<Shift-ButtonPress-1>", lambda e: self.on_down(e, True))
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Shift-B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Shift-ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Motion>", self.on_hover)
        # scroll: plain = pan, Cmd/Ctrl = zoom (Win/Mac); Button-4/5 = Linux vert
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)
        # window-level fallback: some macOS trackpad wheels arrive at the toplevel
        # rather than the canvas. on_wheel only acts when the pointer is over the
        # canvas, so this won't hijack scrolling of the file/box lists.
        self.bind("<MouseWheel>", self.on_wheel)
        self.bind("<Shift-MouseWheel>", self.on_wheel)
        # Linux horizontal scroll buttons (not valid on macOS/Windows Tk)
        for seq in ("<Button-6>", "<Button-7>"):
            try:
                self.canvas.bind(seq, self.on_hscroll)
            except tk.TclError:
                pass
        # pan also via middle-drag or right-drag (in addition to space+drag)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<ButtonRelease-3>", self.on_pan_end)

        # compare canvas (hidden until used)
        self.cmp_canvas = tk.Canvas(body, bg="#111", width=380, highlightthickness=0)

        # right: panel
        right = ttk.Frame(body, width=300, padding=4)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ttk.Label(right, text="Current label (for new boxes):").pack(anchor="w")
        self.label_box = ttk.Combobox(right, values=self.label_history)
        self.label_box.pack(fill="x")
        self.label_box.bind("<<ComboboxSelected>>", self.on_pick_label)
        self.label_box.bind("<Return>", self.on_new_label)
        ttk.Button(right, text="Set / Add label", command=self.on_new_label).pack(fill="x", pady=2)
        # live "what am I labeling with / what's selected" indicator
        self.cur_label_lbl = ttk.Label(right, text="Active label: (none)",
                                       foreground="#cc0000")
        self.cur_label_lbl.pack(anchor="w", pady=(0, 2))
        # type-to-search: "m" -> "mou" narrows classes.txt entries live
        self._attach_search(self.label_box, lambda: self.label_history)

        ttk.Separator(right).pack(fill="x", pady=6)
        ttk.Label(right, text="Boxes in this image:").pack(anchor="w")
        self.box_list = tk.Listbox(right, height=14, exportselection=False,
                                   activestyle="none", selectmode="extended",
                                   selectbackground="#2d6cdf",
                                   selectforeground="white")
        self.box_list.pack(fill="both", expand=False)
        self.box_list.bind("<<ListboxSelect>>", self.on_select_box)

        bb = ttk.Frame(right)
        bb.pack(fill="x", pady=2)
        ttk.Button(bb, text="Edit label", command=self.edit_box_label).pack(side="left", expand=True, fill="x")
        ttk.Button(bb, text="Delete (⌫)", command=self.delete_box).pack(side="left", expand=True, fill="x")

        ttk.Separator(right).pack(fill="x", pady=6)
        ttk.Label(right, text="Image list:").pack(anchor="w")
        self.file_list = tk.Listbox(right, height=14, exportselection=False,
                                    activestyle="none",
                                    selectbackground="#2d6cdf",
                                    selectforeground="white")
        self.file_list.pack(fill="both", expand=True)
        self.file_list.bind("<<ListboxSelect>>", self.on_pick_file)

        self.status = ttk.Label(self, text="Pick an input folder to begin.",
                                relief="sunken", anchor="w", padding=3)
        self.status.pack(side="bottom", fill="x")

    def _typing(self):
        """True if a text entry currently has keyboard focus."""
        return isinstance(self.focus_get(),
                          (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text))

    _SKIP_KEYS = {"Return", "Escape", "Up", "Down", "Left", "Right", "Tab",
                  "Shift_L", "Shift_R", "Control_L", "Control_R",
                  "Meta_L", "Meta_R", "Alt_L", "Alt_R", "Super_L", "Super_R"}

    def _attach_search(self, combo, source_fn):
        """Make a ttk.Combobox filter its list as you type: 'm' -> 'mou' narrows
        to matching classes (case-insensitive substring). Typing a brand-new
        name still works (free text)."""
        def on_key(e):
            if e.keysym in self._SKIP_KEYS:
                return
            typed = combo.get()
            src = list(source_fn())
            matches = [v for v in src if typed.lower() in v.lower()] if typed else src
            combo["values"] = matches if matches else src
            if typed and matches:
                try:                              # live-open the filtered dropdown
                    combo.tk.call("ttk::combobox::Post", combo)
                    combo.set(typed)              # Post may pick item 0; keep text
                    combo.icursor("end")
                except Exception:
                    pass                          # filtering still works via arrow
        combo.bind("<KeyRelease>", on_key, add="+")

    def _ci(self, fn):
        """Wrap a key handler so it still fires when Caps Lock is on (we bind the
        uppercase keysym to the same wrapper) and warns loudly the first time."""
        def handler(e):
            if (e.state & 0x0002) and not getattr(self, "_caps_warned", False):
                self._caps_warned = True
                self.status.config(text="\u26a0 CAPS LOCK IS ON \u2014 shortcuts "
                                        "still work, but you may want to turn it off.")
                try:
                    messagebox.showwarning(
                        "Caps Lock is on",
                        "Caps Lock is ON.\n\nYour shortcuts (A / D / H / S, copy, "
                        "paste, save\u2026) still work with it on \u2014 but you may "
                        "want to turn it off.")
                except Exception:
                    pass
            return fn(e)
        return handler

    def _bind_ci(self, letter, fn, mods=("",)):
        """Bind fn to <mod+letter> in BOTH lower- and upper-case keysyms (so it
        works with Caps Lock) across each modifier prefix in `mods`."""
        wrapped = self._ci(fn)
        for m in mods:
            self.bind(f"<{m}{letter.lower()}>", wrapped)
            self.bind(f"<{m}{letter.upper()}>", wrapped)

    def _update_cur_label(self):
        """Show what new boxes get labeled, or the selected box's label."""
        if not hasattr(self, "cur_label_lbl"):
            return
        n = len(self.selected)
        if n == 1:
            i = next(iter(self.selected))
            self.cur_label_lbl.config(text=f"Selected box: {self.boxes[i].label}")
        elif n > 1:
            labs = {self.boxes[i].label for i in self.selected}
            txt = next(iter(labs)) if len(labs) == 1 else f"{len(labs)} different"
            self.cur_label_lbl.config(text=f"{n} selected \u2192 {txt}")
        else:
            self.cur_label_lbl.config(
                text=f"Active label: {self.current_label or '(none)'}")

    def _nav(self, fn):
        """Wrap a shortcut so it's ignored while typing in a text field."""
        def handler(e):
            if self._typing():
                return            # let the keystroke go to the text box
            fn()
            return "break"
        return handler

    def _bind_keys(self):
        self._caps_warned = False
        # image nav — A / D, plain AND Ctrl/Cmd, upper/lower (Caps-Lock-proof)
        self._bind_ci("a", self._nav(self.prev_img), mods=("", "Control-", "Command-"))
        self._bind_ci("d", self._nav(self.next_img), mods=("", "Control-", "Command-"))
        self.bind("<Left>", self._nav(lambda: self.arrow(-1, 0)))
        self.bind("<Right>", self._nav(lambda: self.arrow(1, 0)))
        self.bind("<Up>", self._nav(lambda: self.arrow(0, -1)))
        self.bind("<Down>", self._nav(lambda: self.arrow(0, 1)))
        self.bind("<KeyPress-space>", self.on_space_down)
        self.bind("<KeyRelease-space>", self.on_space_up)
        # save / copy / paste — Ctrl+Cmd, upper/lower (Caps-Lock-proof)
        self._bind_ci("s", lambda e: self.save(), mods=("Control-", "Command-"))
        self._bind_ci("c", lambda e: self.copy_boxes(), mods=("Control-", "Command-"))
        self._bind_ci("v", lambda e: self.paste_boxes(), mods=("Control-", "Command-"))
        self.bind("<Command-0>", lambda e: self.zoom_reset())
        # group / ungroup (Cmd/Ctrl+G toggles) — Caps-Lock-proof
        self._bind_ci("g", self._nav(self.toggle_group), mods=("Control-", "Command-"))
        for seq in ("<Control-Shift-G>", "<Control-Shift-g>",
                    "<Command-Shift-G>", "<Command-Shift-g>"):
            self.bind(seq, self._nav(self.ungroup_selected))
        self.bind("<Delete>", lambda e: self.delete_box())
        self.bind("<BackSpace>", self._nav(self.delete_box))
        self.bind("<Escape>", lambda e: self.deselect())
        # W = jump to label box to type a new label; H = toggle pan/hand tool
        self._bind_ci("w", self.focus_new_label, mods=("",))
        self._bind_ci("h", self.toggle_pan_mode, mods=("", "Control-", "Command-"))
        # zoom keys: + / = zoom in, - zoom out, Ctrl+0 reset
        self.bind("<plus>", self._nav(self.zoom_in))
        self.bind("<KP_Add>", self._nav(self.zoom_in))
        self.bind("<equal>", self._nav(self.zoom_in))
        self.bind("<minus>", self._nav(self.zoom_out))
        self.bind("<KP_Subtract>", self._nav(self.zoom_out))
        self.bind("<Control-0>", lambda e: self.zoom_reset())

    # ---------------- folders ----------------
    def pick_input(self):
        start = self.input_dir or os.path.expanduser("~")
        d = filedialog.askdirectory(title="Select input image folder",
                                    initialdir=start)
        if not d:
            return
        self.input_dir = d
        # default output to the same folder as input (LabelImg-style) — but only
        # if the user hasn't explicitly chosen a separate output folder. This is
        # what previously made the output stick to input no matter what.
        if not self.output_explicit:
            self.output_dir = d
            self.output_lbl.config(text="Output: " + d)
        # --- fresh start: never carry classes/boxes from a previous folder ---
        self.index = -1
        self.label_history = []
        self.current_label = ""
        self.clipboard = []
        self.boxes = []
        self.sel = None
        self.selected = set()
        # adopt only the classes that already belong to THIS folder, if any
        self._load_classes_file(self.output_dir)
        self.label_box.config(values=self.label_history)
        self.label_box.set("")
        self.images = sorted(f for f in os.listdir(d)
                             if f.lower().endswith(IMG_EXTS))
        self.file_list.delete(0, "end")
        for f in self.images:
            self.file_list.insert("end", f)
        self._scan_annotated()
        self.index = -1
        if self.images:
            self.load_image(0)
        self._save_config()

    def pick_output(self):
        start = self.output_dir or self.input_dir or os.path.expanduser("~")
        d = filedialog.askdirectory(title="Select output (labels) folder",
                                    initialdir=start)
        if d:
            self.output_dir = d
            self.output_explicit = True
            self.output_lbl.config(text="Output: " + d)
            self._scan_annotated()
            self.redraw()
            self._save_config()

    def pick_compare(self):
        start = self.input_dir or os.path.expanduser("~")
        d = filedialog.askdirectory(title="Select folder with image+label pairs to compare",
                                    initialdir=start)
        if not d:
            return
        self.compare_dir = d
        self.cmp_canvas.pack(side="right", fill="y", before=None)
        self.cmp_canvas.pack(side="left", fill="y")
        self.redraw()

    # ---------------- image loading ----------------
    def _scan_annotated(self):
        self.annotated.clear()
        if not self.output_dir:
            return
        for f in os.listdir(self.output_dir):
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".txt", ".xml"):
                self.annotated.add(stem)
        self._refresh_file_marks()

    def _refresh_file_marks(self):
        for i, f in enumerate(self.images):
            stem = os.path.splitext(f)[0]
            mark = "● " if stem in self.annotated else "   "
            self.file_list.delete(i)
            self.file_list.insert(i, mark + f)
        # keep the row of the image we're on highlighted
        if 0 <= self.index < len(self.images):
            self.file_list.selection_clear(0, "end")
            self.file_list.selection_set(self.index)
            self.file_list.see(self.index)

    def _mark_one(self, i):
        """Update just ONE file-list row's ●/space mark (fast — avoids the
        full-list rebuild on every image switch, which is the main nav lag)."""
        if not (0 <= i < len(self.images)):
            return
        f = self.images[i]
        stem = os.path.splitext(f)[0]
        mark = "● " if stem in self.annotated else "   "
        try:
            sel = i in self.file_list.curselection()
            self.file_list.delete(i)
            self.file_list.insert(i, mark + f)
            if sel:
                self.file_list.selection_set(i)
        except Exception:
            pass

    # ---- decoded-image cache + prefetch (fast back/forward navigation) ----
    def _decode(self, path):
        with self._cache_lock:
            im = self._img_cache.get(path)
            if im is not None:
                self._img_cache.move_to_end(path)
                return im
        im = Image.open(path).convert("RGB")
        with self._cache_lock:
            self._img_cache[path] = im
            self._img_cache.move_to_end(path)
            while len(self._img_cache) > self._cache_cap:
                self._img_cache.popitem(last=False)
        return im

    def _prefetch(self, indices):
        """Decode neighbour images in the background so the next switch is instant."""
        def work():
            for j in indices:
                if 0 <= j < len(self.images):
                    p = os.path.join(self.input_dir, self.images[j])
                    with self._cache_lock:
                        have = p in self._img_cache
                    if not have:
                        try:
                            self._decode(p)
                        except Exception:
                            pass
        threading.Thread(target=work, daemon=True).start()

    def _box_state(self):
        """Cheap fingerprint of the current labels; used to skip no-op saves."""
        return (tuple((round(b.x1, 2), round(b.y1, 2), round(b.x2, 2),
                       round(b.y2, 2), b.label, b.group) for b in self.boxes),
                tuple(self.label_history))

    def load_image(self, i):
        if not (0 <= i < len(self.images)):
            return
        # auto-save current before leaving (if enabled).
        # Save regardless of whether boxes exist, so deletions / emptied images
        # are persisted too. Triggered by next, prev, AND file-list clicks.
        if self.index >= 0 and self.autosave.get():
            self.save(silent=True)

        prev_boxes = [b.copy() for b in self.boxes]
        self.index = i
        path = os.path.join(self.input_dir, self.images[i])
        self.img = self._decode(path)          # cached decode (fast on revisit)
        self.boxes = []

        # try to load existing labels
        loaded = self._load_existing(self.images[i])
        if not loaded and self.carry.get() and prev_boxes:
            # carry over boxes from previous image
            self.boxes = [b.copy() for b in prev_boxes]

        self._snapshot = self._box_state()     # baseline to detect real edits
        self.file_list.selection_clear(0, "end")
        self.file_list.selection_set(i)
        self.file_list.see(i)
        self.sel = None
        self.selected = set()
        self.mouse_img = None       # so a paste before hovering uses copied pos
        self.fit()
        self.refresh_box_list()
        self.redraw()
        self.status.config(text=f"{self.images[i]}  ({i+1}/{len(self.images)})  "
                                 f"{self.img.width}x{self.img.height}")
        self._prefetch([i + 1, i + 2, i - 1])  # warm neighbours in background

    def fit(self):
        """Reset view: fit whole image to canvas and clear zoom/pan."""
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        self.fit_scale = min(cw / self.img.width, ch / self.img.height, 1.0)
        if self.fit_scale <= 0:
            self.fit_scale = 1.0
        self.zoom = 1.0
        self.scale = self.fit_scale
        dispw = int(self.img.width * self.scale)
        disph = int(self.img.height * self.scale)
        self.offx = (cw - dispw) // 2
        self.offy = (ch - disph) // 2
        self._render_image()

    def _render_image(self):
        """(Re)build the scaled bitmap for the current self.scale."""
        dispw = max(1, int(self.img.width * self.scale))
        disph = max(1, int(self.img.height * self.scale))
        # LANCZOS for downscale, NEAREST when zoomed way in (faster, crisp pixels)
        resample = Image.NEAREST if self.scale > 3 else Image.LANCZOS
        self.tkimg = ImageTk.PhotoImage(self.img.resize((dispw, disph), resample))

    def set_zoom(self, new_zoom, anchor=None):
        """Zoom about a screen point (anchor); keeps that point fixed."""
        new_zoom = max(0.1, min(new_zoom, 20.0))
        if anchor is None:
            anchor = (self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        ax, ay = anchor
        # image coord under the anchor before zoom
        ix, iy = self.to_image(ax, ay)
        self.zoom = new_zoom
        self.scale = self.fit_scale * self.zoom
        # keep (ix,iy) under the anchor after zoom
        self.offx = ax - ix * self.scale
        self.offy = ay - iy * self.scale
        self._render_image()
        self._clamp_pan()
        self.redraw()
        self.status.config(text=f"Zoom {int(self.scale * 100)}%   "
                                f"scroll = pan · ⌘/Ctrl+scroll = zoom · "
                                f"space+drag = pan · Fit/Ctrl+0 to reset")

    def zoom_in(self):
        self.set_zoom(self.zoom * 1.25)

    def zoom_out(self):
        self.set_zoom(self.zoom / 1.25)

    def zoom_reset(self):
        self.fit()
        self.redraw()

    def on_resize(self, e):
        """Canvas resized: refit but keep the user's current zoom factor."""
        if not self.img:
            return
        z = self.zoom
        self.fit()            # recomputes fit_scale, resets zoom to 1
        if abs(z - 1.0) > 1e-3:
            self.set_zoom(z)  # re-apply previous zoom, centered
        else:
            self.redraw()

    def on_wheel(self, e):
        """Trackpad two-finger scroll / mouse wheel.

        Plain scroll / pinch -> PAN the image (natural on a Mac trackpad).
        Cmd/Ctrl + scroll    -> ZOOM about the pointer.
        Linux/X11 sends Button-4/5 instead of <MouseWheel>.
        """
        if not self.img:
            return
        # only act when the pointer is over the image canvas, so wheel-scrolling
        # the file/box lists still scrolls them instead of moving the image
        px = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        py = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if not (0 <= px <= cw and 0 <= py <= ch):
            return                      # let the widget under the pointer handle it
        mod = bool(e.state & 0x4) or bool(e.state & 0x40000) or bool(e.state & 0x8)
        num = getattr(e, "num", None)
        if num == 4:
            delta = 1
        elif num == 5:
            delta = -1
        else:
            d = getattr(e, "delta", 0)
            if not d:
                return "break"          # ignore zero-delta noise
            delta = 1 if d > 0 else -1

        if mod:                              # ZOOM about the pointer (Ctrl/Cmd+scroll)
            self.set_zoom(self.zoom * (1.15 ** delta), anchor=(px, py))
        else:                                # PAN
            step = 60 * delta
            if e.state & 0x1:  # Shift -> horizontal
                self.offx += step
            else:
                self.offy += step
            self._clamp_pan()
            self.redraw()
        return "break"

    def on_hscroll(self, e):
        """Horizontal two-finger swipe on trackpads that emit Shift-MouseWheel
        or a separate horizontal event."""
        if not self.img:
            return
        delta = 1 if getattr(e, "delta", 0) > 0 or getattr(e, "num", None) == 6 else -1
        self.offx += 60 * delta
        self._clamp_pan()
        self.redraw()

    def on_pan_start(self, e):
        self.panning = True
        self.pan_start = (e.x, e.y)
        self.canvas.config(cursor="fleur")

    def on_pan_move(self, e):
        if not self.panning:
            return
        dx = e.x - self.pan_start[0]
        dy = e.y - self.pan_start[1]
        self.offx += dx
        self.offy += dy
        self.pan_start = (e.x, e.y)
        self._clamp_pan()
        self.redraw()

    def on_pan_end(self, e):
        self.panning = False
        self.canvas.config(cursor="cross")

    def _clamp_pan(self):
        """Keep the image from being dragged completely off-screen.
        Purely a view limit; never touches box/image coordinates."""
        if not self.img:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        dispw = self.img.width * self.scale
        disph = self.img.height * self.scale
        margin = 40  # allow a little empty space past each edge
        if dispw <= cw:
            self.offx = (cw - dispw) / 2   # center when it fits
        else:
            self.offx = min(margin, max(cw - dispw - margin, self.offx))
        if disph <= ch:
            self.offy = (ch - disph) / 2
        else:
            self.offy = min(margin, max(ch - disph - margin, self.offy))
        self._sync_scrollbars()

    def _sync_scrollbars(self):
        """Update scrollbar thumb position/size from current view."""
        if not self.img:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        dispw = self.img.width * self.scale
        disph = self.img.height * self.scale
        # fraction of the image visible, and where the view starts
        if dispw > 0:
            x0 = max(0.0, -self.offx / dispw)
            x1 = min(1.0, (cw - self.offx) / dispw)
            self.hbar.set(x0, x1)
        if disph > 0:
            y0 = max(0.0, -self.offy / disph)
            y1 = min(1.0, (ch - self.offy) / disph)
            self.vbar.set(y0, y1)

    def _xscroll(self, *args):
        if not self.img:
            return
        dispw = self.img.width * self.scale
        if args[0] == "moveto":
            self.offx = -float(args[1]) * dispw
        elif args[0] == "scroll":
            self.offx -= int(args[1]) * (40 if args[2] == "units" else 200)
        self._clamp_pan()
        self.redraw()

    def _yscroll(self, *args):
        if not self.img:
            return
        disph = self.img.height * self.scale
        if args[0] == "moveto":
            self.offy = -float(args[1]) * disph
        elif args[0] == "scroll":
            self.offy -= int(args[1]) * (40 if args[2] == "units" else 200)
        self._clamp_pan()
        self.redraw()

    def _update_cursor(self):
        if self.pan_mode.get():
            self.canvas.config(cursor="fleur")
        else:
            self.canvas.config(cursor="cross")

    def toggle_pan_mode(self, e=None):
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            return  # let 'h' type normally in text fields
        self.pan_mode.set(not self.pan_mode.get())
        self._update_cursor()
        self.status.config(text="Pan/hand tool: "
                                + ("ON — drag to move image" if self.pan_mode.get()
                                   else "OFF — drag to draw boxes"))
        return "break"

    def on_space_down(self, e):
        """Hold space to turn the left mouse button into a pan/hand tool."""
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            return  # typing a space in a text field
        if not self.space_held:
            self.space_held = True
            self.space_pan_used = False
            self.canvas.config(cursor="fleur")
        return "break"

    def on_space_up(self, e):
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            return
        # a quick space *tap* (no panning) = go to next image; a space-drag
        # used as the pan tool must NOT advance the image on release
        panned = self.space_pan_used
        self.space_held = False
        self.space_pan_used = False
        self.panning = False
        self.canvas.config(cursor="cross")
        if not panned:
            self.next_img()
        return "break"

    def arrow(self, dx, dy):
        """Arrow keys: pan when zoomed in, otherwise change image."""
        if self.img and self.zoom > 1.001:
            self.offx -= dx * 80
            self.offy -= dy * 80
            self._clamp_pan()
            self.redraw()
        else:
            if dx < 0:
                self.prev_img()
            elif dx > 0:
                self.next_img()

    # ---------------- coordinate transforms ----------------
    def to_screen(self, x, y):
        return self.offx + x * self.scale, self.offy + y * self.scale

    def to_image(self, sx, sy):
        ix = (sx - self.offx) / self.scale
        iy = (sy - self.offy) / self.scale
        # clamp to image bounds so clicks in the surrounding margin can never
        # produce negative or out-of-range coordinates
        ix = max(0.0, min(ix, float(self.img.width)))
        iy = max(0.0, min(iy, float(self.img.height)))
        return ix, iy

    # ---------------- drawing ----------------
    def redraw(self):
        self.canvas.delete("all")
        if not self.img:
            return
        self._sync_scrollbars()
        self.canvas.create_image(self.offx, self.offy, anchor="nw", image=self.tkimg)
        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = b.norm()
            sx1, sy1 = self.to_screen(x1, y1)
            sx2, sy2 = self.to_screen(x2, y2)
            col = color_for(b.label, self.label_history)
            sel = i in self.selected
            w = 3 if sel else 2
            dash = (3, 2) if b.group is not None else ()
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=col, width=w,
                                         dash=dash)
            tag = "▣ " if b.group is not None else ""
            txt = tag + b.label
            self.canvas.create_rectangle(sx1, sy1 - 16, sx1 + 8 + 7 * len(txt), sy1,
                                         fill=col, outline=col)
            self.canvas.create_text(sx1 + 3, sy1 - 8, anchor="w", text=txt,
                                    fill="white", font=("TkDefaultFont", 9, "bold"))
            # corner squares on every selected box (resize handles when it's the
            # only selection; selection indicators when several are selected)
            if sel:
                for hx, hy in ((sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)):
                    self.canvas.create_rectangle(hx - HANDLE, hy - HANDLE,
                                                 hx + HANDLE, hy + HANDLE,
                                                 fill="white", outline=col)
        if self.temp:
            self.canvas.create_rectangle(*self.temp, outline="#ffff00", width=2, dash=(4, 3))
        self._draw_compare()
        self._update_cur_label()

    def _draw_compare(self):
        if not self.compare_dir or not self.img:
            return
        self.cmp_canvas.delete("all")
        name = self.images[self.index]
        path = self._match_in(self.compare_dir, name)
        if not path:
            self.cmp_canvas.create_text(190, 30, text="no match: " + name,
                                        fill="#999")
            return
        cimg = Image.open(path).convert("RGB")
        cw = self.cmp_canvas.winfo_width() or 380
        ch = self.cmp_canvas.winfo_height() or 600
        s = min(cw / cimg.width, ch / cimg.height, 1.0)
        dw, dh = int(cimg.width * s), int(cimg.height * s)
        ox, oy = (cw - dw) // 2, 10
        self._cmp_tk = ImageTk.PhotoImage(cimg.resize((dw, dh), Image.LANCZOS))
        self.cmp_canvas.create_image(ox, oy, anchor="nw", image=self._cmp_tk)
        # draw its labels if a matching label file exists
        for b in self._load_pairs(self.compare_dir, name, cimg.width, cimg.height):
            x1, y1, x2, y2 = b.norm()
            self.cmp_canvas.create_rectangle(ox + x1 * s, oy + y1 * s,
                                             ox + x2 * s, oy + y2 * s,
                                             outline="#3cf", width=2)
            self.cmp_canvas.create_text(ox + x1 * s + 2, oy + y1 * s - 6,
                                        anchor="w", text=b.label, fill="#3cf")

    # ---------------- mouse ----------------
    def on_down(self, e, shift=False):
        if not self.img:
            return
        if self.space_held or self.pan_mode.get():   # hand tool = pan
            self.panning = True
            self.pan_start = (e.x, e.y)
            if self.space_held:
                self.space_pan_used = True   # not a tap -> don't advance on space-up
            self.canvas.config(cursor="fleur")
            return
        # any click on the image takes keyboard focus away from the label box,
        # so it stops "editing itself" and a/d/arrows act as shortcuts again
        self.canvas.focus_set()
        ix, iy = self.to_image(e.x, e.y)
        self.mouse_img = (ix, iy)
        self.start = (ix, iy)
        self._press_screen = (e.x, e.y)
        self._pending_shift = shift or bool(e.state & 0x0001)   # Shift held?
        # 1) grab a resize handle — only when exactly one box is selected
        if (self.sel is not None and len(self.selected) <= 1
                and not self._pending_shift):
            h = self._hit_handle(e.x, e.y, self.boxes[self.sel])
            if h:
                self.action, self.handle = "resize", h
                return
        # 2) dragging inside ANY selected box moves the whole selection
        if (not self._pending_shift and self.selected
                and any(self._inside(i, ix, iy) for i in self.selected)):
            self.action = "move"
            return
        # 3) otherwise undecided until we see motion (resolved in on_drag):
        #      drag + a label set  -> draw a NEW box (even over a bigger one)
        #      drag + no label     -> move the box under the cursor
        #      click, no drag      -> select that box (Shift toggles), or deselect
        self.action = "pending"
        self._pending_box = self._box_at(ix, iy)

    def on_drag(self, e):
        if not self.img:
            return
        if self.panning:              # space+left-drag panning
            dx = e.x - self.pan_start[0]
            dy = e.y - self.pan_start[1]
            self.offx += dx
            self.offy += dy
            self.pan_start = (e.x, e.y)
            if self.space_held:
                self.space_pan_used = True
            self._clamp_pan()
            self.redraw()
            return
        if not self.action:
            return
        # resolve an undecided press as soon as the mouse really moves
        if self.action == "pending":
            mvx = e.x - self._press_screen[0]
            mvy = e.y - self._press_screen[1]
            if (mvx * mvx + mvy * mvy) ** 0.5 < 3:
                return                       # still a click, keep waiting
            if self._pending_shift:
                return                       # Shift is for click-toggle only
            if self.current_label:
                self._select_only(None)      # drawing wins, even over a big box
                self.action = "draw"
            elif self._pending_box is not None:
                self._select_only(self._pending_box)  # no label -> move it
                self.refresh_box_list()
                self.action = "move"
            else:
                self.action = None
                self.status.config(text="Set a label first (press W) to draw a box.")
                return
        ix, iy = self.to_image(e.x, e.y)
        if self.action == "draw":
            sx, sy = self.to_screen(*self.start)
            self.temp = (sx, sy, e.x, e.y)
            self.redraw()
        elif self.action == "move":
            dx, dy = ix - self.start[0], iy - self.start[1]
            W, H = self.img.width, self.img.height
            idxs = self._sel_indices()
            if not idxs:
                return
            # clamp the shared delta so the WHOLE selection stays in-bounds
            gx1 = min(self.boxes[i].norm()[0] for i in idxs)
            gy1 = min(self.boxes[i].norm()[1] for i in idxs)
            gx2 = max(self.boxes[i].norm()[2] for i in idxs)
            gy2 = max(self.boxes[i].norm()[3] for i in idxs)
            dx = max(-gx1, min(dx, W - gx2))
            dy = max(-gy1, min(dy, H - gy2))
            for i in idxs:
                b = self.boxes[i]
                b.x1 += dx; b.x2 += dx; b.y1 += dy; b.y2 += dy
            self.start = (ix, iy)
            self.redraw()
        elif self.action == "resize":
            b = self.boxes[self.sel]
            W, H = self.img.width, self.img.height
            ix = max(0, min(ix, W)); iy = max(0, min(iy, H))
            if "l" in self.handle: b.x1 = ix
            if "r" in self.handle: b.x2 = ix
            if "t" in self.handle: b.y1 = iy
            if "b" in self.handle: b.y2 = iy
            self.redraw()

    def on_up(self, e):
        if self.panning:
            self.panning = False
            # keep hand cursor while hand tool / space still active
            keep_hand = self.space_held or self.pan_mode.get()
            self.canvas.config(cursor="fleur" if keep_hand else "cross")
            return
        if self.action == "pending":
            # a plain click (no drag): select box under it (Shift toggles it in
            # or out of a multi-selection); empty space clears the selection
            if self._pending_shift:
                self._select_add(self._pending_box)
            elif self._pending_box is not None:
                self._select_only(self._pending_box)
            else:
                self._select_only(None)
            self.refresh_box_list()
        elif self.action == "draw" and self.temp:
            ix, iy = self.to_image(e.x, e.y)
            x1, y1 = self.start
            if abs(ix - x1) > MIN_BOX and abs(iy - y1) > MIN_BOX:
                b = Box(x1, y1, ix, iy, self.current_label)
                x1, y1, x2, y2 = b.norm()
                b.x1, b.y1, b.x2, b.y2 = (max(0, x1), max(0, y1),
                                          min(self.img.width, x2),
                                          min(self.img.height, y2))
                self.boxes.append(b)
                self._select_only(len(self.boxes) - 1)
                self.refresh_box_list()
        self.action = None
        self.temp = None
        self.handle = None
        self._pending_box = None
        self.redraw()

    def on_hover(self, e):
        if not self.img:
            return
        self.mouse_img = self.to_image(e.x, e.y)   # for paste-at-cursor
        # while the hand/pan tool is active (space held, Pan checkbox, or a
        # pan in progress) always show the move cursor and never the draw cross
        if self.space_held or self.pan_mode.get() or self.panning:
            self.canvas.config(cursor="fleur")
            return
        if self.sel is not None and len(self.selected) <= 1:
            h = self._hit_handle(e.x, e.y, self.boxes[self.sel])
            cur = {"tl": "top_left_corner", "tr": "top_right_corner",
                   "bl": "bottom_left_corner", "br": "bottom_right_corner"}.get(h)
            self.canvas.config(cursor=cur or "cross")
        else:
            self.canvas.config(cursor="cross")

    def _hit_handle(self, sx, sy, b):
        x1, y1, x2, y2 = b.norm()
        pts = {"tl": (x1, y1), "tr": (x2, y1), "bl": (x1, y2), "br": (x2, y2)}
        for name, (px, py) in pts.items():
            spx, spy = self.to_screen(px, py)
            if abs(sx - spx) <= HANDLE and abs(sy - spy) <= HANDLE:
                return {"tl": "tl", "tr": "tr", "bl": "bl", "br": "br"}[name]
        return None

    def _box_at(self, ix, iy):
        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.boxes[i].norm()
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return i
        return None

    def _inside(self, idx, ix, iy):
        x1, y1, x2, y2 = self.boxes[idx].norm()
        return x1 <= ix <= x2 and y1 <= iy <= y2

    # ---------------- labels ----------------
    def focus_new_label(self, e=None):
        """'w' shortcut: jump to the label entry to type/create a label.
        Ignored if the user is already typing in a text widget."""
        w = self.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            return  # let the 'w' be typed normally
        self.label_box.focus_set()
        self.label_box.selection_range(0, "end")
        self.status.config(text="Type a label name, then Enter to set it.")
        return "break"

    def on_pick_label(self, e=None):
        self.current_label = self.label_box.get().strip()
        # stop editing the label box once a label is chosen
        self.canvas.focus_set()

    def deselect(self, e=None):
        """Clear the current selection (Escape). Lets you draw over a box
        that is currently selected, and returns focus to the canvas."""
        self._select_only(None)
        self.box_list.selection_clear(0, "end")
        self.canvas.focus_set()
        self.redraw()
        return "break"

    def on_new_label(self, e=None):
        lbl = self.label_box.get().strip()
        if not lbl:
            return
        self.current_label = lbl
        if lbl not in self.label_history:
            self.label_history.append(lbl)
            self.label_box.config(values=self.label_history)
            self._save_config()
        # apply to every selected box (a whole group if one is grouped)
        for i in self._sel_indices():
            self.boxes[i].label = lbl
        self.refresh_box_list()
        self.redraw()
        # hand focus back to the canvas so a/d/arrows navigate again
        self.canvas.focus_set()
        return "break"

    def edit_box_label(self):
        idxs = self._sel_indices()
        if not idxs:
            return
        anchor = self.sel if self.sel in idxs else min(idxs)
        new = self._choose_label(self.boxes[anchor].label, count=len(idxs))
        if not new:
            return
        for i in idxs:
            self.boxes[i].label = new
        if new not in self.label_history:
            self.label_history.append(new)
            self.label_box.config(values=self.label_history)
        self.current_label = new
        self._save_config()
        self.refresh_box_list()
        self.redraw()

    def delete_box(self):
        if self._typing():
            return                      # don't delete boxes while editing text
        idxs = sorted(self._sel_indices(), reverse=True)
        if not idxs:
            return
        for i in idxs:                  # delete high->low so indices stay valid
            del self.boxes[i]
        self._select_only(None)
        self.refresh_box_list()
        self.redraw()

    def refresh_box_list(self, select=None):
        self.box_list.delete(0, "end")
        for b in self.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.norm())
            tag = "▣ " if b.group is not None else ""
            self.box_list.insert("end", f"{tag}{b.label}  [{x1},{y1},{x2},{y2}]")
        # a legacy single-select request updates the selection to that box's group
        if select is not None:
            self._select_only(select)
        self.box_list.selection_clear(0, "end")
        for i in self.selected:
            if 0 <= i < len(self.boxes):
                self.box_list.selection_set(i)

    def on_select_box(self, e):
        sel = self.box_list.curselection()
        if sel:
            self.selected = set(sel)
            self.sel = sel[-1]
        else:
            self.selected = set()
            self.sel = None
        self.redraw()

    # ---------------- selection & grouping ----------------
    def _group_members(self, idx):
        """All box indices that belong to the same group as idx (or just {idx})."""
        if idx is None or not (0 <= idx < len(self.boxes)):
            return set()
        g = self.boxes[idx].group
        if g is None:
            return {idx}
        return {i for i, b in enumerate(self.boxes) if b.group == g}

    def _select_only(self, idx):
        """Select box idx and its whole group; idx=None clears the selection."""
        if idx is None:
            self.sel = None
            self.selected = set()
        else:
            self.sel = idx
            self.selected = self._group_members(idx)

    def _select_add(self, idx):
        """Shift-click: toggle box idx (and its group) in/out of the selection."""
        if idx is None:
            return
        members = self._group_members(idx)
        if members <= self.selected:            # already selected -> remove
            self.selected -= members
            if self.sel in members:
                self.sel = next(iter(self.selected), None)
        else:                                   # add
            self.selected |= members
            self.sel = idx

    def _sel_indices(self):
        """The active selection as a set, falling back to the primary index."""
        return set(self.selected) if self.selected else (
            {self.sel} if self.sel is not None else set())

    def toggle_group(self):
        """Cmd/Ctrl+G: group the selection, or ungroup it if it's already one
        group. Clicking any grouped box selects the whole group, so pressing
        Cmd/Ctrl+G again on it ungroups."""
        idxs = self._sel_indices()
        if not idxs:
            self.status.config(text="Select 2+ boxes (Shift-click them) then "
                                    "Cmd/Ctrl+G to group.")
            return
        groups = {self.boxes[i].group for i in idxs}
        is_one_group = (None not in groups and len(groups) == 1)
        if is_one_group:                       # already grouped -> ungroup
            for i in idxs:
                self.boxes[i].group = None
            self.status.config(text=f"Ungrouped {len(idxs)} boxes.")
        elif len(idxs) >= 2:                   # group them
            gid = self._next_group
            self._next_group += 1
            for i in idxs:
                self.boxes[i].group = gid
            self.status.config(text=f"Grouped {len(idxs)} boxes — they move, "
                                    f"copy and relabel together. Press "
                                    f"Cmd/Ctrl+G again to ungroup.")
        else:
            self.status.config(text="Select 2+ boxes (Shift-click them) to group.")
            return
        self.refresh_box_list()
        self.redraw()

    def ungroup_selected(self):
        idxs = self._sel_indices()
        n = 0
        for i in idxs:
            if self.boxes[i].group is not None:
                self.boxes[i].group = None
                n += 1
        self.status.config(text=f"Ungrouped {n} box(es)." if n
                           else "Nothing grouped in the selection.")
        self.refresh_box_list()
        self.redraw()

    def _choose_label(self, initial="", count=1):
        """Modal label picker: searchable dropdown of known classes + free text,
        and a display of the label currently set on the box(es) being edited."""
        dlg = tk.Toplevel(self)
        dlg.title("Edit label")
        dlg.transient(self)
        dlg.resizable(False, False)
        head = f"Editing {count} boxes" if count > 1 else "Editing 1 box"
        ttk.Label(dlg, text=head).pack(padx=14, pady=(14, 2))
        ttk.Label(dlg, text=f"Currently labeled: {initial or '(none)'}",
                  foreground="#cc0000").pack(padx=14, pady=(0, 6))
        ttk.Label(dlg, text="Type to search classes, or enter a new label:").pack(
            padx=14, pady=(0, 4))
        var = tk.StringVar(value=initial)
        cb = ttk.Combobox(dlg, textvariable=var, values=self.label_history,
                          width=32)
        cb.pack(padx=14, pady=4)
        self._attach_search(cb, lambda: self.label_history)   # type-to-search
        result = {"val": None}

        def ok(e=None):
            result["val"] = var.get().strip()
            dlg.destroy()

        def cancel(e=None):
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(pady=10)
        ttk.Button(bf, text="OK", command=ok).pack(side="left", padx=4)
        ttk.Button(bf, text="Cancel", command=cancel).pack(side="left", padx=4)
        cb.bind("<Return>", ok)
        dlg.bind("<Escape>", cancel)
        cb.focus_set()
        cb.selection_range(0, "end")
        dlg.grab_set()
        self.wait_window(dlg)
        return result["val"]

    # ---------------- copy / paste ----------------
    # The buffer persists until you copy again — copy ONCE, paste as many times
    # and across as many images as you like. Nothing here (or in navigation)
    # clears it, and an accidental copy on an empty image won't wipe it.
    def copy_boxes(self):
        idxs = sorted(self.selected) if self.selected else list(range(len(self.boxes)))
        if idxs:
            self.clipboard = [self.boxes[i].copy() for i in idxs]
            self.paste_count = 0
            n = len(self.clipboard)
            grp = " (grouped)" if any(b.group is not None for b in self.clipboard) \
                and len({b.group for b in self.clipboard}) == 1 and n > 1 else ""
            self.status.config(text=f"Copied {n} box(es){grp} to buffer — move the "
                                    f"cursor where you want them and press "
                                    f"Ctrl/Cmd+V. Buffer stays until you copy again.")
        else:
            held = len(self.clipboard)
            self.status.config(text=(f"Nothing to copy here — buffer still holds "
                                     f"{held} box(es)." if held else
                                     "Nothing to copy (buffer empty)."))

    def paste_boxes(self):
        if not self.clipboard:
            self.status.config(text="Buffer empty — select box(es) and press "
                                    "Ctrl/Cmd+C first.")
            return
        W, H = self.img.width, self.img.height
        # bounding box (source coords) of the whole clipboard set
        sx1 = min(b.norm()[0] for b in self.clipboard)
        sy1 = min(b.norm()[1] for b in self.clipboard)
        sx2 = max(b.norm()[2] for b in self.clipboard)
        sy2 = max(b.norm()[3] for b in self.clipboard)
        scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        # target: center the set on the cursor; if the cursor isn't over the
        # image, fall back to a small incremental offset from the copied spot
        if self.mouse_img is not None:
            tcx, tcy = self.mouse_img
        else:
            if self.paste_last_index != self.index:
                self.paste_count = 0
                self.paste_last_index = self.index
            step = max(10, int(0.03 * min(W, H)))
            tcx, tcy = scx + self.paste_count * step, scy + self.paste_count * step
            self.paste_count += 1
        dx, dy = tcx - scx, tcy - scy
        # clamp so the whole pasted set stays inside the image
        dx = max(-sx1, min(dx, W - sx2))
        dy = max(-sy1, min(dy, H - sy2))
        remap, new_idx = {}, []
        for b in self.clipboard:
            nb = b.copy()
            nb.x1 += dx; nb.x2 += dx; nb.y1 += dy; nb.y2 += dy
            if nb.group is not None:      # keep grouping, but with fresh ids
                if nb.group not in remap:
                    remap[nb.group] = self._next_group
                    self._next_group += 1
                nb.group = remap[nb.group]
            self.boxes.append(nb)
            new_idx.append(len(self.boxes) - 1)
            if nb.label not in self.label_history:
                self.label_history.append(nb.label)
        self.selected = set(new_idx)      # select what we just pasted
        self.sel = new_idx[-1]
        self.label_box.config(values=self.label_history)
        self.refresh_box_list()
        self.redraw()
        self.status.config(text=f"Pasted {len(self.clipboard)} box(es) at cursor. "
                                f"Buffer still holds {len(self.clipboard)} — "
                                f"paste again anytime.")

    # ---------------- navigation ----------------
    def next_img(self):
        if self.index < len(self.images) - 1:
            self.load_image(self.index + 1)
        elif self.images:
            self.status.config(text="Last image — end of list.")

    def prev_img(self):
        if self.index > 0:
            self.load_image(self.index - 1)
        elif self.images:
            self.status.config(text="First image — start of list.")

    def on_pick_file(self, e):
        sel = self.file_list.curselection()
        if sel:
            self.load_image(sel[0])

    # ---------------- save / load ----------------
    def save(self, silent=False):
        if self.index < 0 or not self.output_dir:
            if not silent:
                messagebox.showinfo("Save", "Pick an output folder first.")
            return
        # Skip silent (autosave) writes when nothing changed since load — this is
        # the big win when just browsing on a network volume: no disk write per
        # image switch. Explicit saves (Ctrl+S) always write.
        if silent and self._snapshot is not None and self._box_state() == self._snapshot:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        stem = os.path.splitext(self.images[self.index])[0]
        if self.fmt.get() == "YOLO":
            self._save_yolo(stem)
        else:
            self._save_voc(stem)
        self.annotated.add(stem)
        self._mark_one(self.index)          # fast: update only this row's mark
        self._snapshot = self._box_state()  # new baseline
        if not silent:
            self.status.config(text=f"Saved {stem} ({self.fmt.get()})")

    def _save_yolo(self, stem):
        W, H = self.img.width, self.img.height
        lines = []
        for b in self.boxes:
            x1, y1, x2, y2 = b.norm()
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            cid = self.label_history.index(b.label) if b.label in self.label_history else 0
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(self.output_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        # write/refresh classes.txt
        with open(os.path.join(self.output_dir, "classes.txt"), "w") as f:
            f.write("\n".join(self.label_history))

    def _save_voc(self, stem):
        W, H = self.img.width, self.img.height
        ann = ET.Element("annotation")
        ET.SubElement(ann, "folder").text = os.path.basename(self.input_dir)
        ET.SubElement(ann, "filename").text = self.images[self.index]
        ET.SubElement(ann, "path").text = os.path.join(self.input_dir, self.images[self.index])
        size = ET.SubElement(ann, "size")
        ET.SubElement(size, "width").text = str(W)
        ET.SubElement(size, "height").text = str(H)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(ann, "segmented").text = "0"
        for b in self.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.norm())
            obj = ET.SubElement(ann, "object")
            ET.SubElement(obj, "name").text = b.label
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bb = ET.SubElement(obj, "bndbox")
            ET.SubElement(bb, "xmin").text = str(x1)
            ET.SubElement(bb, "ymin").text = str(y1)
            ET.SubElement(bb, "xmax").text = str(x2)
            ET.SubElement(bb, "ymax").text = str(y2)
        xml = minidom.parseString(ET.tostring(ann)).toprettyxml(indent="  ")
        with open(os.path.join(self.output_dir, stem + ".xml"), "w") as f:
            f.write(xml)

    def _load_existing(self, name):
        """Load labels for current image from output_dir. Returns True if found."""
        stem = os.path.splitext(name)[0]
        xml = os.path.join(self.output_dir, stem + ".xml")
        txt = os.path.join(self.output_dir, stem + ".txt")
        if os.path.exists(xml):
            self.boxes = self._parse_voc(xml)
            return True
        if os.path.exists(txt):
            self.boxes = self._parse_yolo(txt, self.img.width, self.img.height)
            return True
        return False

    def _parse_yolo(self, path, W, H):
        boxes = []
        classes = self._read_classes(os.path.dirname(path))
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) != 5:
                    continue
                cid, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W
                y2 = (cy + bh / 2) * H
                label = classes[cid] if cid < len(classes) else f"class{cid}"
                if label not in self.label_history:
                    self.label_history.append(label)
                boxes.append(Box(x1, y1, x2, y2, label))
        self.label_box.config(values=self.label_history)
        return boxes

    def _parse_voc(self, path):
        boxes = []
        root = ET.parse(path).getroot()
        for obj in root.findall("object"):
            label = obj.findtext("name", "object")
            bb = obj.find("bndbox")
            x1 = float(bb.findtext("xmin"))
            y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax"))
            y2 = float(bb.findtext("ymax"))
            if label not in self.label_history:
                self.label_history.append(label)
            boxes.append(Box(x1, y1, x2, y2, label))
        self.label_box.config(values=self.label_history)
        return boxes

    def _read_classes(self, folder):
        p = os.path.join(folder, "classes.txt")
        if os.path.exists(p):
            with open(p) as f:
                return [l.strip() for l in f if l.strip()]
        return list(self.label_history)

    def _load_classes_file(self, folder):
        """Populate label_history from a folder's classes.txt only. Used when a
        new input/output folder is chosen so classes never leak between folders.
        If the folder has no classes.txt, label_history stays empty (fresh)."""
        p = os.path.join(folder, "classes.txt")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    self.label_history = [l.strip() for l in f if l.strip()]
            except Exception:
                self.label_history = []

    # ---------------- compare helpers ----------------
    def _match_in(self, folder, name):
        stem = os.path.splitext(name)[0]
        for f in os.listdir(folder):
            if os.path.splitext(f)[0] == stem and f.lower().endswith(IMG_EXTS):
                return os.path.join(folder, f)
        return None

    def _load_pairs(self, folder, name, W, H):
        stem = os.path.splitext(name)[0]
        xml = os.path.join(folder, stem + ".xml")
        txt = os.path.join(folder, stem + ".txt")
        if os.path.exists(xml):
            return self._parse_voc(xml)
        if os.path.exists(txt):
            return self._parse_yolo(txt, W, H)
        return []

    # ---------------- config ----------------
    def _load_config(self):
        try:
            with open(CONFIG) as f:
                c = json.load(f)
            # NOTE: labels are intentionally NOT restored here. Classes must
            # come only from the folder currently being labeled, so switching
            # folders never carries classes over from a previous one.
            self.input_dir = c.get("input", "")
            self.output_dir = c.get("output", "")
            # if last session used a separate output folder, keep treating it as
            # explicit so picking an input image folder won't reset it
            self.output_explicit = bool(self.output_dir
                                        and self.output_dir != self.input_dir)
        except Exception:
            pass

    def _save_config(self):
        try:
            with open(CONFIG, "w") as f:
                json.dump({"labels": self.label_history,
                           "input": self.input_dir,
                           "output": self.output_dir}, f)
        except Exception:
            pass


if __name__ == "__main__":
    app = Labeler()
    app.mainloop()
