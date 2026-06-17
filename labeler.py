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
    __slots__ = ("x1", "y1", "x2", "y2", "label")

    def __init__(self, x1, y1, x2, y2, label):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.label = label

    def norm(self):
        return (min(self.x1, self.x2), min(self.y1, self.y2),
                max(self.x1, self.x2), max(self.y1, self.y2))

    def copy(self):
        return Box(self.x1, self.y1, self.x2, self.y2, self.label)


class Labeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple Image Labeler")
        self.geometry("1280x820")

        # state
        self.input_dir = ""
        self.output_dir = ""
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
        self.clipboard = []         # copied boxes
        self.annotated = set()      # image names that have been saved

        # interaction
        self.action = None          # 'draw' | 'move' | 'resize'
        self.sel = None             # selected box index
        self.handle = None          # which handle when resizing
        self.start = (0, 0)
        self.temp = None
        self._pending_box = None    # box under an undecided press (click vs drag)
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
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<Motion>", self.on_hover)
        # scroll: plain = pan, Cmd/Ctrl = zoom (Win/Mac); Button-4/5 = Linux vert
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)
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

        ttk.Label(right, text="Current label:").pack(anchor="w")
        self.label_box = ttk.Combobox(right, values=self.label_history)
        self.label_box.pack(fill="x")
        self.label_box.bind("<<ComboboxSelected>>", self.on_pick_label)
        self.label_box.bind("<Return>", self.on_new_label)
        ttk.Button(right, text="Set / Add label", command=self.on_new_label).pack(fill="x", pady=2)

        ttk.Separator(right).pack(fill="x", pady=6)
        ttk.Label(right, text="Boxes in this image:").pack(anchor="w")
        self.box_list = tk.Listbox(right, height=14, exportselection=False,
                                   activestyle="none",
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

    def _nav(self, fn):
        """Wrap a shortcut so it's ignored while typing in a text field."""
        def handler(e):
            if self._typing():
                return            # let the keystroke go to the text box
            fn()
            return "break"
        return handler

    def _bind_keys(self):
        self.bind("<a>", self._nav(self.prev_img))
        self.bind("<d>", self._nav(self.next_img))
        self.bind("<Left>", self._nav(lambda: self.arrow(-1, 0)))
        self.bind("<Right>", self._nav(lambda: self.arrow(1, 0)))
        self.bind("<Up>", self._nav(lambda: self.arrow(0, -1)))
        self.bind("<Down>", self._nav(lambda: self.arrow(0, 1)))
        self.bind("<KeyPress-space>", self.on_space_down)
        self.bind("<KeyRelease-space>", self.on_space_up)
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Control-c>", lambda e: self.copy_boxes())
        self.bind("<Control-v>", lambda e: self.paste_boxes())
        # macOS Command-key equivalents
        self.bind("<Command-s>", lambda e: self.save())
        self.bind("<Command-c>", lambda e: self.copy_boxes())
        self.bind("<Command-v>", lambda e: self.paste_boxes())
        self.bind("<Command-0>", lambda e: self.zoom_reset())
        self.bind("<Delete>", lambda e: self.delete_box())
        self.bind("<BackSpace>", self._nav(self.delete_box))
        self.bind("<Escape>", lambda e: self.deselect())
        # w = create a new label (jump to the label box, ready to type)
        self.bind("<w>", self.focus_new_label)
        self.bind("<h>", self.toggle_pan_mode)
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
        # default output to the same folder as input (LabelImg-style).
        # User can still override afterwards via "Output Folder".
        self.output_dir = d
        self.output_lbl.config(text="Output: " + d)
        # --- fresh start: never carry classes/boxes from a previous folder ---
        self.index = -1
        self.label_history = []
        self.current_label = ""
        self.clipboard = []
        self.boxes = []
        self.sel = None
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

    def load_image(self, i):
        if not (0 <= i < len(self.images)):
            return
        # auto-save current before leaving (if enabled)
        if self.index >= 0 and self.boxes and self.autosave.get():
            self.save(silent=True)

        prev_boxes = [b.copy() for b in self.boxes]
        self.index = i
        path = os.path.join(self.input_dir, self.images[i])
        self.img = Image.open(path).convert("RGB")
        self.boxes = []

        # try to load existing labels
        loaded = self._load_existing(self.images[i])
        if not loaded and self.carry.get() and prev_boxes:
            # carry over boxes from previous image
            self.boxes = [b.copy() for b in prev_boxes]

        self.file_list.selection_clear(0, "end")
        self.file_list.selection_set(i)
        self.file_list.see(i)
        self.sel = None
        self.fit()
        self.refresh_box_list()
        self.redraw()
        self.status.config(text=f"{self.images[i]}  ({i+1}/{len(self.images)})  "
                                 f"{self.img.width}x{self.img.height}")

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

        Plain scroll  -> PAN the image (natural on a Mac trackpad).
        Cmd/Ctrl+scroll, or pinch -> ZOOM about the cursor.
        Linux/X11 sends Button-4/5 instead of <MouseWheel>.
        """
        if not self.img:
            return
        # state mask: 0x4 = Control, 0x8/0x10 = Alt, 0x40000/0x8 = Command on mac
        mod = bool(e.state & 0x4) or bool(e.state & 0x40000) or bool(e.state & 0x8)
        if getattr(e, "num", None) == 4:
            delta = 1
        elif getattr(e, "num", None) == 5:
            delta = -1
        else:
            delta = 1 if e.delta > 0 else -1

        if mod:  # zoom
            factor = 1.15 ** delta
            self.set_zoom(self.zoom * factor, anchor=(e.x, e.y))
        else:    # pan vertically (shift+scroll pans horizontally)
            step = 60 * delta
            if e.state & 0x1:  # Shift -> horizontal
                self.offx += step
            else:
                self.offy += step
            self._clamp_pan()
            self.redraw()

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
            w = 3 if i == self.sel else 2
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=col, width=w)
            self.canvas.create_rectangle(sx1, sy1 - 16, sx1 + 8 + 7 * len(b.label), sy1,
                                         fill=col, outline=col)
            self.canvas.create_text(sx1 + 3, sy1 - 8, anchor="w", text=b.label,
                                    fill="white", font=("TkDefaultFont", 9, "bold"))
            if i == self.sel:
                for hx, hy in ((sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)):
                    self.canvas.create_rectangle(hx - HANDLE, hy - HANDLE,
                                                 hx + HANDLE, hy + HANDLE,
                                                 fill="white", outline=col)
        if self.temp:
            self.canvas.create_rectangle(*self.temp, outline="#ffff00", width=2, dash=(4, 3))
        self._draw_compare()

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
    def on_down(self, e):
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
        self.start = (ix, iy)
        self._press_screen = (e.x, e.y)
        # 1) grab a resize handle of the currently selected box
        if self.sel is not None:
            h = self._hit_handle(e.x, e.y, self.boxes[self.sel])
            if h:
                self.action, self.handle = "resize", h
                return
        # 2) dragging *inside the already-selected box* moves it
        if self.sel is not None and self._inside(self.sel, ix, iy):
            self.action = "move"
            return
        # 3) otherwise undecided until we see motion (resolved in on_drag):
        #      drag + a label set  -> draw a NEW box (even over a bigger one)
        #      drag + no label     -> move the box under the cursor
        #      click, no drag      -> select that box, or deselect on empty space
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
            if self.current_label:
                self.sel = None              # drawing wins, even over a big box
                self.action = "draw"
            elif self._pending_box is not None:
                self.sel = self._pending_box # no label -> move the grabbed box
                self.refresh_box_list(select=self.sel)
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
            b = self.boxes[self.sel]
            W, H = self.img.width, self.img.height
            # limit delta so the box stays fully inside the image
            x1, y1, x2, y2 = b.norm()
            dx = max(-x1, min(dx, W - x2))
            dy = max(-y1, min(dy, H - y2))
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
            # a plain click (no drag) -> select the box under it, else deselect
            if self._pending_box is not None:
                self.sel = self._pending_box
                self.refresh_box_list(select=self.sel)
            else:
                self.sel = None
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
                self.sel = len(self.boxes) - 1
                self.refresh_box_list(select=self.sel)
        self.action = None
        self.temp = None
        self.handle = None
        self._pending_box = None
        self.redraw()

    def on_hover(self, e):
        if not self.img:
            return
        # while the hand/pan tool is active (space held, Pan checkbox, or a
        # pan in progress) always show the move cursor and never the draw cross
        if self.space_held or self.pan_mode.get() or self.panning:
            self.canvas.config(cursor="fleur")
            return
        if self.sel is not None:
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
        self.sel = None
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
        # apply to selected box if any
        if self.sel is not None:
            self.boxes[self.sel].label = lbl
            self.refresh_box_list(select=self.sel)
        self.redraw()
        # hand focus back to the canvas so a/d/arrows navigate again
        self.canvas.focus_set()
        return "break"

    def edit_box_label(self):
        if self.sel is None:
            return
        new = simpledialog.askstring("Edit label", "New label:",
                                     initialvalue=self.boxes[self.sel].label)
        if new:
            new = new.strip()
            self.boxes[self.sel].label = new
            if new not in self.label_history:
                self.label_history.append(new)
                self.label_box.config(values=self.label_history)
            self.current_label = new
            self._save_config()
            self.refresh_box_list(select=self.sel)
            self.redraw()

    def delete_box(self):
        if self.sel is not None:
            del self.boxes[self.sel]
            self.sel = None
            self.refresh_box_list()
            self.redraw()

    def refresh_box_list(self, select=None):
        self.box_list.delete(0, "end")
        for b in self.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.norm())
            self.box_list.insert("end", f"{b.label}  [{x1},{y1},{x2},{y2}]")
        if select is not None:
            self.box_list.selection_clear(0, "end")
            self.box_list.selection_set(select)

    def on_select_box(self, e):
        sel = self.box_list.curselection()
        if sel:
            self.sel = sel[0]
            self.redraw()

    # ---------------- copy / paste ----------------
    def copy_boxes(self):
        if self.sel is not None:
            self.clipboard = [self.boxes[self.sel].copy()]
            self.status.config(text="Copied 1 selected box — switch image and "
                                    "press Ctrl/Cmd+V to paste just that box.")
        else:
            self.clipboard = [b.copy() for b in self.boxes]
            self.status.config(text=f"Copied {len(self.clipboard)} box(es).")

    def paste_boxes(self):
        if not self.clipboard:
            return
        for b in self.clipboard:
            nb = b.copy()
            # clamp to current image
            nb.x1 = min(nb.x1, self.img.width); nb.x2 = min(nb.x2, self.img.width)
            nb.y1 = min(nb.y1, self.img.height); nb.y2 = min(nb.y2, self.img.height)
            self.boxes.append(nb)
            if nb.label not in self.label_history:
                self.label_history.append(nb.label)
        self.label_box.config(values=self.label_history)
        self.refresh_box_list()
        self.redraw()

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
        os.makedirs(self.output_dir, exist_ok=True)
        stem = os.path.splitext(self.images[self.index])[0]
        if self.fmt.get() == "YOLO":
            self._save_yolo(stem)
        else:
            self._save_voc(stem)
        self.annotated.add(stem)
        self._refresh_file_marks()
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
