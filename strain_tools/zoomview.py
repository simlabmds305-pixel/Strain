#!/usr/bin/env python3
# ============================================================================
#  ZOOMABLE PICTURE on a Tk canvas (used by the Bead Shrinkage and Humidity
#  Cycles windows)
#
#    mouse wheel            zoom in / out, centred on the pointer (1x - 16x)
#    right-click + drag     move around while zoomed
#    double right-click     back to the whole picture
#
#  The picture is cut from the FULL-resolution image at every zoom, so a zoomed
#  view shows real pixels, not an enlarged thumbnail.  Overlays (boxes, lines)
#  are the window's own: it draws them with to_canvas() and reads clicks with
#  to_full(), and redraws them in on_change() after every zoom or move.
# ============================================================================
from __future__ import annotations

import base64

import cv2

ZOOM_MAX = 16.0
ZOOM_STEP = 1.25


class ZoomView:
    def __init__(self, tk, canvas, width, height, on_change=None):
        self.tk, self.c, self.VW, self.VH = tk, canvas, width, height
        self.on_change = on_change
        self.bgr = None
        self.W = self.H = 0
        self.s0 = 1.0          # canvas px per image px at 1x (whole picture)
        self.z = 1.0
        self.ox = self.oy = 0.0  # image px at the canvas's top-left corner
        self.photo = None
        self._pan = None
        canvas.bind("<MouseWheel>", self._wheel)                  # Windows / macOS
        canvas.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, ZOOM_STEP))   # Linux
        canvas.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 1 / ZOOM_STEP))
        canvas.bind("<ButtonPress-3>", self._pan_start)
        canvas.bind("<B3-Motion>", self._pan_move)
        canvas.bind("<Double-Button-3>", lambda e: self.fit())
        canvas.bind("<Enter>", lambda e: canvas.focus_set())     # wheel reaches the canvas

    # ------------------------------------------------------------ geometry
    @property
    def scale(self):
        return self.s0 * self.z

    def to_canvas(self, x, y):
        return (x - self.ox) * self.scale, (y - self.oy) * self.scale

    def to_full(self, cx, cy):
        return self.ox + cx / self.scale, self.oy + cy / self.scale

    def size(self):
        return self.W, self.H

    # -------------------------------------------------------------- image
    def set_image(self, bgr, keep_view=False):
        """Show a full-resolution BGR image; keep_view keeps the zoom and place
        (same-size image, e.g. a nudge redraw)."""
        same = (keep_view and self.bgr is not None and bgr is not None
                and bgr.shape[:2] == self.bgr.shape[:2])
        self.bgr = bgr
        if bgr is None:
            self.c.delete("img")
            return
        self.H, self.W = bgr.shape[:2]
        self.s0 = min(self.VW / self.W, self.VH / self.H)
        if not same:
            self.z, self.ox, self.oy = 1.0, 0.0, 0.0
        self.render()

    def fit(self):
        self.z, self.ox, self.oy = 1.0, 0.0, 0.0
        self.render()
        self._changed()

    def _clamp(self):
        vw, vh = self.VW / self.scale, self.VH / self.scale
        self.ox = min(max(0.0, self.ox), max(0.0, self.W - vw))
        self.oy = min(max(0.0, self.oy), max(0.0, self.H - vh))

    def render(self):
        if self.bgr is None:
            return
        self._clamp()
        vw, vh = self.VW / self.scale, self.VH / self.scale
        x0, y0 = int(self.ox), int(self.oy)
        x1, y1 = min(self.W, int(self.ox + vw) + 2), min(self.H, int(self.oy + vh) + 2)
        crop = self.bgr[y0:y1, x0:x1]
        out_w = max(1, int(round((x1 - x0) * self.scale)))
        out_h = max(1, int(round((y1 - y0) * self.scale)))
        small = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA if self.scale < 1
                           else cv2.INTER_NEAREST)
        ok, png = cv2.imencode(".png", small)
        self.photo = self.tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
        self.c.delete("img")
        # the crop starts at the whole pixel x0, y0 - place it where that pixel falls
        px, py = self.to_canvas(x0, y0)
        self.c.create_image(px, py, anchor="nw", image=self.photo, tags="img")
        self.c.tag_lower("img")
        self.c.delete("zoomnote")
        if self.z > 1.001:
            self.c.create_text(self.VW - 6, 6, anchor="ne", fill="#ffcc00", tags="zoomnote",
                               text=f"{self.z:.1f}x   (double right-click = whole picture)")

    # ---------------------------------------------------------- interaction
    def zoom_at(self, cx, cy, factor):
        if self.bgr is None:
            return
        fx, fy = self.to_full(cx, cy)
        self.z = min(ZOOM_MAX, max(1.0, self.z * factor))
        self.ox, self.oy = fx - cx / self.scale, fy - cy / self.scale
        self.render()
        self._changed()

    def _wheel(self, e):
        self.zoom_at(e.x, e.y, ZOOM_STEP if e.delta > 0 else 1 / ZOOM_STEP)

    def _pan_start(self, e):
        self._pan = (e.x, e.y, self.ox, self.oy)

    def _pan_move(self, e):
        if not self._pan or self.z <= 1.0:
            return
        x, y, ox, oy = self._pan
        self.ox, self.oy = ox - (e.x - x) / self.scale, oy - (e.y - y) / self.scale
        self.render()
        self._changed()

    def _changed(self):
        if self.on_change:
            self.on_change()
