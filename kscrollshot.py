#!/usr/bin/env python3
"""
kscrollshot - scrolling screenshots for KDE Plasma on Wayland.

Select an area of the screen, scroll it (yourself, or let the tool send wheel
events with --auto), and get one tall PNG of everything that passed through.

How it works
  * Frames come from the xdg-desktop-portal ScreenCast API over PipeWire, the
    same path OBS and Spectacle's recorder use, so it works with any window.
  * Each new frame is matched against the previous one to find how many pixels
    the content moved; only the newly revealed rows are appended.
  * Sticky headers/footers inside the selected area are detected (rows that do
    not move) and kept once instead of being repeated down the image.

Dependencies (Arch / CachyOS):
  sudo pacman -S --needed python-numpy python-gobject python-pyqt6 \
      gst-plugin-pipewire gst-plugins-base
Optional: wl-clipboard (more reliable copy-to-clipboard)

Usage
  kscrollshot              manual mode: you scroll
  kscrollshot --auto       auto mode: the tool scrolls for you
  kscrollshot --install    copy to ~/.local/bin and add launcher entries
Run it again (or click its tray icon) while capturing to finish and save.
"""

import argparse
import faulthandler
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

APP = "kscrollshot"
TITLE = "KScrollShot"
_APP = None     # keeps the QApplication alive until the interpreter exits
PACMAN = ("sudo pacman -S --needed python-numpy python-gobject python-pyqt6 "
          "gst-plugin-pipewire gst-plugins-base")


# ----------------------------------------------------------------------------
# Stitching engine (pure numpy, no desktop dependencies)
# ----------------------------------------------------------------------------

class Stitcher:
    """Builds one tall image from successive frames of a scrolling area.

    Frames are HxWx4 uint8 arrays (BGRx) of identical size. Call feed() with
    each new frame; it returns one of:
        'first'    first frame stored
        'idle'     nothing moved
        'moved'    scroll offset found, canvas updated
        'reset'    content changed in place (e.g. an image finished loading)
        'nomatch'  frame changed but no reliable offset was found
    """

    TOL = 12          # per-pixel difference treated as "the same"
    NCOLS = 128       # columns sampled for matching
    KEEP_TOP = 0.25   # top share of the scrolling band never copied from later
                      # frames (header shadows, pointer hover effects live there)

    def __init__(self, max_height=60000):
        self.max_height = max_height
        self.h = 0              # frame height
        self.first = None
        self.ref = None
        self.y = 0              # document row of ref's row 0
        self.doc_max = None     # document rows [0, doc_max) are in the canvas
        self.canvas = None
        self.tail = None        # rows below the scrolling band (sticky footer)
        self.top_run = None     # smallest static top block seen (sticky header)
        self.bot_run = None     # smallest static bottom block seen
        self.last_d = 0
        self.fails = 0
        self.gaps = 0
        self.accepted = 0

    # -- public state ---------------------------------------------------
    @property
    def height(self):
        if self.doc_max is None:
            return self.h
        return self.doc_max + (0 if self.tail is None else self.tail.shape[0])

    @property
    def lost(self):
        return self.fails >= 8

    @property
    def full(self):
        return self.height >= self.max_height

    @property
    def band(self):
        """Height of the part of the frame that actually scrolls."""
        return self.h - (self.top_run or 0) - (self.bot_run or 0)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _run(mask):
        """Length of the leading run of True values."""
        bad = np.flatnonzero(~mask)
        return int(bad[0]) if bad.size else int(mask.size)

    @staticmethod
    def _shift_scores(a, b, tex):
        """Mean squared difference between a and b for every vertical shift.

        Returns (shifts, score, tex_rows) where shift d pairs a[i + d] with
        b[i]; tex_rows[d] counts rows with detail in b's overlapping part.
        """
        n, c = a.shape
        size = 1
        while size < 2 * n:
            size *= 2
        fa = np.fft.rfft(a, n=size, axis=0)
        fb = np.fft.rfft(b, n=size, axis=0)
        corr = np.fft.irfft(fa * np.conj(fb), n=size, axis=0).sum(axis=1)
        ca = np.concatenate(([0.0], np.cumsum((a * a).sum(axis=1))))
        cb = np.concatenate(([0.0], np.cumsum((b * b).sum(axis=1))))
        ct = np.concatenate(([0], np.cumsum(tex)))
        d = np.arange(-(n - 1), n)
        pos = d >= 0
        ad = np.abs(d)
        sa = np.where(pos, ca[n] - ca[np.where(pos, d, 0)], ca[n - ad])
        sb = np.where(pos, cb[n - ad], cb[n] - cb[np.where(pos, 0, ad)])
        cr = corr[np.where(pos, d, size + d)]
        rows = n - ad
        score = (sa + sb - 2.0 * cr) / (rows * c)
        trows = np.where(pos, ct[n - ad], ct[n] - ct[np.where(pos, 0, ad)])
        return d, score, trows

    def _agreement(self, ra, rb):
        """Share of detailed cells that disagree between two aligned blocks.

        The blocks are compared cell by cell (one row x a group of columns)
        so that something that changed in place (an image that finished
        loading, a video, a floating button) does not veto an otherwise
        perfect match, and so that sparse pages cannot match on background
        alone. Returns (bad_share, cells) or None if there is too little detail.
        """
        tol = self.TOL
        rows, cols = ra.shape
        groups = 8 if cols >= 64 else 4 if cols >= 24 else 1
        per = cols // groups
        use = per * groups
        mis = (np.abs(ra[:, :use] - rb[:, :use]) > tol).reshape(rows, groups, per)
        good = mis.sum(axis=2) <= max(1, int(per * 0.07))
        edge = ((np.abs(np.diff(ra[:, :use], axis=0)) > tol) |
                (np.abs(np.diff(rb[:, :use], axis=0)) > tol)).reshape(rows - 1, groups, per).any(axis=2)
        tex = np.zeros((rows, groups), bool)
        tex[:-1] = edge
        tex[1:] |= edge
        n_tex = int(tex.sum())
        if n_tex < 6 * groups:
            return None
        return 1.0 - float((good & tex).sum()) / n_tex, n_tex

    def _write(self, cur):
        """Copy the scrolling part of the current frame into the canvas."""
        h, f, t = self.h, self.bot_run, self.top_run
        if self.canvas is None:
            self.canvas = np.empty((max(4 * h, 4096),) + cur.shape[1:], np.uint8)
            self.canvas[: h - f] = self.first[: h - f]
            self.tail = self.first[h - f:].copy()
            self.doc_max = h - f
            self.first = None
        end = self.y + h - f
        w0 = t + int(self.KEEP_TOP * (h - f - t))
        if self.y + w0 > self.doc_max:          # must stay contiguous
            w0 = self.doc_max - self.y
            if w0 < t:
                w0 = t
                self.gaps += 1
        w0 = max(w0, -self.y)                   # scrolled above the start
        if w0 >= h - f:
            return
        if end > self.canvas.shape[0]:
            grown = np.empty((max(end, 2 * self.canvas.shape[0]),) + cur.shape[1:], np.uint8)
            grown[: self.doc_max] = self.canvas[: self.doc_max]
            self.canvas = grown
        self.canvas[self.y + w0: end] = cur[w0: h - f]
        if end >= self.doc_max:
            self.doc_max = end
            self.tail = cur[h - f:].copy()

    # -- main entry -------------------------------------------------------
    def feed(self, frame):
        if self.ref is None:
            self.first = self.ref = frame
            self.h = frame.shape[0]
            return "first"
        ref, cur = self.ref, frame
        if cur.shape != ref.shape:
            return "nomatch"
        tol = self.TOL
        gr, gc = ref[:, :, 1], cur[:, :, 1]
        h = gc.shape[0]

        # Cheap test on a coarse grid: did anything change at all?
        chg = np.abs(gr[::4, ::4].astype(np.int16) - gc[::4, ::4]) > tol
        if chg.mean() < 0.0005:
            return "idle"
        # Columns that changed are the ones that scroll; static side panels
        # carry no information and would only bias the match.
        active = np.flatnonzero(chg.mean(axis=0) > 0.004) * 4
        if active.size < 6:
            return "idle"
        pick = np.linspace(0, active.size - 1, min(self.NCOLS, active.size))
        idx = active[np.unique(pick.astype(int))]
        sr = gr[:, idx].astype(np.int16)
        sc = gc[:, idx].astype(np.int16)

        # Rows that did not move at the top/bottom = sticky header/footer
        # (or blank space, which is harmless to treat the same way).
        static = (np.abs(sr - sc) > tol).mean(axis=1) <= 0.02
        top = self._run(static)
        bot = self._run(static[::-1])
        if top + bot > h - 32:
            return "idle"
        b0, b1 = top, h - bot
        n = b1 - b0

        # Candidate offsets: best few of an exhaustive search over all shifts.
        a = sr[b0:b1].astype(np.float64) - 128.0
        b = sc[b0:b1].astype(np.float64) - 128.0
        edge_c = np.abs(np.diff(sc[b0:b1], axis=0, append=sc[b1 - 1:b1])) > tol
        shifts, score, trows = self._shift_scores(a, b, edge_c.any(axis=1))
        min_rows = max(12, int(n * 0.08))
        ok = (shifts != 0) & (n - np.abs(shifts) >= min_rows) & (trows >= 4)
        score = np.where(ok, score, np.inf)
        cands = []
        for i in np.argsort(score)[:40]:
            if not np.isfinite(score[i]):
                break
            dd = int(shifts[i])
            if all(abs(dd - e) >= 2 for e in cands):
                cands.append(dd)
                if len(cands) == 6:
                    break
        for extra in (self.last_d, self.last_d - 1, self.last_d + 1):
            if extra and extra not in cands and abs(extra) <= n - min_rows:
                cands.append(extra)

        scored = []
        for d in cands:
            if d > 0:
                res = self._agreement(sr[b0 + d: b1], sc[b0: b1 - d])
            else:
                res = self._agreement(sr[b0: b1 + d], sc[b0 - d: b1])
            if res is not None and res[0] <= 0.4:
                scored.append((res[0], d))

        d = None
        if scored:
            best = min(s[0] for s in scored)
            near = [s[1] for s in scored if s[0] <= best + 0.05]
            if len(near) == 1:
                d = near[0]
            elif self.last_d:
                d = min(near, key=lambda v: (abs(v - self.last_d), abs(v)))
            elif max(near) - min(near) <= 1:
                d = min(near, key=abs)
            # else: ambiguous (repeating content) -> wait for a clearer frame

        if d is None:
            # Changed in place rather than scrolled (lazy image, animation)?
            # Then detail in the part known to scroll is mostly where it was.
            if self.top_run is not None:
                lo, hi = self.top_run, h - self.bot_run
                res = self._agreement(sr[lo:hi], sc[lo:hi])
                if res is not None and res[0] <= 0.5:
                    self.ref = cur
                    self.fails = 0
                    self._write(cur)
                    return "reset"
            self.fails += 1
            return "nomatch"

        # Accepted.
        self.fails = 0
        self.accepted += 1
        self.last_d = d
        self.top_run = top if self.top_run is None else min(self.top_run, top)
        self.bot_run = bot if self.bot_run is None else min(self.bot_run, bot)
        self.y += d
        self.ref = cur
        self._write(cur)
        return "moved"

    def result(self):
        if self.ref is None:
            return None
        if self.doc_max is None:
            return np.ascontiguousarray(self.first)
        parts = [self.canvas[: self.doc_max]]
        if self.tail is not None and self.tail.shape[0]:
            parts.append(self.tail)
        img = np.concatenate(parts, axis=0)
        return np.ascontiguousarray(img[: self.max_height])


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def log(*a):
    print(f"[{APP}]", *a, file=sys.stderr, flush=True)


def notify(msg):
    """Tell the user something went wrong even when started from a shortcut."""
    log(msg)
    if shutil.which("notify-send"):
        try:
            subprocess.Popen(["notify-send", "-a", TITLE, "-i", "camera-photo", TITLE, msg])
        except OSError:
            pass


def config_dir():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    p = Path(base) / APP
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_tokens():
    try:
        return json.loads((config_dir() / "tokens.json").read_text())
    except (OSError, ValueError):
        return {}


def save_tokens(tokens):
    try:
        (config_dir() / "tokens.json").write_text(json.dumps(tokens))
    except OSError as e:
        log("could not save portal token:", e)


def pictures_dir():
    d = None
    if shutil.which("xdg-user-dir"):
        try:
            d = subprocess.run(["xdg-user-dir", "PICTURES"], capture_output=True,
                               text=True, timeout=3).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            d = None
    base = Path(d) if d and d != os.path.expanduser("~") else Path.home() / "Pictures"
    return base / "Screenshots"


def pid_file():
    return Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / f"{APP}.pid"


def signal_running_instance():
    """If a capture is in progress, ask it to finish. Returns True if it was."""
    try:
        pid = int(pid_file().read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return False
    if pid == os.getpid() or APP.encode() not in cmd:
        return False
    try:
        os.kill(pid, signal.SIGUSR1)
    except OSError:
        return False
    return True


def missing_deps():
    problems = []
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gio, GLib, Gst  # noqa: F401
        Gst.init(None)
        for el in ("pipewiresrc", "videoconvert", "appsink"):
            if Gst.ElementFactory.find(el) is None:
                problems.append(f"GStreamer element '{el}' not found")
    except (ImportError, ValueError) as e:
        problems.append(f"PyGObject/GStreamer: {e}")
    try:
        import PyQt6.QtWidgets  # noqa: F401
    except ImportError:
        try:
            import PySide6.QtWidgets  # noqa: F401
        except ImportError:
            problems.append("PyQt6 (or PySide6) not found")
    return problems


# ----------------------------------------------------------------------------
# xdg-desktop-portal: screen cast (+ remote desktop for auto-scroll)
# ----------------------------------------------------------------------------

class PortalError(RuntimeError):
    pass


class Portal:
    BUS = "org.freedesktop.portal.Desktop"
    PATH = "/org/freedesktop/portal/desktop"
    SC = "org.freedesktop.portal.ScreenCast"
    RD = "org.freedesktop.portal.RemoteDesktop"

    def __init__(self):
        from gi.repository import Gio, GLib
        self.Gio, self.GLib = Gio, GLib
        self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self.sender = self.bus.get_unique_name()[1:].replace(".", "_")
        self.count = 0
        self.session = None
        self.node = None
        self.remote = False

    def _token(self):
        self.count += 1
        return f"{APP}{os.getpid()}n{self.count}"

    def _request(self, iface, method, sig, args, options):
        """Call a portal method that replies through a Request object."""
        Gio, GLib = self.Gio, self.GLib
        token = self._token()
        path = f"{self.PATH}/request/{self.sender}/{token}"
        options = dict(options)
        options["handle_token"] = GLib.Variant("s", token)
        loop = GLib.MainLoop()
        out = {}

        def on_response(_conn, _sender, _path, _iface, _signal, params, *_):
            out["code"], out["results"] = params.unpack()
            loop.quit()

        sub = self.bus.signal_subscribe(self.BUS, "org.freedesktop.portal.Request",
                                        "Response", path, None,
                                        Gio.DBusSignalFlags.NONE, on_response)
        try:
            self.bus.call_sync(self.BUS, self.PATH, iface, method,
                               GLib.Variant(sig, tuple(args) + (options,)),
                               None, Gio.DBusCallFlags.NONE, -1, None)
            loop.run()
        except GLib.Error as e:
            raise PortalError(f"{method}: {e.message}") from e
        finally:
            self.bus.signal_unsubscribe(sub)
        if out.get("code") == 1:
            raise PortalError("cancelled")
        if out.get("code") != 0:
            raise PortalError(f"{method} was refused by the portal")
        return out["results"]

    def start(self, remote, restore_token):
        """Open a session; returns (pipewire_fd, node_id, stream_props, new_token)."""
        v = self.GLib.Variant
        self.remote = remote
        iface = self.RD if remote else self.SC
        res = self._request(iface, "CreateSession", "(a{sv})", (),
                            {"session_handle_token": v("s", self._token())})
        self.session = res["session_handle"]
        persist = {"persist_mode": v("u", 2)}
        if restore_token:
            persist["restore_token"] = v("s", restore_token)
        sources = {"types": v("u", 1), "multiple": v("b", False), "cursor_mode": v("u", 1)}
        if remote:
            devices = {"types": v("u", 2)}      # pointer
            devices.update(persist)
            self._request(self.RD, "SelectDevices", "(oa{sv})", (self.session,), devices)
        else:
            sources.update(persist)
        self._request(self.SC, "SelectSources", "(oa{sv})", (self.session,), sources)
        res = self._request(iface, "Start", "(osa{sv})", (self.session, ""), {})
        streams = res.get("streams") or []
        if not streams:
            raise PortalError("the portal returned no screen stream")
        self.node, props = streams[0]
        reply, fds = self.bus.call_with_unix_fd_list_sync(
            self.BUS, self.PATH, self.SC, "OpenPipeWireRemote",
            self.GLib.Variant("(oa{sv})", (self.session, {})),
            self.GLib.VariantType("(h)"), self.Gio.DBusCallFlags.NONE, -1, None, None)
        fd = fds.get(reply.unpack()[0])
        return fd, int(self.node), dict(props), res.get("restore_token")

    def _notify(self, method, sig, args):
        try:
            self.bus.call(self.BUS, self.PATH, self.RD, method,
                          self.GLib.Variant(sig, (self.session, {}) + tuple(args)),
                          None, self.Gio.DBusCallFlags.NONE, 2000, None, None)
        except self.GLib.Error as e:
            log(method, "failed:", e.message)

    def scroll(self, steps):
        """Send mouse-wheel steps (positive = down). Needs a remote session."""
        if self.remote and self.session:
            self._notify("NotifyPointerAxisDiscrete", "(oa{sv}ui)", (0, int(steps)))

    def move_pointer(self, x, y):
        """Move the pointer to stream coordinates (logical pixels)."""
        if self.remote and self.session:
            self._notify("NotifyPointerMotionAbsolute", "(oa{sv}udd)",
                         (int(self.node), float(x), float(y)))

    def close(self):
        if self.session:
            try:
                self.bus.call(self.BUS, self.session, "org.freedesktop.portal.Session",
                              "Close", None, None, self.Gio.DBusCallFlags.NONE, 1000,
                              None, None)
            except self.GLib.Error:
                pass
            self.session = None


# ----------------------------------------------------------------------------
# PipeWire frames through GStreamer
# ----------------------------------------------------------------------------

class Grabber:
    def __init__(self, fd, node):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        self.Gst = Gst
        self.error = None
        self.pipe = Gst.Pipeline.new(APP)
        src = Gst.ElementFactory.make("pipewiresrc", "src")
        conv = Gst.ElementFactory.make("videoconvert", "conv")
        self.sink = Gst.ElementFactory.make("appsink", "sink")
        if not (src and conv and self.sink):
            raise RuntimeError("GStreamer elements missing. Install with:\n  " + PACMAN)
        src.set_property("fd", fd)
        if src.find_property("path") is not None:
            src.set_property("path", str(node))
        else:
            src.set_property("target-object", str(node))
        for name, value in (("do-timestamp", True), ("always-copy", True)):
            if src.find_property(name) is not None:
                src.set_property(name, value)
        self.sink.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGRx"))
        self.sink.set_property("max-buffers", 1)
        self.sink.set_property("drop", True)
        self.sink.set_property("sync", False)
        for el in (src, conv, self.sink):
            self.pipe.add(el)
        if not (src.link(conv) and conv.link(self.sink)):
            raise RuntimeError("could not link the GStreamer pipeline")
        self.bus = self.pipe.get_bus()
        self.pipe.set_state(Gst.State.PLAYING)

    def poll(self):
        """Return the newest frame as an HxWx4 BGRx array, or None."""
        Gst = self.Gst
        msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg is not None:
            if msg.type == Gst.MessageType.ERROR:
                err, _dbg = msg.parse_error()
                self.error = err.message
            else:
                self.error = "the screen stream ended"
        sample = self.sink.emit("try-pull-sample", 0)
        if sample is None:
            return None
        st = sample.get_caps().get_structure(0)
        w, h = st.get_value("width"), st.get_value("height")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            data = bytes(info.data)
        finally:
            buf.unmap(info)
        stride = len(data) // h
        if stride < w * 4:
            return None
        arr = np.frombuffer(data, np.uint8, count=h * stride).reshape(h, stride)
        if stride != w * 4:
            arr = np.ascontiguousarray(arr[:, : w * 4])
        return arr.reshape(h, w, 4)

    def stop(self):
        try:
            self.pipe.set_state(self.Gst.State.NULL)
        except Exception:  # noqa: BLE001 - shutting down
            pass


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------

def run_gui(args, portal, grabber, props):
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        from PySide6 import QtCore, QtGui, QtWidgets
    Qt = QtCore.Qt

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP)
    app.setApplicationDisplayName(TITLE)
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    if (data_home / "applications" / f"{APP}.desktop").exists():
        app.setDesktopFileName(APP)     # otherwise Qt logs a portal warning
    app.setQuitOnLastWindowClosed(False)
    icon = QtGui.QIcon.fromTheme("camera-photo", QtGui.QIcon.fromTheme("spectacle"))
    app.setWindowIcon(icon)

    def to_qimage(arr):
        # A QImage built on a Python buffer only borrows it, and QPixmap.fromImage
        # shares that borrowed memory instead of copying. Return a deep copy that
        # owns its pixels so nothing can outlive the buffer (that was a segfault).
        h, w = arr.shape[:2]
        data = arr.tobytes()
        return QtGui.QImage(data, w, h, w * 4, QtGui.QImage.Format.Format_RGB32).copy()

    # -- region selection over a frozen frame -----------------------------
    class Overlay(QtWidgets.QWidget):
        def __init__(self, pixmap, hint, done):
            super().__init__(None, Qt.WindowType.FramelessWindowHint |
                             Qt.WindowType.WindowStaysOnTopHint)
            self.pix, self.hint, self.done = pixmap, hint, done
            self.origin = None
            self.sel = QtCore.QRect()
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setWindowTitle(TITLE)

        def paintEvent(self, _event):
            p = QtGui.QPainter(self)
            p.drawPixmap(self.rect(), self.pix)
            dim = QtGui.QColor(0, 0, 0, 120)
            if self.sel.isEmpty():
                p.fillRect(self.rect(), dim)
            else:
                p.setClipRegion(QtGui.QRegion(self.rect()).subtracted(QtGui.QRegion(self.sel)))
                p.fillRect(self.rect(), dim)
                p.setClipping(False)
                p.setPen(QtGui.QPen(QtGui.QColor(61, 174, 233), 2))
                p.drawRect(self.sel.adjusted(0, 0, -1, -1))
            font = p.font()
            if font.pointSizeF() > 0:
                font.setPointSizeF(font.pointSizeF() * 1.15)
            p.setFont(font)
            flags = Qt.AlignmentFlag.AlignCenter.value
            box = p.fontMetrics().boundingRect(QtCore.QRect(0, 0, self.width() - 80, 400),
                                               flags, self.hint)
            box.adjust(-18, -12, 18, 12)
            box.moveCenter(QtCore.QPoint(self.width() // 2, 0))
            box.moveTop(28)
            if not self.sel.isEmpty() and box.intersects(self.sel):
                box.moveBottom(self.height() - 28)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(20, 22, 26, 225))
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.drawRoundedRect(box, 10, 10)
            p.setPen(QtGui.QColor(240, 240, 240))
            p.drawText(box, flags, self.hint)

        def mousePressEvent(self, e):
            if e.button() == Qt.MouseButton.LeftButton:
                self.origin = e.position().toPoint()
                self.sel = QtCore.QRect()
                self.update()
            elif e.button() == Qt.MouseButton.RightButton:
                self.done(None)

        def mouseMoveEvent(self, e):
            if self.origin is not None:
                self.sel = QtCore.QRect(self.origin, e.position().toPoint()).normalized() \
                    .intersected(self.rect())
                self.update()

        def mouseReleaseEvent(self, e):
            if e.button() != Qt.MouseButton.LeftButton or self.origin is None:
                return
            self.origin = None
            if self.sel.width() >= 40 and self.sel.height() >= 80:
                self.done(QtCore.QRect(self.sel))
            else:
                self.sel = QtCore.QRect()
                self.update()

        def keyPressEvent(self, e):
            if e.key() == Qt.Key.Key_Escape:
                self.done(None)

    # -- result window -------------------------------------------------------
    class Result(QtWidgets.QWidget):
        def __init__(self, path, qimg, note):
            super().__init__()
            self.path, self.qimg = path, qimg
            self.setWindowTitle(f"{TITLE} - saved")
            lay = QtWidgets.QVBoxLayout(self)
            info = QtWidgets.QLabel(f"<b>{qimg.width()} x {qimg.height()} px</b><br>{path}")
            info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            info.setWordWrap(True)
            lay.addWidget(info)
            if note:
                warn = QtWidgets.QLabel(note)
                warn.setWordWrap(True)
                lay.addWidget(warn)
            preview = QtWidgets.QLabel()
            thumb = qimg.scaledToWidth(min(440, qimg.width()),
                                       Qt.TransformationMode.SmoothTransformation)
            preview.setPixmap(QtGui.QPixmap.fromImage(thumb))
            area = QtWidgets.QScrollArea()
            area.setWidget(preview)
            area.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            area.setMinimumSize(480, 420)
            lay.addWidget(area, 1)
            row = QtWidgets.QHBoxLayout()
            lay.addLayout(row)

            def button(text, fn):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(lambda *_: fn())
                row.addWidget(b)
                return b

            button("Open", lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(path))))
            button("Show in folder", self.show_in_folder)
            button("Copy", self.copy)
            if shutil.which("spectacle"):
                button("Annotate", lambda: subprocess.Popen(
                    ["spectacle", "--edit-existing", str(path)]))
            row.addStretch(1)
            button("Close", self.close).setDefault(True)

        def show_in_folder(self):
            if shutil.which("dolphin"):
                subprocess.Popen(["dolphin", "--select", str(self.path)])
            else:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.path.parent)))

        def copy(self):
            if not copy_to_clipboard(self.path):
                app.clipboard().setImage(self.qimg)

        def closeEvent(self, e):
            e.accept()
            app.quit()

    def copy_to_clipboard(path):
        if not shutil.which("wl-copy"):
            return False
        try:
            with open(path, "rb") as fh:
                subprocess.Popen(["wl-copy", "--type", "image/png"], stdin=fh)
            return True
        except OSError:
            return False

    # -- controller ------------------------------------------------------
    class Controller:
        def __init__(self):
            self.phase = "wait"
            self.t0 = time.monotonic()
            self.frame = None
            self.frozen = None
            self.overlay = None
            self.result = None
            self.region = None           # x0, y0, x1, y1 in frame pixels
            self.logical = None          # region centre etc. in logical px
            self.stitcher = Stitcher(max_height=args.max_height)
            self.finish_requested = False
            self.prev_crop = None
            self.t_hide = self.t_region_change = 0.0
            self.t_move = self.t_change = 0.0
            self.ever_moved = False
            self.auto = bool(args.auto)
            self.tick_t = None
            self.hold_until = 0.0
            self.moved_since_tick = False
            self.y_at_tick = 0
            self.steps = self.pending_steps = 1
            self.rate = None
            self.nomove = 0
            self.t_tooltip = 0.0
            self.tray = None
            self.timer = QtCore.QTimer()
            self.timer.setInterval(25)
            self.timer.timeout.connect(self.safe_tick)
            self.timer.start()

        # .. plumbing ..
        def fail(self, msg):
            notify(msg)
            self.shutdown()
            app.exit(1)

        def shutdown(self):
            self.timer.stop()
            grabber.stop()
            portal.close()
            if self.tray is not None:
                self.tray.hide()
            try:
                pid_file().unlink()
            except OSError:
                pass

        def safe_tick(self):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - report instead of dying silently
                import traceback
                traceback.print_exc()
                self.fail(f"Unexpected error: {e}")

        def crop(self, frame):
            x0, y0, x1, y1 = self.region
            return np.ascontiguousarray(frame[y0:y1, x0:x1])

        # .. phases ..
        def tick(self):
            now = time.monotonic()
            frame = grabber.poll()
            if grabber.error:
                return self.fail(f"Screen capture failed: {grabber.error}")
            if frame is not None:
                self.frame = frame

            if self.phase == "wait":
                if self.frame is not None:
                    self.show_overlay()
                elif now - self.t0 > 8:
                    self.fail("No frames arrived from the screen stream. "
                              "Is gst-plugin-pipewire installed?")

            elif self.phase == "settle":
                # Wait for the selection overlay to fade out of the picture.
                if frame is not None:
                    c = self.crop(frame)
                    if self.prev_crop is None or not np.array_equal(c[::4, ::4], self.prev_crop[::4, ::4]):
                        self.t_region_change = now
                    self.prev_crop = c
                since = now - self.t_hide
                if (since >= 0.35 and now - self.t_region_change >= 0.2) or since >= 2.5:
                    self.begin_capture(now)

            elif self.phase == "capture":
                if frame is not None:
                    status = self.stitcher.feed(self.crop(frame))
                    if args.debug and status != "idle":
                        log(status, "y =", self.stitcher.y, "height =", self.stitcher.height)
                    if status == "moved":
                        self.t_move = now
                        self.ever_moved = True
                        self.moved_since_tick = True
                    if status != "idle":
                        self.t_change = now
                if self.finish_requested or self.stitcher.full:
                    return self.finish()
                if self.auto:
                    if self.stitcher.lost:
                        return self.finish()
                    self.auto_step(now)
                elif args.idle > 0 and self.ever_moved and now - self.t_move >= args.idle:
                    return self.finish()
                if now - self.t_tooltip > 0.4 and self.tray is not None:
                    self.t_tooltip = now
                    state = "lost track - scroll back up a little" if self.stitcher.lost \
                        else f"{self.stitcher.height} px captured"
                    self.tray.setToolTip(f"{TITLE}: {state} - click to finish")

        def show_overlay(self):
            self.phase = "select"
            screen = app.primaryScreen()
            pos = props.get("position")
            if pos is not None:
                for s in app.screens():
                    g = s.geometry()
                    if (g.x(), g.y()) == tuple(pos):
                        screen = s
            if self.auto:
                how = ("The area will be scrolled for you until it stops moving.\n"
                       f"To finish early: click the tray icon or run {APP} again.")
            else:
                stop = f"stop scrolling for {args.idle:g} s, " if args.idle > 0 else ""
                how = ("Then scroll down at a steady pace.\n"
                       f"To finish: {stop}click the tray icon, or run {APP} again.")
            hint = "Drag over the area that scrolls   (Esc or right-click cancels)\n" + how
            self.frozen = self.frame
            pix = QtGui.QPixmap.fromImage(to_qimage(self.frozen))
            self.overlay = Overlay(pix, hint, self.region_chosen)
            self.overlay.winId()
            handle = self.overlay.windowHandle()
            if handle is not None and screen is not None:
                handle.setScreen(screen)
            self.overlay.showFullScreen()
            self.overlay.raise_()
            self.overlay.activateWindow()

        def region_chosen(self, rect):
            if self.phase != "select":
                return
            if rect is None:
                self.overlay.hide()
                self.shutdown()
                app.quit()
                return
            fh, fw = self.frozen.shape[:2]
            sx = fw / max(1, self.overlay.width())
            sy = fh / max(1, self.overlay.height())
            x0 = max(0, min(fw - 1, round(rect.left() * sx)))
            y0 = max(0, min(fh - 1, round(rect.top() * sy)))
            x1 = max(x0 + 1, min(fw, round((rect.left() + rect.width()) * sx)))
            y1 = max(y0 + 1, min(fh, round((rect.top() + rect.height()) * sy)))
            self.region = (x0, y0, x1, y1)
            self.logical = rect
            self.overlay.hide()
            self.t_hide = self.t_region_change = time.monotonic()
            self.prev_crop = None
            self.phase = "settle"

        def begin_capture(self, now):
            self.phase = "capture"
            first = self.prev_crop if self.prev_crop is not None else self.crop(self.frozen)
            self.stitcher.feed(first)
            self.t_move = self.t_change = now
            self.finish_requested = False
            if QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
                self.tray = QtWidgets.QSystemTrayIcon(icon)
                menu = QtWidgets.QMenu()
                menu.addAction("Finish and save").triggered.connect(self.request_finish)
                menu.addAction("Cancel").triggered.connect(self.cancel)
                self.tray_menu = menu
                self.tray.setContextMenu(menu)
                self.tray.activated.connect(self.tray_clicked)
                self.tray.setToolTip(f"{TITLE}: capturing - click to finish")
                self.tray.show()
            if self.auto:
                # Park the pointer inside the area, high up: wheel events go to
                # the window under it, and rows near the top are never copied
                # from later frames, so hover effects stay out of the picture.
                r = self.logical
                portal.move_pointer(r.left() + r.width() / 2, r.top() + r.height() * 0.12)
                self.hold_until = now + 0.4   # let hover effects settle first

        def auto_step(self, now):
            """Send wheel input, wait for the picture to settle, repeat.

            Portal backends disagree on how far one "step" scrolls (a full
            wheel notch on some, a tiny fraction of one on others), so the
            amount is calibrated from what the content actually did.
            """
            st = self.stitcher
            if self.tick_t is None:
                if now < self.hold_until or now - self.t_change < 0.15:
                    return
                portal.scroll(self.steps)
                self.pending_steps = self.steps
                self.tick_t = now
                self.moved_since_tick = False
                self.y_at_tick = st.y
                return
            if self.moved_since_tick:
                if now - self.t_change >= 0.18 and now - self.tick_t >= 0.2:
                    moved = st.y - self.y_at_tick
                    if moved > 0:
                        rate = moved / self.pending_steps        # px per step
                        self.rate = rate if self.rate is None else max(rate, 0.6 * self.rate)
                        self.steps = int(max(1, min(4000, (0.5 * st.band) // self.rate)))
                    self.nomove = 0
                    self.tick_t = None
            elif now - self.tick_t >= (0.9 if self.ever_moved else 0.5):
                self.nomove += 1
                self.tick_t = None
                if not self.ever_moved:
                    # Nothing has moved yet: maybe the step was too small to
                    # register. Try bigger ones before giving up.
                    if self.steps >= 64:
                        self.finish()
                    self.steps *= 4
                elif self.nomove >= 3:
                    self.finish()     # reached the end

        def tray_clicked(self, reason):
            if reason == QtWidgets.QSystemTrayIcon.ActivationReason.Trigger:
                self.request_finish()

        def request_finish(self, *_):
            self.finish_requested = True

        def cancel(self, *_):
            self.shutdown()
            app.quit()

        def finish(self):
            self.phase = "done"
            st = self.stitcher
            img = st.result()
            self.shutdown()
            if img is None:
                return app.quit()
            out_dir = Path(args.output_dir).expanduser() if args.output_dir else pictures_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / time.strftime("Scrollshot_%Y%m%d_%H%M%S.png")
            qimg = to_qimage(img)
            if not qimg.save(str(path), "PNG"):
                notify(f"Could not write {path}")
                return app.exit(1)
            log("saved", path, f"({qimg.width()}x{qimg.height()})")
            if not args.no_copy:
                copy_to_clipboard(path)
            note = ""
            if st.accepted == 0:
                note = ("No scrolling was detected, so this is a plain screenshot of the "
                        "area." + (" Auto-scroll had no effect: check that the area scrolls "
                                   "with the mouse wheel." if self.auto else ""))
            elif st.lost:
                note = ("Tracking was lost near the end (scrolled too fast, or the content "
                        "changed under it). The image stops where it lost track.")
            elif st.full:
                note = f"Stopped at the height limit ({args.max_height} px); see --max-height."
            self.result = Result(path, qimg, note)
            self.result.show()
            self.result.raise_()
            self.result.activateWindow()

    ctl = Controller()
    signal.signal(signal.SIGUSR1, lambda *_: ctl.request_finish())
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    code = app.exec()
    # Tear down in a fixed order: letting Python choose the order at exit can
    # destroy the QApplication before its windows, which crashes Qt.
    global _APP
    _APP = app
    ctl.timer.stop()
    for obj in (ctl.result, ctl.overlay, ctl.tray):
        if obj is not None:
            obj.hide()
            obj.deleteLater()
    app.processEvents()
    return code


# ----------------------------------------------------------------------------
# Install helper and entry point
# ----------------------------------------------------------------------------

DESKTOP = """[Desktop Entry]
Type=Application
Name=KScrollShot
GenericName=Scrolling Screenshot
Comment=Capture a scrolling area as one tall image
Exec={exe}
Icon=camera-photo
Categories=Utility;Graphics;
StartupNotify=false
Actions=Auto;

[Desktop Action Auto]
Name=Scrolling Screenshot (auto-scroll)
Exec={exe} --auto
"""


def install():
    src = Path(__file__).resolve()
    bin_dir = Path.home() / ".local" / "bin"
    apps = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "applications"
    bin_dir.mkdir(parents=True, exist_ok=True)
    apps.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / APP
    if src != exe:
        shutil.copy2(src, exe)
    exe.chmod(0o755)
    (apps / f"{APP}.desktop").write_text(DESKTOP.format(exe=exe))
    print(f"Installed {exe}")
    print(f"Installed {apps / (APP + '.desktop')}")
    problems = missing_deps()
    if problems:
        print("\nMissing dependencies:")
        for p in problems:
            print("  -", p)
        print("Install them with:\n  " + PACMAN)
    else:
        print("All dependencies found.")
    print("\nTo bind a key: System Settings > Keyboard > Shortcuts > Add New > Application,"
          "\npick KScrollShot, then set a shortcut for it (and one for its auto-scroll action)."
          "\nPressing the shortcut again while capturing finishes and saves.")
    return 0


def main():
    ap = argparse.ArgumentParser(prog=APP, description="Scrolling screenshots for KDE Plasma (Wayland).")
    ap.add_argument("--auto", action="store_true",
                    help="scroll the area automatically (asks for permission to send input)")
    ap.add_argument("--idle", type=float, default=4.0, metavar="SEC",
                    help="manual mode: finish after this long without scrolling (0 = never; default 4)")
    ap.add_argument("--max-height", type=int, default=60000, metavar="PX",
                    help="stop when the image reaches this height (default 60000)")
    ap.add_argument("--output-dir", metavar="DIR", help="where to save (default: Pictures/Screenshots)")
    ap.add_argument("--no-copy", action="store_true", help="do not copy the result to the clipboard")
    ap.add_argument("--forget-screen", action="store_true",
                    help="ask again which screen to capture")
    ap.add_argument("--install", action="store_true",
                    help="copy to ~/.local/bin and add an application launcher entry")
    ap.add_argument("--debug", action="store_true", help="log every matched frame")
    args = ap.parse_args()
    faulthandler.enable()   # on a native crash, print where Python was

    if args.install:
        return install()
    if signal_running_instance():
        return 0
    problems = missing_deps()
    if problems:
        notify("Missing dependencies: " + "; ".join(problems) + ". Install with: " + PACMAN)
        return 1
    if os.environ.get("XDG_SESSION_TYPE", "wayland") != "wayland":
        log("warning: not a Wayland session; the screen-cast portal may not work here")

    tokens = load_tokens()
    kind = "remote" if args.auto else "cast"
    portal = Portal()
    try:
        fd, node, props, new_token = portal.start(
            args.auto, None if args.forget_screen else tokens.get(kind))
    except PortalError as e:
        if str(e) != "cancelled":
            notify(f"Could not start screen capture: {e}")
        return 1
    if new_token:
        tokens[kind] = new_token
        save_tokens(tokens)
    try:
        grabber = Grabber(fd, node)
    except RuntimeError as e:
        notify(str(e))
        portal.close()
        return 1
    signal.signal(signal.SIGUSR1, lambda *_: None)   # default action would kill us
    try:
        pid_file().write_text(str(os.getpid()))
    except OSError:
        pass
    try:
        return run_gui(args, portal, grabber, props)
    finally:
        try:
            if pid_file().read_text().strip() == str(os.getpid()):
                pid_file().unlink()
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    sys.exit(main())
