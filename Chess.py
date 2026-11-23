SCALE_OVERRIDE = None
ANALYSIS_TIME = 0.6
MONITOR_INDEX = 1
UPDATE_INTERVAL = 1.0

import sys
import ctypes
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import math
import random
import threading
import time

import tkinter as tk
import tkinter.ttk as ttk

import cv2
import mss
import numpy as np
import chess
import chess.engine

SQ = 48
MARGIN = 0.04
COLOR_THRESH = 28


def get_scale():
    if SCALE_OVERRIDE:
        return float(SCALE_OVERRIDE)
    if sys.platform.startswith("win"):
        try:
            v = ctypes.windll.shcore.GetScaleFactorForDevice(0)
            if v:
                return v / 100.0
        except Exception:
            pass
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
            ctypes.windll.user32.ReleaseDC(0, hdc)
            if dpi:
                return dpi / 96.0
        except Exception:
            pass
    return 1.0


def grab(region=None):
    with mss.mss() as sct:
        mon = sct.monitors[MONITOR_INDEX] if MONITOR_INDEX < len(sct.monitors) else sct.monitors[1]
        if region is None:
            box = mon
        else:
            box = {"left": int(region[0]), "top": int(region[1]),
                   "width": int(region[2]), "height": int(region[3])}
        img = np.array(sct.grab(box))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), mon


def split_squares(board_img):
    h, w = board_img.shape[:2]
    out = {}
    for r in range(8):
        for c in range(8):
            y0, y1 = int(r * h / 8), int((r + 1) * h / 8)
            x0, x1 = int(c * w / 8), int((c + 1) * w / 8)
            my, mx = int((y1 - y0) * MARGIN), int((x1 - x0) * MARGIN)
            cell = board_img[y0 + my:y1 - my, x0 + mx:x1 - mx]
            out[(r, c)] = cv2.resize(cell, (SQ, SQ), interpolation=cv2.INTER_AREA)
    return out


def square_features(sq):
    ring = np.concatenate([sq[:3].reshape(-1, 3), sq[-3:].reshape(-1, 3),
                           sq[:, :3].reshape(-1, 3), sq[:, -3:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    diff = np.abs(sq.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = (diff > COLOR_THRESH).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(mask)
    big = [c for c in contours if cv2.contourArea(c) > 0.02 * SQ * SQ]
    if big:
        cv2.drawContours(sil, big, -1, 255, thickness=cv2.FILLED)
    frac = float(np.count_nonzero(sil)) / (SQ * SQ)
    gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
    inner = cv2.erode(sil, np.ones((3, 3), np.uint8))
    if np.count_nonzero(inner) == 0:
        inner = sil
    bright = float(gray[inner > 0].mean()) if np.count_nonzero(inner) else 0.0
    return sil > 0, frac, bright


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 0.0


def screen_to_square(r, c, white_bottom):
    if white_bottom:
        return chess.square(c, 7 - r)
    return chess.square(7 - c, r)


OVERLAY_COLOR = "#010203"
ARROW_COLOR = "#ffee00"
WS_EX_TRANSPARENT = 0x20
WS_EX_NOACTIVATE = 0x08000000
GWL_EXSTYLE = -20
WS_CHILD = 0x40000000


def square_center(square, roi, white_bottom):
    f = chess.square_file(square)
    r = chess.square_rank(square)
    if white_bottom:
        col, row = f, 7 - r
    else:
        col, row = 7 - f, r
    w, h = roi[2], roi[3]
    return (col + 0.5) * w / 8.0, (row + 0.5) * h / 8.0


def apply_click_through(win):
    if not sys.platform.startswith("win"):
        return
    set_ex = getattr(ctypes.windll.user32, "SetWindowLongPtrW", None)
    if set_ex is None:
        set_ex = ctypes.windll.user32.SetWindowLongW
    else:
        set_ex.restype = ctypes.c_ssize_t
    get_long = ctypes.windll.user32.GetWindowLongW
    hwnd = win.winfo_id()
    if get_long(hwnd, -16) & WS_CHILD:
        get_parent = ctypes.windll.user32.GetParent
        get_parent.restype = ctypes.c_void_p
        parent = get_parent(hwnd)
        if parent:
            hwnd = int(parent)
    ex = set_ex(hwnd, GWL_EXSTYLE, 0)
    set_ex(hwnd, GWL_EXSTYLE, ex | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE)


def calibrate(board_img, white_bottom):
    sq = split_squares(board_img)
    empty = {True: sq[(3, 1)], False: sq[(3, 0)]}
    start = chess.Board()
    templates = {}
    bright = {True: [], False: []}
    piece_fracs, empty_fracs = [], []
    for (r, c), img in sq.items():
        mask, frac, br = square_features(img)
        p = start.piece_at(screen_to_square(r, c, white_bottom))
        if p is None:
            empty_fracs.append(frac)
            continue
        piece_fracs.append(frac)
        templates.setdefault(p.symbol(), []).append(mask)
        bright[p.color].append(br)
    if not piece_fracs or not empty_fracs:
        raise ValueError("Calibration failed: the selected area is not a full chessboard.")
    empty_hi = float(np.percentile(empty_fracs, 90))
    piece_lo = float(np.percentile(piece_fracs, 10))
    empty_mid = float(np.median(empty_fracs))
    piece_mid = float(np.median(piece_fracs))
    if piece_mid <= empty_mid:
        raise ValueError("Calibration failed: no pieces detected. The board must be at the "
                         "starting position and the selection must cover the whole 8x8 board.")
    empty_thresh = ((empty_hi + piece_lo) / 2.0 if piece_lo > empty_hi
                    else (empty_mid + piece_mid) / 2.0)
    w_b, b_b = float(np.mean(bright[True])), float(np.mean(bright[False]))
    if abs(w_b - b_b) < 15:
        raise ValueError("Calibration failed: white and black pieces could not be told apart.")
    return {
        "empty": empty,
        "templates": templates,
        "empty_thresh": empty_thresh,
        "color_mid": (w_b + b_b) / 2.0,
        "white_brighter": w_b > b_b,
        "white_bottom": white_bottom,
    }


def recognize(board_img, calib, white_bottom):
    sq = split_squares(board_img)
    board = chess.Board(None)
    for (r, c), img in sq.items():
        mask, frac, br = square_features(img)
        if frac < calib["empty_thresh"]:
            continue
        is_white = (br > calib["color_mid"]) == calib["white_brighter"]
        best, best_s = None, -1.0
        for sym, masks in calib["templates"].items():
            if sym.isupper() != is_white:
                continue
            s = max(iou(mask, m) for m in masks)
            if s > best_s:
                best, best_s = sym, s
        if best:
            board.set_piece_at(screen_to_square(r, c, white_bottom), chess.Piece.from_symbol(best))
    return board


def set_castling(board):
    fen = ""
    P = board.piece_at
    if P(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
        if P(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE): fen += "K"
        if P(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE): fen += "Q"
    if P(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
        if P(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK): fen += "k"
        if P(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK): fen += "q"
    board.set_castling_fen(fen or "-")


def infer_turn(prev_board, new_placement, prev_turn):
    if prev_board is not None:
        for turn in (prev_turn, not prev_turn):
            b = prev_board.copy(stack=False)
            b.turn = turn
            for mv in b.legal_moves:
                b.push(mv)
                if b.board_fen() == new_placement:
                    return not turn
                b.pop()
        b = prev_board.copy(stack=False)
        b.turn = prev_turn
        for m1 in list(b.legal_moves):
            b.push(m1)
            for m2 in list(b.legal_moves):
                b.push(m2)
                if b.board_fen() == new_placement:
                    return prev_turn
                b.pop()
            b.pop()
    return not prev_turn


def format_score(score):
    w = score.white()
    if w.is_mate():
        m = w.mate()
        return f"Mate {'+' if m > 0 else '-'}{abs(m)}"
    return f"{w.score() / 100:+.2f}"


QUALITY_LABELS = (
    (0, "Best", "#1565c0"),
    (20, "Excellent", "#1b5e20"),
    (50, "Very good", "#2e7d32"),
    (100, "Good", "#558b2f"),
    (200, "Mistake", "#b26a00"),
    (300, "Blunder", "#e65100"),
    (float("inf"), "Fiasco", "#c62828"),
)
BADGE_MS = 1000


def score_cp(sc, side):
    s = sc.pov(side)
    if s.is_mate():
        m = s.mate()
        return 100000.0 - 100.0 * abs(m) if m > 0 else -(100000.0 - 100.0 * abs(m))
    return float(s.score() or 0)


def quality_label(loss):
    cp = max(0.0, loss)
    for th, label, color in QUALITY_LABELS:
        if cp <= th:
            return label, color
    return "Fiasco", "#c62828"


def format_pov(sc, side):
    s = sc.pov(side)
    if s.is_mate():
        m = s.mate()
        return f"Mate {'+' if m > 0 else '-'}{abs(m)}"
    return f"{s.score() / 100:+.2f}"


VAR_PV_PLIES = 10


def pv_san(board, pv, max_plies=VAR_PV_PLIES):
    b = board.copy(stack=False)
    out = []
    for i, mv in enumerate(pv[:max_plies]):
        if mv not in b.legal_moves:
            break
        san = b.san(mv)
        if b.turn == chess.WHITE:
            out.append(f"{b.fullmove_number}.{san}")
        elif i == 0:
            out.append(f"{b.fullmove_number}...{san}")
        else:
            out.append(san)
        b.push(mv)
    return " ".join(out)


def build_variations_text(board, infos):
    lines = []
    for i, info in enumerate(infos, 1):
        pv, sc = info.get("pv"), info.get("score")
        if not pv or sc is None:
            continue
        lines.append(f"{i}. {format_score(sc)}  {pv_san(board, pv)}")
    return "\n".join(lines) if lines else "Stockfish returned no variations."


def advantage_from_info(info):
    sc = info.get("score")
    if sc is None:
        return None
    w = sc.white()
    if w.is_mate():
        m = w.mate()
        evl = f"Mate {'+' if m > 0 else '-'}{abs(m)}"
    else:
        evl = f"{w.score() / 100:+.2f}"
    wdl = info.get("wdl")
    if wdl is not None:
        try:
            wins, draws, losses = wdl.white()
            total = wins + draws + losses
            if total > 0:
                return 100.0 * (wins + draws / 2.0) / total, evl
        except Exception:
            pass
    if w.is_mate():
        return (100.0 if m > 0 else 0.0), evl
    cp = float(w.score() or 0)
    return 100.0 / (1.0 + math.exp(-0.00368208 * cp)), evl


VK_Q = 0x51
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


def q_pressed():
    if not sys.platform.startswith("win"):
        return False
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(VK_Q) & 0x8000)
    except Exception:
        return False


def agent_complexity(board, infos, move):
    turn = board.turn

    def cp_of(info):
        try:
            return score_cp(info["score"], turn)
        except Exception:
            return 0.0

    c1 = cp_of(infos[0]) if infos else 0.0
    c2 = cp_of(infos[1]) if len(infos) > 1 else c1
    notes = []
    s = 0.35
    mate_in = None
    if infos:
        try:
            pov = infos[0]["score"].pov(turn)
            if pov.is_mate():
                mate_in = pov.mate()
        except Exception:
            pass
    if mate_in is not None:
        if mate_in <= 1:
            s -= 0.25
            notes.append("mate in 1")
        elif mate_in <= 3:
            s += 0.25
            notes.append(f"mate in {mate_in}, delicate")
        else:
            s += 0.1
            notes.append("mate path")
    elif c1 <= -500:
        s -= 0.2
        notes.append("bad position")
    elif c1 >= 600:
        s -= 0.15
        notes.append("won position")
    else:
        gap = c1 - c2
        if gap >= 250:
            s += 0.3
            notes.append("only good move")
        elif gap <= 40:
            s -= 0.15
            notes.append("equivalent moves")
        if abs(c1) < 80:
            s += 0.15
            notes.append("balanced fine position")
    try:
        pc = board.piece_at(move.from_square)
        val = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}.get(pc.piece_type, 0) if pc else 0
        if pc is not None and val >= 3:
            b2 = board.copy(stack=False)
            b2.push(move)
            if b2.attackers(not turn, move.to_square) and not b2.attackers(turn, move.to_square):
                s += 0.45 if val >= 9 else 0.35
                notes.append("sacrifice-like")
    except Exception:
        pass
    if board.is_capture(move):
        s -= 0.1
        notes.append("capture")
    if board.gives_check(move):
        s -= 0.05
    if board.is_castling(move):
        s -= 0.2
        notes.append("castling")
    if move.promotion:
        s += 0.2
        notes.append("promotion")
    if board.fullmove_number <= 6:
        s -= 0.25
        notes.append("opening move")
    s = max(0.0, min(1.0, s))
    return s, (", ".join(notes) if notes else "normal move")


def agent_delay(s, fmin, fmax, lmin, lmax):
    if s <= 0.3:
        return random.uniform(fmin, fmax), "fast"
    if s >= 0.75:
        return random.uniform(lmin, lmax), "long"
    t = (s - 0.3) / 0.45
    d = fmax + t * (lmin - fmax)
    d *= random.uniform(0.9, 1.1)
    d = max(fmin, min(lmax, d))
    return d, "medium"


def play_move_on_screen(move, roi, wb):
    if not sys.platform.startswith("win"):
        raise RuntimeError("mouse control works only on Windows")
    from ctypes import wintypes
    u32 = ctypes.windll.user32
    half = roi[2] / 8.0 * 0.18

    def pt(sq):
        x, y = square_center(sq, roi, wb)
        return (roi[0] + x + random.uniform(-half, half),
                roi[1] + y + random.uniform(-half, half))

    fx, fy = pt(move.from_square)
    tx, ty = pt(move.to_square)
    p = wintypes.POINT()
    u32.GetCursorPos(ctypes.byref(p))
    ox, oy = p.x, p.y
    u32.SetCursorPos(int(fx), int(fy))
    time.sleep(random.uniform(0.10, 0.18))
    u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(random.uniform(0.07, 0.13))
    steps = random.randint(9, 14)
    for i in range(1, steps + 1):
        t = i / steps
        e = t * t * (3.0 - 2.0 * t)
        u32.SetCursorPos(int(fx + (tx - fx) * e), int(fy + (ty - fy) * e))
        time.sleep(random.uniform(0.010, 0.022))
    time.sleep(random.uniform(0.08, 0.16))
    u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    if move.promotion:
        time.sleep(0.6)
        u32.SetCursorPos(int(tx), int(ty))
        time.sleep(0.12)
        u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.06)
        u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.35)
    u32.SetCursorPos(ox, oy)


OPENINGS = [
    ("Italian Game", "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d4 exd4 6.cxd4 Bb4+ 7.Bd2 Bxd2+ 8.Nbxd2 d5"),
    ("Ruy Lopez (Closed)", "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O 9.h3 Nb8"),
    ("Sicilian Najdorf", "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 6.Be2 e5 7.Nb3 Be7"),
    ("French Defense (Classical)", "1.e4 e6 2.d4 d5 3.Nc3 Nf6 4.Bg5 Be7 5.e5 Nfd7 6.Bxe7 Qxe7 7.Nf3 Nc6"),
    ("Caro-Kann (Classical)", "1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 5.Ng3 Bg6 6.h4 h6 7.Nf3 Nd7 8.h5 Bh7"),
    ("Queen's Gambit Declined", "1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 h6 7.Bh4 b6 8.cxd5 Nxd5"),
    ("London System", "1.d4 d5 2.Bf4 Nf6 3.e3 e6 4.Nf3 c5 5.c3 Nc6 6.Nbd2 Bd6 7.Bg3 O-O 8.Bd3"),
    ("King's Indian Defense", "1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5 7.O-O Nc6 8.d5 Ne7"),
    ("Scotch Game", "1.e4 e5 2.Nf3 Nc6 3.d4 exd4 4.Nxd4 Nf6 5.Nc3 Bb4 6.Nxc6 dxc6 7.Bd3 O-O"),
    ("Scandinavian Defense", "1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5 4.d4 Nf6 5.Nf3 c6 6.Bc4 Bf5"),
    ("Queen's Gambit Accepted", "1.d4 d5 2.c4 dxc4 3.Nf3 Nf6 4.e3 e6 5.Bxc4 c5 6.O-O a6 7.Qe2 b5"),
    ("Slav Defense", "1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 dxc4 5.a4 Bf5 6.e3 e6 7.Bxc4 Bb4"),
    ("English Opening", "1.c4 e5 2.Nc3 Nf6 3.Nf3 Nc6 4.g3 d5 5.cxd5 Nxd5 6.Bg2 Nb6 7.O-O Be7"),
    ("Pirc Defense", "1.e4 d6 2.d4 Nf6 3.Nc3 g6 4.Nf3 Bg7 5.Be2 O-O 6.O-O c6 7.a4 Nbd7"),
    ("Vienna Game", "1.e4 e5 2.Nc3 Nf6 3.f4 d5 4.fxe5 Nxe4 5.Nf3 Be7 6.d4 O-O 7.Bd3"),
]


def opening_book(moves_text):
    board = chess.Board()
    book = {}
    for tok in moves_text.split():
        if tok[0].isdigit():
            tok = tok.split(".", 1)[1]
        mv = board.parse_san(tok)
        book[(board.board_fen(), board.turn)] = mv
        board.push(mv)
    return book


def find_stockfish():
    app_file = Path(sys.executable) if getattr(sys, "frozen", False) else Path(__file__)
    return app_file.resolve().parent / "Mrburky.exe"


def main():
    state = {
        "roi": None, "calib": None, "white_bottom": True, "turn": chess.WHITE,
        "last_fen": None, "candidate": None, "analyzed": False, "prev_board": None,
        "best_move": None, "cfg_ver": 0, "last_move_txt": None, "badge": None,
    }
    cfg = {"skill": 20, "time": ANALYSIS_TIME, "multipv": 1, "paused": False,
           "only_mine": True, "show_arrow": True, "analysis_on": True, "opening": None,
           "variations": False, "agent_on": False, "agent_fast_min": 2.5,
           "agent_fast_max": 7.0, "agent_long_min": 30.0, "agent_long_max": 45.0}
    books = [opening_book(moves) for _, moves in OPENINGS]
    lock = threading.Lock()
    stop = threading.Event()

    root = tk.Tk()
    root.title("Chess Assistant")
    root.attributes("-topmost", True)
    root.geometry("+0+0")
    root.minsize(280, 260)
    root.resizable(False, True)

    nb = ttk.Notebook(root)
    tab1 = ttk.Frame(nb)
    tab2 = ttk.Frame(nb)
    nb.add(tab1, text="Main")
    nb.add(tab2, text="Move Analysis")
    tab3 = ttk.Frame(nb)
    nb.add(tab3, text="Openings")
    tab4 = ttk.Frame(nb)
    nb.add(tab4, text="Variations")
    tab5 = ttk.Frame(nb)
    nb.add(tab5, text="Agent")
    nb.pack(fill="both", expand=True)

    analysis_txt = ("Move Analysis\n"
                    "\n"
                    "This tab shows the Stockfish quality\n"
                    "rating only for the move YOU played:\n"
                    "the quality label and the estimated\n"
                    "centipawn loss (if any).\n"
                    "\n"
                    "After the move is played the result\n"
                    "appears here and briefly as a badge\n"
                    "on the destination square.\n"
                    "No result is produced for opponent\n"
                    "moves.")
    analysis_lbl = tk.Label(tab2, text=analysis_txt, font=("Consolas", 11),
                            justify="left", anchor="nw", width=34, bg="black",
                            fg="lime", wraplength=320, padx=8, pady=8)

    def set_analysis(text):
        root.after(0, lambda t=text: analysis_lbl.config(text=t))

    def toggle_analysis():
        with lock:
            cfg["analysis_on"] = not cfg["analysis_on"]
            on = cfg["analysis_on"]
            if not on:
                state["badge"] = None
        btn_analysis.config(text="Move analysis: On" if on else "Move analysis: Off")
        if on:
            set_analysis("Move analysis on.\nYour next move will be\nevaluated.")
        else:
            set_analysis("Move analysis off.\nNo result will be produced\nfor your own moves.")
        overlay_refresh()

    btn_analysis = tk.Button(tab2, text="Move analysis: On",
                             font=("Consolas", 11), command=toggle_analysis)
    btn_analysis.pack(fill="x", padx=8, pady=(8, 0))
    analysis_lbl.pack(fill="both", expand=True)

    lbl = tk.Label(tab1, text="Starting...", font=("Consolas", 14, "bold"),
                   justify="left", anchor="w", width=32, bg="black", fg="lime")
    lbl.pack(fill="x", padx=4, pady=(4, 2))

    def show(text):
        root.after(0, lambda t=text: lbl.config(text=t))

    engine = None
    stockfish_path = find_stockfish()
    if stockfish_path.is_file():
        try:
            engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
            try:
                engine.configure({"UCI_ShowWDL": True})
            except Exception:
                pass
        except Exception as e:
            lbl.config(text=f"Could not start Stockfish:\n{stockfish_path.name}\n({e})")
    else:
        lbl.config(text="Engine not found!\nPut Mrburky.exe in the\nsame folder as the app.")

    frm = tk.Frame(tab1)
    frm.pack(fill="x")

    def select_board():
        root.withdraw()
        root.update()
        try:
            img, mon = grab()
            scale = get_scale()
            disp = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
            win = "Select the board - ENTER confirm, C cancel"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            x, y, w, h = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            if w > 0 and h > 0:
                roi = (mon["left"] + int(x * scale), mon["top"] + int(y * scale),
                       int(w * scale), int(h * scale))
                with lock:
                    state.update(roi=roi, calib=None, last_fen=None, candidate=None,
                                 analyzed=False, prev_board=None, best_move=None,
                                 last_move_txt=None, badge=None)
                show("Board selected.\nWith pieces at the start\nposition, press 'Calibrate'.")
                set_analysis(analysis_txt)
            else:
                show("Selection canceled.")
        except Exception as e:
            show(f"Board selection error:\n{e}")
        finally:
            root.deiconify()
            root.attributes("-topmost", True)
            overlay_refresh()

    def do_calibrate():
        with lock:
            roi, wb = state["roi"], state["white_bottom"]
        if roi is None:
            show("Press 'Select Board' first.")
            return
        try:
            img, _ = grab(roi)
            calib = calibrate(img, wb)
            with lock:
                state.update(calib=calib, last_fen=None, candidate=None,
                             analyzed=False, prev_board=None, best_move=None,
                             last_move_txt=None, badge=None)
                turn = state["turn"]
            btn_turn.config(text="Turn: White" if turn else "Turn: Black")
            show("Calibration done.\nAnalysis starting...")
            set_analysis(analysis_txt)
            overlay_refresh()
        except Exception as e:
            show(f"Calibration error:\n{e}")

    def toggle_side():
        with lock:
            state["white_bottom"] = not state["white_bottom"]
            state["calib"] = None
            state["last_fen"] = None
            state["best_move"] = None
            wb = state["white_bottom"]
        btn_side.config(text="White at bottom" if wb else "Black at bottom")
        show("Orientation changed.\nPress 'Calibrate' again.")
        overlay_refresh()

    def toggle_turn():
        with lock:
            state["turn"] = not state["turn"]
            state["analyzed"] = False
            state["best_move"] = None
            t = state["turn"]
        btn_turn.config(text="Turn: White" if t else "Turn: Black")
        overlay_refresh()

    def toggle_pause():
        with lock:
            cfg["paused"] = not cfg["paused"]
            p = cfg["paused"]
        btn_pause.config(text="Analysis: On" if not p else "Analysis: Paused")
        if p:
            show("Analysis paused.")
        overlay_refresh()

    def reset_state():
        with lock:
            state.update(last_fen=None, candidate=None, analyzed=False, prev_board=None,
                         best_move=None, last_move_txt=None, badge=None)
        show("Reset.\nThe next reading will be\nanalyzed again.")
        set_analysis(analysis_txt)
        overlay_refresh()

    f_btn = ("Consolas", 11)

    tk.Button(frm, text="Select Board", font=f_btn, command=select_board).grid(
        row=0, column=0, sticky="ew", padx=3, pady=3, ipady=3)
    tk.Button(frm, text="Calibrate", font=f_btn, command=do_calibrate).grid(
        row=0, column=1, sticky="ew", padx=3, pady=3, ipady=3)
    btn_side = tk.Button(frm, text="White at bottom", font=f_btn, command=toggle_side)
    btn_side.grid(row=1, column=0, sticky="ew", padx=3, pady=3, ipady=3)
    btn_turn = tk.Button(frm, text="Turn: White", font=f_btn, command=toggle_turn)
    btn_turn.grid(row=1, column=1, sticky="ew", padx=3, pady=3, ipady=3)
    btn_pause = tk.Button(frm, text="Analysis: On", font=f_btn, command=toggle_pause)
    btn_pause.grid(row=2, column=0, sticky="ew", padx=3, pady=3, ipady=3)
    tk.Button(frm, text="Reset", font=f_btn, command=reset_state).grid(
        row=2, column=1, sticky="ew", padx=3, pady=3, ipady=3)
    frm.columnconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    setfrm = tk.LabelFrame(tab1, text="Settings", font=("Consolas", 10, "bold"))
    setfrm.pack(fill="x", padx=6, pady=(0, 6))

    skill_var = tk.IntVar(value=20)
    time_var = tk.DoubleVar(value=ANALYSIS_TIME)
    mpv_var = tk.IntVar(value=1)

    tk.Label(setfrm, text="Skill level (0-20):", font=f_btn).grid(
        row=0, column=0, sticky="w", padx=4, pady=2)
    tk.Spinbox(setfrm, from_=0, to=20, width=5, font=f_btn,
               textvariable=skill_var).grid(row=0, column=1, padx=4, pady=2)
    tk.Label(setfrm, text="Think time (s):", font=f_btn).grid(
        row=1, column=0, sticky="w", padx=4, pady=2)
    tk.Spinbox(setfrm, from_=0.2, to=10.0, increment=0.2, width=5, font=f_btn,
               textvariable=time_var).grid(row=1, column=1, padx=4, pady=2)
    tk.Label(setfrm, text="Alternative moves:", font=f_btn).grid(
        row=2, column=0, sticky="w", padx=4, pady=2)
    tk.Spinbox(setfrm, from_=1, to=3, width=5, font=f_btn,
               textvariable=mpv_var).grid(row=2, column=1, padx=4, pady=2)

    only_var = tk.BooleanVar(value=True)
    tk.Checkbutton(setfrm, text="Suggest only on my turn", font=f_btn,
                   variable=only_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=4)
    arrow_var = tk.BooleanVar(value=True)
    tk.Checkbutton(setfrm, text="Show move arrow", font=f_btn,
                   variable=arrow_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=4)

    def on_cfg_change(*_):
        try:
            skill = max(0, min(20, int(skill_var.get())))
            think = max(0.2, min(10.0, float(time_var.get())))
            mpv = max(1, min(3, int(mpv_var.get())))
            mine = bool(only_var.get())
            show_arrow = bool(arrow_var.get())
        except (ValueError, tk.TclError):
            return
        with lock:
            cfg.update(skill=skill, time=think, multipv=mpv, only_mine=mine,
                       show_arrow=show_arrow)
            state["cfg_ver"] += 1
            state["analyzed"] = False
        overlay_refresh()

    for v in (skill_var, time_var, mpv_var, only_var, arrow_var):
        v.trace_add("write", on_cfg_change)

    open_lbl = tk.Label(tab3, text="Pick an opening and press 'Start opening'.\n"
                                   "When it is your turn the opening move\n"
                                   "is shown with an arrow.\n"
                                   "If the opponent leaves the opening,\n"
                                   "Stockfish takes over.",
                        font=("Consolas", 10), justify="left", anchor="nw", width=34,
                        bg="black", fg="lime", wraplength=320, padx=8, pady=8)
    open_list = tk.Listbox(tab3, height=len(OPENINGS), font=("Consolas", 10),
                           exportselection=False)
    for name, _ in OPENINGS:
        open_list.insert("end", name)
    open_list.selection_set(0)

    def show_opening_line(*_):
        sel = open_list.curselection()
        if sel:
            name, moves = OPENINGS[sel[0]]
            open_lbl.config(text=f"Selected: {name}\n{moves}")

    def set_opening(idx):
        with lock:
            cfg["opening"] = idx
            state["cfg_ver"] += 1
            state["analyzed"] = False
            state["best_move"] = None
        if idx is None:
            open_lbl.config(text="Opening off.\nStockfish is doing normal analysis.")
        else:
            open_lbl.config(text=f"Opening: {OPENINGS[idx][0]}\n"
                                 "When it is your turn the book move is shown.\n"
                                 "If the opponent leaves the opening, Stockfish takes over.")
        overlay_refresh()

    def start_opening():
        sel = open_list.curselection()
        if sel:
            set_opening(sel[0])

    open_list.bind("<<ListboxSelect>>", show_opening_line)
    open_btns = tk.Frame(tab3)
    tk.Button(open_btns, text="Start opening", font=f_btn, command=start_opening).pack(
        side="left", fill="x", expand=True, padx=3, pady=3)
    tk.Button(open_btns, text="Stop opening", font=f_btn,
              command=lambda: set_opening(None)).pack(
        side="left", fill="x", expand=True, padx=3, pady=3)
    open_btns.pack(fill="x", padx=8, pady=(8, 0))
    open_list.pack(fill="x", padx=8, pady=(8, 0))
    open_lbl.pack(fill="both", expand=True, padx=8, pady=8)

    var_txt = ("Variation analysis is off.\n"
               "Turn it on with the button above:\n"
               "Stockfish will analyze the current\n"
               "position and show the 5 most likely\n"
               "options, each with its follow-up move\n"
               "line. Updates automatically when the\n"
               "position changes.")

    def set_variations(text):
        root.after(0, lambda t=text: var_lbl.config(text=t))

    def toggle_variations():
        with lock:
            on = cfg["variations"] = not cfg["variations"]
        if on:
            btn_var.config(text="Variation analysis: On")
            set_variations("Variation analysis on.\nIn the next Stockfish analysis the\n"
                           "5 most likely options and their\nfollow-up moves will appear here.")
        else:
            btn_var.config(text="Variation analysis: Off")
            set_variations(var_txt)
        overlay_refresh()

    btn_var = tk.Button(tab4, text="Variation analysis: Off",
                        font=("Consolas", 11), command=toggle_variations)
    btn_var.pack(fill="x", padx=8, pady=(8, 0))

    adv_lbl = tk.Label(tab4, text="Advantage: waiting for analysis...",
                       font=("Consolas", 11, "bold"), bg="black", fg="lime")
    adv_lbl.pack(fill="x", padx=8, pady=(8, 2))
    adv_cv = tk.Canvas(tab4, height=22, bg="#404040", highlightthickness=0)
    adv_cv.pack(fill="x", padx=8, pady=(0, 4))
    adv_state = {"p": 50.0, "evl": ""}

    def draw_advantage():
        cw = max(160, adv_cv.winfo_width())
        p = adv_state["p"]
        adv_cv.delete("all")
        adv_cv.create_rectangle(0, 0, cw, 22, fill="#2b2b2b", outline="")
        adv_cv.create_rectangle(0, 0, cw * p / 100.0, 22, fill="#e8e8e8", outline="")
        adv_cv.create_line(cw / 2, 0, cw / 2, 22, fill="#e53935", width=2)

    def set_advantage(p, evl):
        adv_state["p"] = max(0.0, min(100.0, float(p)))
        adv_state["evl"] = evl

        def upd():
            pw = adv_state["p"]
            ev = f"  ({adv_state['evl']})" if adv_state["evl"] else ""
            adv_lbl.config(text=f"Advantage: White {pw:.1f}% — Black {100.0 - pw:.1f}%{ev}")
            draw_advantage()

        root.after(0, upd)

    adv_cv.bind("<Configure>", lambda _e: root.after_idle(draw_advantage))

    var_lbl = tk.Label(tab4, text=var_txt, font=("Consolas", 10),
                       justify="left", anchor="nw", width=34, bg="black",
                       fg="lime", wraplength=320, padx=8, pady=8)
    var_lbl.pack(fill="both", expand=True)

    agent_txt = ("Agent is off.\n"
                 "\n"
                 "When the agent is on it plays the\n"
                 "best move itself on the board when\n"
                 "it is your turn. The response time\n"
                 "is set by the move's value, beauty\n"
                 "and logic: simple moves are played\n"
                 "fast, critical and subtle moves take\n"
                 "longer.\n"
                 "\n"
                 "Pressing Q turns the agent off at\n"
                 "any time.")
    agent_lbl = tk.Label(tab5, text=agent_txt, font=("Consolas", 10),
                         justify="left", anchor="nw", width=34, bg="black",
                         fg="lime", wraplength=320, padx=8, pady=8)

    def set_agent(text):
        root.after(0, lambda t=text: agent_lbl.config(text=t))

    def toggle_agent():
        with lock:
            on = cfg["agent_on"] = not cfg["agent_on"]
        btn_agent.config(text="Agent: ON" if on else "Agent: Off")
        if on:
            set_agent("Agent on.\nThe move will be played when\nit is your turn. Q = off.")
        else:
            set_agent(agent_txt)
        overlay_refresh()

    btn_agent = tk.Button(tab5, text="Agent: Off",
                          font=("Consolas", 11), command=toggle_agent)
    btn_agent.pack(fill="x", padx=8, pady=(8, 0))

    afrm = tk.LabelFrame(tab5, text="Response times (s)", font=("Consolas", 10, "bold"))
    afrm.pack(fill="x", padx=6, pady=(6, 0))
    afmin_var = tk.DoubleVar(value=2.5)
    afmax_var = tk.DoubleVar(value=7.0)
    almin_var = tk.DoubleVar(value=30.0)
    almax_var = tk.DoubleVar(value=45.0)
    for row, (txt, var) in enumerate((
            ("Fast min:", afmin_var), ("Fast max:", afmax_var),
            ("Long min:", almin_var), ("Long max:", almax_var))):
        tk.Label(afrm, text=txt, font=f_btn).grid(row=row, column=0, sticky="w",
                                                  padx=4, pady=1)
        tk.Spinbox(afrm, from_=0.5, to=300.0, increment=0.5, width=6, font=f_btn,
                   textvariable=var).grid(row=row, column=1, padx=4, pady=1)

    def on_agent_cfg_change(*_):
        try:
            vals = [float(v.get()) for v in (afmin_var, afmax_var, almin_var, almax_var)]
        except (ValueError, tk.TclError):
            return
        fmin, fmax = sorted((max(0.5, min(60.0, vals[0])), max(0.5, min(60.0, vals[1]))))
        lmin, lmax = sorted((max(1.0, min(300.0, vals[2])), max(1.0, min(300.0, vals[3]))))
        with lock:
            cfg.update(agent_fast_min=fmin, agent_fast_max=fmax,
                       agent_long_min=lmin, agent_long_max=lmax)

    for v in (afmin_var, afmax_var, almin_var, almax_var):
        v.trace_add("write", on_agent_cfg_change)

    agent_lbl.pack(fill="both", expand=True)

    def agent_off(msg):
        with lock:
            cfg["agent_on"] = False
        root.after(0, lambda: btn_agent.config(text="Agent: Off"))
        if msg:
            set_agent(msg)

    def q_watcher():
        while not stop.is_set():
            with lock:
                aon = cfg["agent_on"]
            if aon and q_pressed():
                agent_off("Q pressed — agent turned off.")
                while not stop.is_set() and q_pressed():
                    stop.wait(0.05)
            stop.wait(0.1)

    def agent_play(board, move, infos, placement, turn):
        with lock:
            roi = state["roi"]
            calib = state["calib"]
            wb = state["white_bottom"]
            fmin = cfg["agent_fast_min"]
            fmax = cfg["agent_fast_max"]
            lmin = cfg["agent_long_min"]
            lmax = cfg["agent_long_max"]
        if roi is None or calib is None:
            set_agent("Agent: no board selected/calibrated.")
            return
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()
        if infos:
            s, reason = agent_complexity(board, infos, move)
            delay, cat = agent_delay(s, fmin, fmax, lmin, lmax)
        else:
            reason, cat = "book/opening move", "fast"
            delay = random.uniform(fmin, fmax)
        set_agent(f"Move chosen: {san}\nReason: {reason}\n"
                  f"Category: {cat} · will be played\nin ~{delay:.1f} s. Q = cancel.")
        end_t = time.time() + delay
        last_rem = -1
        while time.time() < end_t:
            stop.wait(0.5)
            with lock:
                alive = (cfg["agent_on"] and not cfg["paused"]
                         and state["last_fen"] == placement and state["turn"] == turn
                         and state["roi"] == roi and state["calib"] is calib)
            if not alive:
                return
            rem = max(1, int(round(end_t - time.time())))
            if rem != last_rem:
                last_rem = rem
                set_agent(f"{san} will be played ({cat})\nRemaining: {rem} s\nQ = cancel")
            try:
                img2, _ = grab(roi)
                if recognize(img2, calib, wb).board_fen() != placement:
                    set_agent("Position changed — move canceled.")
                    return
            except Exception:
                pass
        with lock:
            alive = (cfg["agent_on"] and not cfg["paused"]
                     and state["last_fen"] == placement and state["turn"] == turn
                     and state["roi"] == roi and state["calib"] is calib)
        if not alive:
            return
        try:
            play_move_on_screen(move, roi, wb)
            set_agent(f"Move played: {san}\nCategory: {cat} ({delay:.1f} s)\n"
                      "Waiting for opponent...")
        except Exception as e:
            set_agent(f"Could not play the move:\n{e}")
        stop.wait(1.0)

    if engine is not None:
        lbl.config(text="Start with 'Select Board'.")

    overlay_win, overlay_canvas, overlay_ok = None, None, False
    overlay_refresh_err = False

    def make_overlay():
        nonlocal overlay_win, overlay_canvas, overlay_ok
        if not sys.platform.startswith("win"):
            return
        win = None
        try:
            win = tk.Toplevel(root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            win.attributes("-transparentcolor", OVERLAY_COLOR)
            cv = tk.Canvas(win, width=100, height=100, bg=OVERLAY_COLOR,
                           highlightthickness=0, bd=0)
            cv.pack(fill="both", expand=True)
            overlay_win, overlay_canvas = win, cv
            apply_click_through(win)
            overlay_ok = True
        except Exception as e:
            overlay_ok = False
            if win is not None:
                try:
                    win.destroy()
                except Exception:
                    pass
            try:
                lbl.config(text="Could not set up arrow overlay:\n%s" % e)
            except Exception:
                pass

    def overlay_refresh():
        nonlocal overlay_refresh_err
        if not overlay_ok:
            return
        try:
            with lock:
                roi = state["roi"]
                wb = state["white_bottom"]
                move = state["best_move"]
                analyzed = state["analyzed"]
                paused = cfg["paused"]
                turn = state["turn"]
                only_mine = cfg["only_mine"]
                show_arrow = cfg["show_arrow"]
                badge = state["badge"]
            badge_on = badge is not None and time.time() < badge[3]
            arrow_on = (roi is not None and show_arrow and move is not None and analyzed
                        and not paused and (not only_mine or turn == wb))
            if roi is None or (not arrow_on and not badge_on):
                overlay_win.withdraw()
                return
            left, top, w, h = roi
            overlay_canvas.config(width=w, height=h)
            overlay_win.geometry(f"{w}x{h}+{left}+{top}")
            overlay_win.deiconify()
            overlay_canvas.delete("all")
            if arrow_on:
                sx, sy = square_center(move.from_square, roi, wb)
                tx, ty = square_center(move.to_square, roi, wb)
                d = min(w, h)
                lw = max(12, d // 28)
                olw = lw + 4
                al = max(30, d // 11)
                aw = max(30, d // 13)
                ak = max(6, d // 100)
                overlay_canvas.create_line(sx, sy, tx, ty, width=olw,
                                           fill="#0a0a0a", arrow=tk.LAST,
                                           arrowshape=(al + 8, aw + 8, ak + 2))
                overlay_canvas.create_line(sx, sy, tx, ty, width=lw,
                                           fill=ARROW_COLOR, arrow=tk.LAST,
                                           arrowshape=(al, aw, ak))
                overlay_canvas.create_oval(sx - lw, sy - lw, sx + lw, sy + lw,
                                           fill=ARROW_COLOR, outline="")
                overlay_canvas.create_oval(tx - lw, ty - lw, tx + lw, ty + lw,
                                           fill=ARROW_COLOR, outline="")
            if badge_on:
                bs, blabel, bcolor, _ = badge
                bx, by = square_center(bs, roi, wb)
                d = min(w, h)
                bf = max(11, d // 38)
                bw = min(w - 4, max(92, d // 6))
                bh = min(h - 4, max(40, d // 14))
                bx = max(bw / 2.0 + 2, min(w - bw / 2.0 - 2, bx))
                oy = by - bh / 2.0 - d // 16
                if oy < 2:
                    oy = by + bh / 2.0 + d // 16
                oy = max(2, min(h - bh - 2, oy))
                overlay_canvas.create_oval(bx - bw / 2.0, oy, bx + bw / 2.0, oy + bh,
                                           fill=bcolor, outline="#ffffff",
                                           width=max(2, d // 200))
                overlay_canvas.create_text(bx, oy + bh / 2.0, text=blabel,
                                           fill="#ffffff",
                                           font=("Segoe UI", bf, "bold"))
        except Exception as e:
            if not overlay_refresh_err:
                overlay_refresh_err = True
                lbl.config(text="Could not draw the arrow:\n%s" % e)

    make_overlay()

    def worker():
        applied_skill = None

        def eval_advantage(bd):
            try:
                with lock:
                    th = cfg["time"]
                info = engine.analyse(bd, chess.engine.Limit(time=th))
                if isinstance(info, list):
                    info = info[0]
                return advantage_from_info(info)
            except Exception:
                return None

        while not stop.is_set():
            try:
                with lock:
                    roi, calib, wb = state["roi"], state["calib"], state["white_bottom"]
                    paused = cfg["paused"]
                if engine is None or roi is None or calib is None or paused:
                    stop.wait(UPDATE_INTERVAL)
                    continue

                img, _ = grab(roi)
                placement = recognize(img, calib, wb).board_fen()

                refresh_overlay = False
                with lock:
                    if placement != state["candidate"]:
                        state["candidate"] = placement
                        stable = False
                        if placement != state["last_fen"]:
                            state["best_move"] = None
                            state["analyzed"] = False
                            refresh_overlay = True
                    else:
                        stable = True
                    if stable and placement != state["last_fen"]:
                        if state["last_fen"] is not None:
                            state["turn"] = infer_turn(state["prev_board"], placement, state["turn"])
                        state["last_fen"] = placement
                        state["analyzed"] = False
                        state["best_move"] = None
                        refresh_overlay = True
                    need = stable and not state["analyzed"]
                    turn = state["turn"]
                    ver = state["cfg_ver"]

                if refresh_overlay:
                    root.after(0, overlay_refresh)

                if need:
                    agent_infos = []
                    board = chess.Board(None)
                    board.set_board_fen(placement)
                    board.turn = turn
                    set_castling(board)
                    root.after(0, lambda t=turn: btn_turn.config(
                        text="Turn: White" if t else "Turn: Black"))

                    with lock:
                        prev_bd = state["prev_board"]
                    if (cfg["analysis_on"] and prev_bd is not None
                            and prev_bd.turn == wb and prev_bd.turn == (not turn)):
                        matches = []
                        pb = prev_bd.copy(stack=False)
                        for mv2 in pb.legal_moves:
                            pb.push(mv2)
                            if pb.board_fen() == placement:
                                matches.append(mv2)
                            pb.pop()
                        if len(matches) == 1:
                            played = matches[0]
                            try:
                                san = prev_bd.san(played)
                            except Exception:
                                san = played.uci()
                            try:
                                with lock:
                                    sk, th = cfg["skill"], cfg["time"]
                                mt = max(0.8, th)
                                try:
                                    engine.configure({"Skill Level": 20})
                                except Exception:
                                    pass
                                try:
                                    bi = engine.analyse(prev_bd, chess.engine.Limit(time=mt), multipv=1)
                                    pi = engine.analyse(prev_bd, chess.engine.Limit(time=mt),
                                                        root_moves=[played], multipv=1)
                                finally:
                                    try:
                                        engine.configure({"Skill Level": sk})
                                    except Exception:
                                        pass
                                    applied_skill = sk

                                if isinstance(bi, dict):
                                    bi = [bi]
                                if isinstance(pi, dict):
                                    pi = [pi]
                                bsc = bi[0].get("score") if bi else None
                                psc = pi[0].get("score") if pi else None
                                if bsc is None or psc is None:
                                    raise ValueError("no move result")
                                if cfg["analysis_on"]:
                                    best_cp = score_cp(bsc, prev_bd.turn)
                                    played_cp = score_cp(psc, prev_bd.turn)
                                    loss = best_cp - played_cp
                                    if loss < 0:
                                        loss = 0.0
                                    label, color = quality_label(loss)
                                    txt = (f"Last move (SAN): {san}\n"
                                           f"Quality (Stockfish, approx): {label}\n"
                                           f"Estimated loss: ~{int(round(loss))} cp\n"
                                           f"Best evaluation: {format_pov(bsc, prev_bd.turn)}\n"
                                           f"Played move: {format_pov(psc, prev_bd.turn)}")
                                    with lock:
                                        state["last_move_txt"] = txt
                                        state["badge"] = (played.to_square, label, color,
                                                          time.time() + BADGE_MS / 1000.0)
                                    set_analysis(txt)
                                    root.after(0, overlay_refresh)
                                    root.after(BADGE_MS, overlay_refresh)
                            except Exception:
                                set_analysis("Could not compute the Stockfish\n"
                                             "result for the last move (engine\nreturned nothing).")
                        else:
                            set_analysis("Could not evaluate the last move:\nno single legal move change was\nrecognized on the board.")

                    with lock:
                        opening = cfg["opening"]
                        var_on = cfg["variations"]
                    book_hit = None
                    if opening is not None and turn == wb:
                        book_hit = books[opening].get((placement, turn))

                    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                        with lock:
                            state["best_move"] = None
                        show("Could not read the position:\nking not found.\nRecalibrate.")
                    elif not board.is_valid():
                        with lock:
                            state["best_move"] = None
                        show("Invalid position!\nThe turn may be wrong\n('Turn' button) or recognition\nfailed.")
                    elif board.is_game_over():
                        with lock:
                            state["best_move"] = None
                        show(f"Game over: {board.result()}")
                    elif book_hit is not None:
                        with lock:
                            state["best_move"] = book_hit
                        show(f"Opening: {OPENINGS[opening][0]}\n"
                             f"Book move: {board.san(book_hit)} "
                             f"({chess.square_name(book_hit.from_square)} → "
                             f"{chess.square_name(book_hit.to_square)})\n"
                             "If there is a deviation Stockfish takes over.")
                        if var_on:
                            set_variations(f"Opening book: {OPENINGS[opening][0]}\n"
                                           "Book move available; variations\n"
                                           "update once the book is left.")
                        adv = eval_advantage(board)
                        if adv is not None:
                            set_advantage(adv[0], adv[1])
                    elif cfg["only_mine"] and turn != wb:
                        me = "White" if wb else "Black"
                        show(f"You: {me}\nOpponent thinking...")
                        if var_on:
                            set_variations("Opponent thinking...\nWhen it is your turn the\n"
                                           "5 most likely options will be shown.")
                        adv = eval_advantage(board)
                        if adv is not None:
                            set_advantage(adv[0], adv[1])
                        with lock:
                            ver = state["cfg_ver"]
                            state["best_move"] = None
                    else:
                        with lock:
                            skill, think, mpv = cfg["skill"], cfg["time"], cfg["multipv"]
                            variations = cfg["variations"]
                            agent_on = cfg["agent_on"]
                            ver = state["cfg_ver"]
                        eng_mpv = max(5, mpv) if variations else mpv
                        if agent_on and turn == wb:
                            eng_mpv = max(3, eng_mpv)
                            think = max(think, 0.8)
                        if skill != applied_skill:
                            try:
                                engine.configure({"Skill Level": skill})
                            except Exception:
                                pass
                            applied_skill = skill
                        infos = engine.analyse(board, chess.engine.Limit(time=think),
                                               multipv=eng_mpv)
                        if isinstance(infos, dict):
                            infos = [infos]
                        agent_infos = infos
                        if variations:
                            set_variations(build_variations_text(board, infos))
                        if infos:
                            adv = advantage_from_info(infos[0])
                            if adv is not None:
                                set_advantage(adv[0], adv[1])
                        side = "White" if turn else "Black"
                        lines = [f"Turn: {side} (you: {'White' if wb else 'Black'})"]
                        found = 0
                        best_move = None
                        for info in infos[:mpv]:
                            pv = info.get("pv")
                            sc = info.get("score")
                            if not pv or sc is None:
                                continue
                            if best_move is None:
                                best_move = pv[0]
                            found += 1
                            try:
                                san = board.san(pv[0])
                            except Exception:
                                san = pv[0].uci()
                            num = f"{found}. " if mpv > 1 else ""
                            pc = board.piece_at(pv[0].from_square)
                            if pc is not None:
                                names = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
                                lines.append(f"{num}{names[pc.piece_type]}: "
                                             f"{chess.square_name(pv[0].from_square)} → "
                                             f"{chess.square_name(pv[0].to_square)} "
                                             f"({san}) ({format_score(sc)})")
                            else:
                                lines.append(f"{num}{san}  ({format_score(sc)})")
                        with lock:
                            state["best_move"] = best_move
                        show("\n".join(lines) if found else "Stockfish returned no move.")
                    with lock:
                        if (state["last_fen"] == placement and state["turn"] == turn
                                and state["cfg_ver"] == ver):
                            state["analyzed"] = True
                            state["prev_board"] = board
                    root.after(0, overlay_refresh)
                    with lock:
                        aon = cfg["agent_on"]
                        bm = state["best_move"]
                    if (aon and bm is not None and turn == wb
                            and board.is_valid() and not board.is_game_over()):
                        try:
                            agent_play(board, bm, agent_infos, placement, turn)
                        except Exception as e:
                            set_agent(f"Agent error:\n{e}")
            except chess.engine.EngineTerminatedError:
                show("Stockfish died!\nRestart the program.")
                break
            except Exception as e:
                show(f"Error:\n{e}")
            stop.wait(UPDATE_INTERVAL)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    threading.Thread(target=q_watcher, daemon=True).start()

    def on_close():
        stop.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    stop.set()
    th.join(timeout=ANALYSIS_TIME + 3)
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
