#!/usr/bin/env python3
"""
Schemata WebSocket server — PyTorch backend for schemata.html

Usage:
    python server.py [--port 8765] [--host 0.0.0.0]

All ALRNN computation (training, Fisher, phi, free-run) runs here.
schemata.html connects via ws://localhost:<port>/ws and handles rendering only.
"""
import argparse
import asyncio
import copy
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from alrnn import ALRNN
from piagets import (
    PIAGETSContinual, diff_phi, signature_fisher, _normalize,
    _eval_mse_quick, train_with_ewc,
)
import systems

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[schemata] compute device: {DEVICE}")

# ── Thread pool ──────────────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train")

# ── Preset task streams ─────────────────────────────────────────────────────
_N, _DT_LOR, _DT = 5000, 0.01, 0.05
_CLS_COLOR = {0: "#3e6e8e", 1: "#b98e3e", 2: "#6f7355"}
_DEFAULT_EP = 400


def _mk(name, gen, cls, epochs=_DEFAULT_EP, label=None):
    return dict(name=name, gen=gen, cls=cls, epochs=epochs,
                label=label or name, color=_CLS_COLOR[cls])


PRESETS = {
    "standard_10": [
        _mk("Lor_r28",   lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=28)[0].astype("f4"),   0),
        _mk("Tor_0.382", lambda: systems.torus(n=_N, dt=_DT, omega2=0.382)[0].astype("f4"),   1),
        _mk("VdP_m1.5",  lambda: systems.van_der_pol(n=_N, dt=_DT, mu=1.5)[0].astype("f4"),  2),
        _mk("Lor_r35",   lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=35)[0].astype("f4"),   0),
        _mk("Tor_0.618", lambda: systems.torus(n=_N, dt=_DT, omega2=0.618)[0].astype("f4"),  1),
        _mk("VdP_m3.0",  lambda: systems.van_der_pol(n=_N, dt=_DT, mu=3.0)[0].astype("f4"),  2),
        _mk("Lor_r40",   lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=40)[0].astype("f4"),   0),
        _mk("VdP_m5.0",  lambda: systems.van_der_pol(n=_N, dt=_DT, mu=5.0)[0].astype("f4"),  2),
        _mk("Tor_0.271", lambda: systems.torus(n=_N, dt=_DT, omega2=0.271)[0].astype("f4"),  1),
        _mk("Lor_r45",   lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=45)[0].astype("f4"),   0),
    ],
    "lorenz_ramp": [
        _mk("Lor_r28", lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=28)[0].astype("f4"), 0),
        _mk("Lor_r35", lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=35)[0].astype("f4"), 0),
        _mk("Lor_r40", lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=40)[0].astype("f4"), 0),
        _mk("Lor_r45", lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=45)[0].astype("f4"), 0),
    ],
    "class_alternating": [
        _mk("Lor_r28",   lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=28)[0].astype("f4"),  0),
        _mk("Tor_0.382", lambda: systems.torus(n=_N, dt=_DT, omega2=0.382)[0].astype("f4"), 1),
        _mk("VdP_m1.5",  lambda: systems.van_der_pol(n=_N, dt=_DT, mu=1.5)[0].astype("f4"),2),
        _mk("Lor_r35",   lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=35)[0].astype("f4"),  0),
        _mk("Tor_0.618", lambda: systems.torus(n=_N, dt=_DT, omega2=0.618)[0].astype("f4"), 1),
        _mk("VdP_m3.0",  lambda: systems.van_der_pol(n=_N, dt=_DT, mu=3.0)[0].astype("f4"),2),
    ],
    "pathological": [
        _mk("Lor_r28",   lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=28)[0].astype("f4"),  0,
            label="Lor_r28 (anchor)"),
        _mk("VdP_m1.5",  lambda: systems.van_der_pol(n=_N, dt=_DT, mu=1.5)[0].astype("f4"), 2,
            label="VdP_m1.5 ← probe stress"),
        _mk("Tor_0.382", lambda: systems.torus(n=_N, dt=_DT, omega2=0.382)[0].astype("f4"), 1),
        _mk("VdP_m3.0",  lambda: systems.van_der_pol(n=_N, dt=_DT, mu=3.0)[0].astype("f4"), 2),
        _mk("Lor_r35",   lambda: systems.lorenz(n=_N, dt=_DT_LOR, rho=35)[0].astype("f4"),  0),
    ],
}

# ── Custom stream system registry ────────────────────────────────────────────
_SYS_CLS = {"lorenz": 0, "rossler": 0, "torus": 1, "limit_cycle": 2, "van_der_pol": 2}

def _make_custom_task(sys_name: str, param: float, epochs: int) -> dict:
    if sys_name == "lorenz":
        rho = param
        gen = lambda r=rho: systems.lorenz(n=_N, dt=_DT_LOR, rho=r)[0].astype("f4")
        name = f"Lor_r{rho:.0f}"
    elif sys_name == "rossler":
        c = param
        gen = lambda cv=c: systems.rossler(n=_N, dt=_DT, c=cv)[0].astype("f4")
        name = f"Ros_c{c:.1f}"
    elif sys_name == "torus":
        omega = param
        gen = lambda w=omega: systems.torus(n=_N, dt=_DT, omega2=w)[0].astype("f4")
        name = f"Tor_{omega:.3f}"
    elif sys_name == "limit_cycle":
        omega = param
        gen = lambda w=omega: systems.limit_cycle(n=_N, dt=_DT, omega=w)[0].astype("f4")
        name = f"LC_w{omega:.2f}"
    elif sys_name == "van_der_pol":
        mu = param
        gen = lambda m=mu: systems.van_der_pol(n=_N, dt=_DT, mu=m)[0].astype("f4")
        name = f"VdP_m{mu:.1f}"
    else:
        raise ValueError(f"Unknown system: {sys_name}")
    cls = _SYS_CLS.get(sys_name, 0)
    return _mk(name, gen, cls, epochs)

# ── Session ──────────────────────────────────────────────────────────────────

class Session:
    def __init__(self):
        self.model: ALRNN | None = None
        self.cl: PIAGETSContinual | None = None
        self.stream: list = []
        self.task_data: list = []
        self.current_task: int = -1
        self.trained: set = set()
        self.snapshots: dict = {}
        self.oracle_phis: dict = {}
        self.phi_history: list = []
        self.lock = asyncio.Lock()
        self.M, self.P, self.rank, self.d = 48, 16, 12, 3
        self.use_cl = True
        self.use_merge = True
        self.use_qr = True
        self.fisher_type = "sig"
        self.adaptive = False
        self.lam = 5.0
        self.lam_assim = 10.0
        self.lam_accom = 2.0
        self.assim_ratio = 10.0
        self.lr_train = 6e-4          # default 0.2 × 3e-3
        self.train_pause = threading.Event()   # set = paused
        self.train_cancel = threading.Event()  # set = cancel/reset


_sess = Session()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cpu_copy(model: ALRNN) -> ALRNN:
    """Cheap CPU copy for analysis functions — avoids mutating the training model."""
    if next(model.parameters()).is_cuda:
        return copy.deepcopy(model).cpu()
    return model


def _new_model(s: Session, seed: int = 0) -> ALRNN:
    torch.manual_seed(seed)
    m = ALRNN(latent_dim=s.M, obs_dim=s.d, P=s.P, rank=s.rank)
    return m.to(DEVICE)


def _new_cl(s: Session) -> PIAGETSContinual:
    return PIAGETSContinual(
        lam_ewc=s.lam,
        lam_assim=s.lam_assim if s.adaptive else None,
        lam_accom=s.lam_accom if s.adaptive else None,
        assim_mse_ratio=s.assim_ratio,
        sig_fisher_kwargs=dict(n_avg=3, T_warmup=100, T_track=200, beta=10.0),
    )


def _get_data(s: Session, t: int) -> np.ndarray:
    while len(s.task_data) <= t:
        s.task_data.append(None)
    if s.task_data[t] is None:
        s.task_data[t] = s.stream[t]["gen"]()
    return s.task_data[t]


def _phi(model: ALRNN) -> np.ndarray:
    m = _cpu_copy(model)
    with torch.no_grad():
        return diff_phi(m, T_warmup=100, T_track=200, beta=10.0).numpy().astype(np.float32)


def _freerun(model: ALRNN, n: int = 3000, seed_data=None):
    """Free-run with teacher-forcing warmup so latent state is on the attractor manifold."""
    m = _cpu_copy(model)
    with torch.no_grad():
        z = torch.zeros(m.M, dtype=torch.float32)
        if seed_data is not None and len(seed_data) > 1:
            z[:m.d] = torch.from_numpy(seed_data[0, :m.d]).float()
            warmup = min(500, len(seed_data) - 1)
            for t in range(warmup):
                z = m.step(z)
                z[:m.d] = torch.from_numpy(seed_data[t + 1, :m.d]).float()
        z0_full = z.numpy()
    traj, pats = m.free_run(z0_full[:m.d], n, return_patterns=True, z0_full=z0_full)
    powers = (1 << np.arange(m.P))
    regions = (pats.astype(np.int32) * powers).sum(1).tolist()
    return traj.astype(np.float32), regions


def _eval_mse(model: ALRNN, data: np.ndarray) -> float:
    m = _cpu_copy(model)
    return float(_eval_mse_quick(m, data))


def _weights(s: Session) -> dict:
    m = s.model
    if m is None:
        return {}
    mc = _cpu_copy(m)
    out = dict(
        WB=mc.W_B.data.numpy().flatten().tolist(),
        WA=mc.W_A.data.numpy().flatten().tolist(),
        A=mc.A.data.numpy().tolist(),
        h=mc.h.data.numpy().tolist(),
        M=mc.M, P=mc.P, rank=mc.rank,
    )
    cl = s.cl
    out["FB"] = cl._F_B.cpu().numpy().flatten().tolist() if (cl and cl._F_B is not None) else []
    out["FA"] = cl._F_A_diag.cpu().numpy().tolist()      if (cl and cl._F_A_diag is not None) else []
    out["Fh"] = cl._F_h.cpu().numpy().tolist()           if (cl and cl._F_h is not None) else []
    return out


def _phi_pca(phi_mat: np.ndarray):
    mean = phi_mat.mean(0)
    X = phi_mat - mean
    C = (X.T @ X) / max(len(phi_mat), 1)
    vecs = []
    Cw = C.copy()
    rng = np.random.default_rng(0)
    for _ in range(2):
        v = rng.standard_normal(C.shape[0])
        for _ in range(300):
            v = Cw @ v
            nrm = np.linalg.norm(v)
            if nrm < 1e-14:
                break
            v /= nrm
        lam = float(v @ Cw @ v)
        vecs.append(v.tolist())
        Cw -= lam * np.outer(v, v)
    return mean.tolist(), vecs


class _TrainStop(Exception):
    pass


# ── Training coroutine ───────────────────────────────────────────────────────

async def _do_train(ws, s: Session, t_idx: int, epochs: int, lr: float,
                    send, send_status, msg_queue: asyncio.Queue):
    task = s.stream[t_idx]
    await send_status(f"Loading data for {task['name']}…")
    td = await asyncio.to_thread(_get_data, s, t_idx)
    use_cl = s.use_cl and s.cl is not None and t_idx > 0
    use_sig = s.fisher_type == "sig"
    probe_mode = None

    async with s.lock:
        if use_cl and s.use_qr:
            s.cl.qr_init(s.model)

        if use_cl and s.adaptive:
            lam, mode, _ = s.cl.probe_from_task_start(_cpu_copy(s.model), td)
            s.cl.lam_ewc = lam
            probe_mode = mode
            await send({"type": "probe", "mode": mode, "lam": lam, "task": task["name"]})

        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def cb(ep, recon, ewc):
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "progress", "ep": ep + 1, "total": epochs,
                       "recon": float(recon), "ewc": float(ewc)}), loop)
            if (ep + 1) % 20 == 0:
                tr, rg = _freerun(s.model, n=800, seed_data=td)
                asyncio.run_coroutine_threadsafe(
                    q.put({"type": "freerun_preview",
                           "xyz": tr.flatten().tolist(), "regions": rg}), loop)
            # Pause: block training thread here until resumed or cancelled
            while s.train_pause.is_set() and not s.train_cancel.is_set():
                time.sleep(0.05)
            if s.train_cancel.is_set():
                raise _TrainStop()

        await send_status(f"Training {task['name']} ({epochs} ep) on {DEVICE}…")
        fut = loop.run_in_executor(_executor, lambda: train_with_ewc(
            s.model, td,
            cl=s.cl if use_cl else None,
            epochs=epochs, seq_len=120, batch=64, lr=lr,
            alpha=0.5, alpha_end=0.05,
            device=DEVICE,
            epoch_callback=cb,
            log=lambda _: None,
            current_class=task.get("cls"),
        ))

        cancelled = False
        while not fut.done():
            # Drain control messages from the shared WS queue
            while not msg_queue.empty():
                try:
                    raw_ctrl = msg_queue.get_nowait()
                    if raw_ctrl is None:
                        cancelled = True; s.train_cancel.set(); break
                    ctrl = json.loads(raw_ctrl)
                    ctrl_op = ctrl.get("op", "")
                    if ctrl_op == "pause":
                        s.train_pause.set()
                        await send_status("Paused")
                    elif ctrl_op == "resume":
                        s.train_pause.clear()
                        await send_status("Resumed")
                    elif ctrl_op == "reset":
                        cancelled = True; s.train_cancel.set()
                    elif ctrl_op == "ping":
                        await send({"type": "pong"})
                    # other ops (next, set_stream, etc.) silently discarded during training
                except asyncio.QueueEmpty:
                    break
            # Process training progress messages
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.05)
                await send(msg)
            except asyncio.TimeoutError:
                pass

        try:
            await fut
        except _TrainStop:
            pass
        while not q.empty():
            await send(q.get_nowait())

        if cancelled:
            # Apply reset and return without sending task_done
            s.train_pause.clear()
            s.train_cancel.clear()
            s.model = _new_model(s)
            s.cl    = _new_cl(s) if s.use_cl else None
            s.trained.clear()
            s.phi_history.clear()
            s.current_task = 0 if s.stream else -1
            await send({"type": "reset_ok", "task_idx": s.current_task})
            await send_status("Model reset")
            return

        if use_cl:
            m_cpu = _cpu_copy(s.model)
            await asyncio.to_thread(s.cl.store_task, m_cpu, td, task.get("cls"), use_sig)
            if s.use_merge:
                with torch.no_grad():
                    s.model.W_B.data.copy_(s.cl.B_merged.to(DEVICE))
                    if s.cl.A_merge is not None:
                        s.model.W_A.data.copy_(s.cl.A_merge.to(DEVICE))

        phi_vec = await asyncio.to_thread(_phi, s.model)
        traj, regions = await asyncio.to_thread(_freerun, s.model, 3000, td)
        mse = await asyncio.to_thread(_eval_mse, s.model, td)
        s.trained.add(t_idx)
        s.current_task = t_idx
        s.phi_history.append({"t": t_idx, "phi": phi_vec.tolist(),
                               "name": task["name"], "mse": mse})
        w = _weights(s)

    await send({
        "type": "task_done", "t": t_idx, "name": task["name"],
        "phi": phi_vec.tolist(), "n_regions": len(set(regions)),
        "xyz": traj.flatten().tolist(), "regions": regions,
        "mse": mse, "probe_mode": probe_mode, **w,
    })


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_DIR = os.path.dirname(os.path.abspath(__file__))

def _build_page() -> str:
    with open(os.path.join(_DIR, "piagets.html"), encoding="utf-8") as f:
        html = f.read()
    three_path = os.path.join(_DIR, "three.min.js")
    if os.path.exists(three_path):
        with open(three_path, encoding="utf-8") as f:
            three_js = f.read()
        html = html.replace(
            '<script src="/three.min.js"></script>',
            f'<script>{three_js}</script>'
        )
    return html

from fastapi.responses import HTMLResponse, FileResponse

@app.get("/")
async def serve_ui():
    return HTMLResponse(_build_page())

@app.get("/overview.html")
async def serve_overview():
    p = os.path.join(_DIR, "overview.html")
    with open(p) as f:
        return HTMLResponse(f.read())

@app.get("/three.min.js")
async def serve_three():
    return FileResponse(os.path.join(_DIR, "three.min.js"), media_type="application/javascript")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    s = _sess
    s.train_pause.clear()
    s.train_cancel.clear()
    _stop = asyncio.Event()
    _msg_queue: asyncio.Queue = asyncio.Queue()

    async def send(msg: dict):
        try:
            await ws.send_json(msg)
        except Exception:
            pass

    async def send_status(msg: str, level: str = "info"):
        await send({"type": "status", "msg": msg, "level": level})

    async def _reader():
        """Continuously read WS messages into queue so main loop and training pump can share them."""
        try:
            while not _stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                    await _msg_queue.put(raw)
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            await _msg_queue.put(None)
        except Exception:
            await _msg_queue.put(None)

    async def _keepalive():
        while not _stop.is_set():
            await asyncio.sleep(15)
            if _stop.is_set(): break
            try:
                await ws.send_json({"type": "ping"})
            except Exception:
                break

    asyncio.ensure_future(_keepalive())
    reader_task = asyncio.ensure_future(_reader())

    try:
        while True:
            raw = await _msg_queue.get()
            if raw is None:
                break  # disconnect
            data = json.loads(raw)
            op = data.get("op", "")

            if op in ("ping",) or data.get("type") == "pong":
                # ping from client → pong; pong from client (response to our ping) → ignore
                if op == "ping":
                    await send({"type": "pong"})

            elif op == "init":
                async with s.lock:
                    s.M    = int(data.get("M", 16))
                    s.P    = int(data.get("P", 6))
                    s.rank = int(data.get("rank", 6))
                    s.lam  = float(data.get("lam", 5.0))
                    s.model = _new_model(s)
                    s.cl    = _new_cl(s) if s.use_cl else None
                    s.trained.clear()
                    s.phi_history.clear()
                    s.current_task = 0 if s.stream else -1
                await send({"type": "ready", "M": s.M, "P": s.P, "rank": s.rank,
                            "task_idx": s.current_task, "device": DEVICE})

            elif op == "set_stream":
                preset = data.get("preset", "standard_10")
                if preset not in PRESETS:
                    await send_status(f"Unknown preset: {preset}", "error"); continue
                async with s.lock:
                    s.stream = PRESETS[preset]
                    s.task_data = []
                    s.current_task = 0
                    s.trained.clear()
                    s.oracle_phis.clear()
                    s.phi_history.clear()
                    s.model = _new_model(s)
                    s.cl    = _new_cl(s) if s.use_cl else None
                tasks_info = [{"idx": i, "name": t["name"], "label": t["label"],
                               "cls": t["cls"], "color": t["color"], "epochs": t["epochs"]}
                              for i, t in enumerate(s.stream)]
                # Send lightweight stream info first (keeps WS alive during data gen)
                await send({"type": "stream_info", "tasks": tasks_info, "preset": preset})
                # Bundle task-0 data separately so client can draw immediately
                td0 = await asyncio.to_thread(_get_data, s, 0)
                stride0 = max(1, len(td0) // 5000)
                await send({"type": "data", "task_idx": 0,
                            "name": s.stream[0]["name"],
                            "xyz": td0[::stride0].flatten().tolist(),
                            "color": s.stream[0].get("color", "#c87028")})

            elif op == "set_custom_stream":
                specs = data.get("tasks", [])
                if not specs:
                    await send_status("No tasks provided", "warn"); continue
                try:
                    custom = [_make_custom_task(t["system"], float(t["param"]),
                                               int(t.get("epochs", _DEFAULT_EP)))
                              for t in specs]
                except Exception as e:
                    await send_status(f"Bad task spec: {e}", "error"); continue
                async with s.lock:
                    s.stream = custom
                    s.task_data = []
                    s.current_task = 0
                    s.trained.clear()
                    s.oracle_phis.clear()
                    s.phi_history.clear()
                    s.model = _new_model(s)
                    s.cl    = _new_cl(s) if s.use_cl else None
                tasks_info = [{"idx": i, "name": t["name"], "label": t["label"],
                               "cls": t["cls"], "color": t["color"], "epochs": t["epochs"]}
                              for i, t in enumerate(s.stream)]
                await send({"type": "stream_info", "tasks": tasks_info, "preset": "custom"})
                td0 = await asyncio.to_thread(_get_data, s, 0)
                stride0 = max(1, len(td0) // 5000)
                await send({"type": "data", "task_idx": 0,
                            "name": s.stream[0]["name"],
                            "xyz": td0[::stride0].flatten().tolist(),
                            "color": s.stream[0].get("color", "#c87028")})

            elif op == "set_cl":
                s.use_cl      = bool(data.get("use_cl", True))
                s.use_merge   = bool(data.get("use_merge", True))
                s.use_qr      = bool(data.get("use_qr", True))
                s.fisher_type = str(data.get("fisher", "sig"))
                s.adaptive    = bool(data.get("adaptive", False))
                s.lam         = float(data.get("lam", 5.0))
                s.lam_assim   = float(data.get("lam_assim", 10.0))
                s.lam_accom   = float(data.get("lam_accom", 2.0))
                s.assim_ratio = float(data.get("assim_ratio", 10.0))
                if s.cl is not None:
                    s.cl.lam_ewc         = s.lam
                    s.cl.lam_assim       = s.lam_assim if s.adaptive else None
                    s.cl.lam_accom       = s.lam_accom if s.adaptive else None
                    s.cl.assim_mse_ratio = s.assim_ratio
                await send({"type": "cl_updated", "use_cl": s.use_cl,
                            "lam": s.lam, "adaptive": s.adaptive})

            elif op == "pause":
                s.train_pause.set()
                await send_status("Paused")

            elif op == "resume":
                s.train_pause.clear()

            elif op == "train":
                t_idx = int(data.get("task_idx", s.current_task))
                if t_idx < 0 or t_idx >= len(s.stream):
                    await send_status("Invalid task_idx", "error"); continue
                if s.model is None:
                    await send_status("Model not initialised", "error"); continue
                epochs = int(data.get("epochs", s.stream[t_idx]["epochs"]))
                lr = float(data.get("lr", s.lr_train)); s.lr_train = lr
                await _do_train(ws, s, t_idx, epochs, lr, send, send_status, _msg_queue)

            elif op == "next":
                if not s.stream:
                    await send_status("No stream loaded", "warn"); continue
                if s.model is None:
                    await send_status("Model not initialised", "error"); continue
                nxt = next((i for i in range(len(s.stream)) if i not in s.trained), None)
                if nxt is None:
                    await send_status("All tasks trained!", "info"); continue
                s.current_task = nxt
                td_vis = await asyncio.to_thread(_get_data, s, nxt)
                stride = max(1, len(td_vis) // 5000)
                sub = td_vis[::stride]
                await send({"type": "advance", "task_idx": nxt, "name": s.stream[nxt]["name"],
                            "xyz": sub.flatten().tolist(),
                            "color": s.stream[nxt].get("color", "#c87028")})
                epochs = int(data.get("epochs", s.stream[nxt]["epochs"]))
                lr = float(data.get("lr", s.lr_train)); s.lr_train = lr
                await _do_train(ws, s, nxt, epochs, lr, send, send_status, _msg_queue)

            elif op == "freerun":
                if s.model is None: continue
                n = int(data.get("steps", 3000))
                seed_data = None
                if s.current_task >= 0 and len(s.task_data) > s.current_task:
                    td = s.task_data[s.current_task]
                    if td is not None: seed_data = td
                async with s.lock:
                    traj, regions = await asyncio.to_thread(_freerun, s.model, n, seed_data)
                await send({"type": "freerun", "xyz": traj.flatten().tolist(),
                            "regions": regions, "n_regions": len(set(regions))})

            elif op == "get_data":
                t_idx = int(data.get("task_idx", 0))
                if t_idx < 0 or t_idx >= len(s.stream): continue
                td = await asyncio.to_thread(_get_data, s, t_idx)
                stride = max(1, len(td) // 5000)
                sub = td[::stride]
                await send({"type": "data", "task_idx": t_idx,
                            "name": s.stream[t_idx]["name"],
                            "xyz": sub.flatten().tolist(),
                            "color": s.stream[t_idx]["color"]})

            elif op == "get_weights":
                async with s.lock:
                    await send({"type": "weights", **_weights(s)})

            elif op == "get_phi":
                if s.model is None: continue
                async with s.lock:
                    phi_vec = await asyncio.to_thread(_phi, s.model)
                M, P = s.M, s.P
                await send({"type": "phi", "phi": phi_vec.tolist(),
                            "groups": {
                                "eigenvals": phi_vec[:M-P].tolist(),
                                "act_mean":  phi_vec[M-P:M-P+P].tolist(),
                                "act_std":   phi_vec[M-P+P:M-P+2*P].tolist(),
                                "topo":      [float(phi_vec[-2])],
                                "lyap":      [float(phi_vec[-1])],
                            }})

            elif op == "get_fisher":
                if s.model is None: continue
                await send_status("Computing signature Fisher (exact autograd)…")
                async with s.lock:
                    m_cpu = _cpu_copy(s.model)
                    raw = await asyncio.to_thread(signature_fisher, m_cpu, 3, 100, 200, 10.0)
                FB = _normalize(raw.get("W_B", torch.zeros(s.M, s.rank)))
                FA = _normalize(raw.get("A",   torch.zeros(s.M)))
                Fh = _normalize(raw.get("h",   torch.zeros(s.M)))
                await send({"type": "fisher_done",
                            "FB": FB.numpy().flatten().tolist(),
                            "FA": FA.numpy().tolist(),
                            "Fh": Fh.numpy().tolist()})

            elif op == "get_oracle_phis":
                if not s.stream: continue
                await send_status(f"Training {len(s.stream)} oracle models…")
                phis_out = []
                for i, task in enumerate(s.stream):
                    if i not in s.oracle_phis:
                        td = await asyncio.to_thread(_get_data, s, i)
                        m = _new_model(s)
                        await asyncio.to_thread(
                            train_with_ewc, m, td, None, task["epochs"],
                            120, 64, 3e-3, 0.5, 0.05, 0.0,
                            DEVICE, None, lambda _: None)
                        pv = await asyncio.to_thread(_phi, m)
                        s.oracle_phis[i] = pv
                        await send_status(f"Oracle {i+1}/{len(s.stream)}: {task['name']} done")
                    phis_out.append({"idx": i, "phi": s.oracle_phis[i].tolist(),
                                     "name": task["name"], "cls": task.get("cls", 0),
                                     "color": task.get("color", "#888")})
                phi_mat = np.stack([p["phi"] for p in phis_out])
                pca_mean, pca_vecs = _phi_pca(phi_mat)
                await send({"type": "oracle_phis", "phis": phis_out,
                            "pca_mean": pca_mean, "pca_vecs": pca_vecs})

            elif op == "get_mse_matrix":
                if not s.stream or not s.trained: continue
                task_list = sorted(s.trained)
                current_mses = []
                async with s.lock:
                    for i in task_list:
                        td = await asyncio.to_thread(_get_data, s, i)
                        mse = await asyncio.to_thread(_eval_mse, s.model, td)
                        current_mses.append({"task_idx": i, "name": s.stream[i]["name"], "mse": mse})
                snap_rows = {}
                for slot, snap in s.snapshots.items():
                    m = _new_model(s); m.load_state_dict(snap["state"])
                    snap_mses = []
                    for i in task_list:
                        td = await asyncio.to_thread(_get_data, s, i)
                        snap_mses.append(await asyncio.to_thread(_eval_mse, m, td))
                    snap_rows[str(slot)] = snap_mses
                await send({"type": "mse_matrix", "current": current_mses,
                            "snapshots": snap_rows,
                            "tasks": [s.stream[i]["name"] for i in task_list]})

            elif op == "snapshot":
                slot  = int(data.get("slot", 0)) % 6
                label = str(data.get("label", f"t={s.current_task}"))
                if s.model is None: continue
                async with s.lock:
                    # store as CPU state_dict (always portable)
                    state = {k: v.cpu().clone() for k, v in s.model.state_dict().items()}
                    pv  = await asyncio.to_thread(_phi, s.model)
                    mse = None
                    if s.current_task >= 0 and len(s.task_data) > s.current_task:
                        td = s.task_data[s.current_task]
                        if td is not None:
                            mse = await asyncio.to_thread(_eval_mse, s.model, td)
                s.snapshots[slot] = dict(state=state, label=label,
                                          phi=pv.tolist(), mse=mse,
                                          task=s.current_task)
                await send({"type": "snapshot_saved", "slot": slot, "label": label,
                            "phi": pv.tolist(), "mse": mse})

            elif op == "load_snapshot":
                slot = int(data.get("slot", 0))
                if slot not in s.snapshots:
                    await send_status(f"Slot {slot} is empty", "warn"); continue
                snap = s.snapshots[slot]
                seed_data = None
                if s.current_task >= 0 and len(s.task_data) > s.current_task:
                    seed_data = s.task_data[s.current_task]
                async with s.lock:
                    s.model.load_state_dict(
                        {k: v.to(DEVICE) for k, v in snap["state"].items()})
                    traj, regions = await asyncio.to_thread(_freerun, s.model, 3000, seed_data)
                await send_status(f"Loaded: {snap['label']}")
                await send({"type": "freerun", "xyz": traj.flatten().tolist(),
                            "regions": regions, "n_regions": len(set(regions))})

            elif op == "reset":
                async with s.lock:
                    s.model = _new_model(s)
                    s.cl    = _new_cl(s) if s.use_cl else None
                    s.trained.clear()
                    s.phi_history.clear()
                    s.current_task = 0 if s.stream else -1
                await send({"type": "reset_ok", "task_idx": s.current_task})
                await send_status("Model reset. Stream preserved.")

            elif op == "reset_all":
                async with s.lock:
                    s.model = _new_model(s)
                    s.cl    = _new_cl(s) if s.use_cl else None
                    s.stream = []
                    s.task_data = []
                    s.trained.clear()
                    s.oracle_phis.clear()
                    s.phi_history.clear()
                    s.current_task = -1
                await send({"type": "reset_ok", "task_idx": -1})

            else:
                await send_status(f"Unknown op: {op!r}", "warn")

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        import traceback; traceback.print_exc()
        try:
            await ws.send_json({"type": "error", "msg": str(exc)})
        except Exception:
            pass
    finally:
        _stop.set()
        s.train_cancel.set()   # unblock any paused training thread
        s.train_pause.clear()
        reader_task.cancel()


def main():
    ap = argparse.ArgumentParser(description="Schemata WebSocket server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"schemata server  →  http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
