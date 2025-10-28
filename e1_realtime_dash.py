#!/usr/bin/env python3
# Elephant Brain Labs — E1 Real-Time EEG (Pick Channel)
# - COM28 default (use with simulator on COM29)
# - EXACT MATLAB conversion: mV = raw * 0.000286 ; plotted as µV
# - Channel dropdown (1–32) drives Live EEG, Band Power, and PSD
# - Robust connect/disconnect + status line

# Author: @Mathan K. Raja  Elephant Brain Labs • October 2025

from __future__ import annotations
import os, io, sys, threading, time, base64
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Generator, List
from collections import deque

import numpy as np
import serial
from serial.tools import list_ports

import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input, State, callback_context
from scipy import signal
from scipy.integrate import trapezoid

# ---- console safety (optional) ----
if sys.stdout is None: sys.stdout = io.StringIO()
if sys.stderr is None: sys.stderr = io.StringIO()

# ---- Settings ----
PORT_HTTP          = 4548
DEFAULT_COM_PORT   = "COM28"       # viewer side of COM28 <-> COM29
SAMPLE_RATE_HZ     = 250
REFRESH_S          = 0.2
PLOT_DURATION_SEC  = 2.0
ROLLING_WINDOW_SEC = 2.0
NOTCH_HZ           = 50
HP_CUTOFF_HZ       = 5
LP_CUTOFF_HZ       = 45

NUM_CH_EEG = 32
NUM_CH_ACC = 6
NUM_CH_TOT = NUM_CH_EEG + NUM_CH_ACC

BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta":  (15, 18),
    "Gamma": (30, 40),
}

# ---- E1 configuration & stream ----
@dataclass
class E1Config:
    port: str = DEFAULT_COM_PORT
    baudrate: int = 460_800
    timeout_s: float = 0.15
    mode: int = 0                 # 0=EEG, 2=Impedance, 3=Test
    samp_freq: int = SAMPLE_RATE_HZ
    num_ch_eeg: int = NUM_CH_EEG
    num_ch_acc: int = NUM_CH_ACC
    refresh_s: float = REFRESH_S

    # MATLAB-faithful conversions
    conv_eeg_mv: float = 0.000286
    conv_eda_mv: float = 3.3/4096*1000
    conv_temp_c: float = 0.0078125

    idx_ppg: Tuple[int, int] = (32, 33)
    idx_eda: int = 34
    idx_temp: int = 35

class E1Stream:
    def __init__(self, cfg: E1Config):
        self.cfg = cfg
        self.ser: Optional[serial.Serial] = None
        self.num_ch_tot = cfg.num_ch_eeg + cfg.num_ch_acc

    def _go(self):   self.ser.write(bytes([170, self.cfg.mode*2 + 1, 85]))
    def _stop(self):
        try: self.ser.write(bytes([170, 0, 85]))
        except Exception: pass

    def open(self) -> bool:
        try:
            self.ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=self.cfg.timeout_s)
            time.sleep(0.05); self._go(); return True
        except Exception:
            self.ser = None; return False

    def close(self):
        if self.ser and self.ser.is_open:
            self._stop(); time.sleep(0.05)
            try: self.ser.close()
            except Exception: pass
        self.ser = None

    def __enter__(self): self.open(); return self
    def __exit__(self, *_): self.close()

    @staticmethod
    def _bytes24_to_int32_signed(msb: np.ndarray, mid: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        val = (msb.astype(np.int32) << 16) | (mid.astype(np.int32) << 8) | lsb.astype(np.int32)
        neg = val >= (1 << 23)
        val[neg] -= (1 << 24)
        return val

    def _read_exact_or_none(self, nbytes: int) -> Optional[bytes]:
        buf = self.ser.read(nbytes)
        return buf if (buf and len(buf) == nbytes) else None

    def stream_chunks(self) -> Generator[Tuple[np.ndarray, Dict[str, np.ndarray]], None, None]:
        cfg = self.cfg
        ch_tot = self.num_ch_tot
        ns = int(round(cfg.samp_freq * cfg.refresh_s))
        refresh_real = ns / float(cfg.samp_freq)
        rows = np.arange(ch_tot * 3).reshape(ch_tot, 3)
        t_local = np.linspace(0.0, refresh_real, ns + 1, dtype=np.float32)
        raw_block = np.zeros((ch_tot, ns + 1), dtype=np.int32)

        need = ch_tot * 3 * ns
        while True:
            buf = self._read_exact_or_none(need)
            if buf is None:
                yield (t_local, {"RAW": None, "EEG_mv": None}); continue
            temp = np.frombuffer(buf, dtype=np.uint8).reshape(ch_tot * 3, ns, order="F")
            msb, mid, lsb = temp[rows[:,0],:], temp[rows[:,1],:], temp[rows[:,2],:]
            vals = self._bytes24_to_int32_signed(msb, mid, lsb)

            raw_block[:,1:] = vals
            raw_block[:,0]  = raw_block[:,1]
            eeg_mv = raw_block[:cfg.num_ch_eeg,:].astype(np.float32) * cfg.conv_eeg_mv
            yield (t_local, {"RAW": raw_block.copy(), "EEG_mv": eeg_mv})

# ---- State / workers ----
ser_lock     = threading.Lock()
stop_event   = threading.Event()

desired_port = DEFAULT_COM_PORT
last_status  = "Disconnected"
last_error   = ""
mode_current = 0

stream: Optional[E1Stream] = None

# Rolling buffers per-channel (µV) + timestamps
ROLL_MAX = int((PLOT_DURATION_SEC + 0.5) * SAMPLE_RATE_HZ)
ts_buf: deque = deque(maxlen=ROLL_MAX)
ch_buf: List[deque] = [deque(maxlen=ROLL_MAX) for _ in range(NUM_CH_EEG)]

# Analysis filters
nyq = 0.5 * SAMPLE_RATE_HZ
b_hp, a_hp       = signal.butter(4,  HP_CUTOFF_HZ/nyq, btype="high")
b_lp, a_lp       = signal.butter(8,  LP_CUTOFF_HZ/nyq, btype="low")
b_notch, a_notch = signal.iirnotch(NOTCH_HZ/nyq, 30)

def min_padlen() -> int:
    return int(3 * (max(len(a_lp), len(b_lp)) - 1))

def bandpower(sig, fs, band):
    f, psd = signal.welch(sig, fs, nperseg=min(1024, len(sig)))
    sel = (f >= band[0]) & (f <= band[1])
    return float(trapezoid(psd[sel], f[sel])) if np.any(sel) else 0.0

def connection_worker():
    global stream, last_status, last_error
    while not stop_event.is_set():
        ok = False
        try:
            with ser_lock:
                if stream is None:
                    stream = E1Stream(E1Config(port=desired_port, mode=mode_current))
                if stream.ser is None or not stream.ser.is_open:
                    last_status = f"Connecting to {desired_port}…"; last_error = ""
                    ok = stream.open()
                    if ok: last_status = f"Connected ({desired_port})"
                else:
                    ok = True
        except Exception as e:
            last_error = str(e); last_status = f"Connecting to {desired_port}…"; ok = False
        time.sleep(0.5 if ok else 0.7)

def reader_worker():
    global stream
    while not stop_event.is_set():
        with ser_lock:
            s = stream
            ser_ok = (s is not None and s.ser is not None and s.ser.is_open)
        if not ser_ok:
            time.sleep(0.05); continue
        try:
            t_local, blk = next(s.stream_chunks())
            if blk["EEG_mv"] is None:
                continue
            eeg_uv = blk["EEG_mv"][:NUM_CH_EEG,:] * 1000.0  # (32, ns+1) in µV
            now = time.time()
            dur = float(t_local[-1] - t_local[0])
            n   = eeg_uv.shape[1]
            if n <= 0: continue
            dt = dur / max(n-1, 1)
            base = now - dur
            for i in range(n):
                ts_buf.append(base + i*dt)
                col = eeg_uv[:, i]
                for k in range(NUM_CH_EEG):
                    ch_buf[k].append(float(col[k]))
        except StopIteration:
            with ser_lock:
                if s and s.ser: s.close()
                stream = None
            time.sleep(0.2)
        except Exception:
            time.sleep(0.05)

threading.Thread(target=connection_worker, daemon=True).start()
threading.Thread(target=reader_worker, daemon=True).start()

# ---- Dash UI ----
def _encoded_logo():
    for candidate in ("logo.png", "assets/logo.png"):
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return ""

badge_style = lambda color: {'padding':'6px','color':'white','backgroundColor':color,'marginBottom':'6px'}

def list_serial_options():
    opts = [{"label": p.device, "value": p.device} for p in list_ports.comports()]
    if not any(o["value"] == DEFAULT_COM_PORT for o in opts):
        opts = [{"label": DEFAULT_COM_PORT+" (default)", "value": DEFAULT_COM_PORT}] + opts
    return opts

app = Dash(__name__)
app.title = "Elephant Brain Labs — E1 (Pick Channel)"

app.layout = html.Div([
    html.Div([
        (html.Img(src=f"data:image/png;base64,{_encoded_logo()}", style={'height':'56px','marginRight':'12px'})
         if _encoded_logo() else html.Div()),
        html.Div([
            html.H2("Elephant Brain Labs", style={'color':'#1E90FF','margin':'0'}),
            html.H2("E1 Real-Time EEG", style={'margin':'0','fontWeight':'bold'})
        ])
    ], style={'display':'flex','alignItems':'center','gap':'10px','marginBottom':'10px'}),

    html.Div([
        dcc.Dropdown(id="port-dd", clearable=False, style={'width':'220px','display':'inline-block'}),
        html.Button("Refresh Ports", id="btn-refresh", n_clicks=0, style={'marginLeft':'8px'}),
        html.Button("Connect", id="btn-connect", n_clicks=0, style={'marginLeft':'12px'}),
        html.Button("Disconnect", id="btn-disconnect", n_clicks=0, style={'marginLeft':'8px'}),
        html.Span(id="status-text", style={'marginLeft':'16px','fontWeight':'bold'})
    ], style={'marginBottom':'8px'}),

    html.Div([
        html.Label("Channel"),
        dcc.Dropdown(
            id="ch-dd",
            options=[{"label": f"Ch {i}", "value": i} for i in range(1, NUM_CH_EEG+1)],
            value=1, clearable=False, style={'width':'140px'}
        ),
    ], style={'display':'flex','gap':'12px','alignItems':'center','marginBottom':'8px'}),

    dcc.Store(id="store-port", data=DEFAULT_COM_PORT),

    html.Div([
        html.Div("Amplifier: Disconnected", id="amp-status",  style=badge_style('gray')),
        html.Div("Data Sync: Waiting",      id="sync-status", style=badge_style('gray')),
    ]),

    dcc.Graph(id='band-power-graph'),
    dcc.Graph(id='psd-graph'),
    dcc.Graph(id='live-eeg-graph'),

    dcc.Interval(id='interval-eeg',    interval=100,  n_intervals=0),
    dcc.Interval(id='interval-status', interval=800,  n_intervals=0),
])

# ---- Callbacks ----
@app.callback(
    Output("port-dd", "options"),
    Output("port-dd", "value"),
    Input("btn-refresh", "n_clicks"),
    State("store-port", "data")
)
def refresh_ports(_n, current):
    opts = list_serial_options()
    vals = [o["value"] for o in opts]
    value = current if current in vals else (vals[0] if vals else DEFAULT_COM_PORT)
    return opts, value

@app.callback(
    Output("store-port", "data"),
    Input("btn-connect", "n_clicks"),
    Input("btn-disconnect", "n_clicks"),
    State("port-dd", "value"),
    State("store-port", "data"),
    prevent_initial_call=True
)
def on_buttons(n_conn, n_disc, dd_port, cur_port):
    trig = [t["prop_id"] for t in callback_context.triggered][0]
    global desired_port, stream, last_status
    desired_port = dd_port or cur_port or DEFAULT_COM_PORT
    with ser_lock:
        if stream is not None:
            stream.close(); stream = None
    last_status = "Disconnected" if "btn-disconnect" in trig else f"Connecting to {desired_port}…"
    return desired_port

@app.callback(Output("status-text", "children"), Input("interval-status", "n_intervals"))
def status_line(_):
    with ser_lock:
        ok = (stream is not None and stream.ser is not None and stream.ser.is_open)
        port = stream.cfg.port if (stream and stream.ser) else desired_port
    return (f"Connected on {port} • mode=EEG • fs={SAMPLE_RATE_HZ} Hz"
            if ok else (last_status or "Disconnected"))

@app.callback(
    Output('band-power-graph', 'figure'),
    Output('psd-graph', 'figure'),
    Output('amp-status', 'children'),
    Output('amp-status', 'style'),
    Output('sync-status', 'children'),
    Output('sync-status', 'style'),
    Input('interval-status', 'n_intervals'),
    Input('ch-dd', 'value')
)
def update_status(_, ch_idx):
    ch = max(1, min(NUM_CH_EEG, int(ch_idx or 1))) - 1  # 0-based
    with ser_lock:
        connected = (stream is not None and stream.ser is not None and stream.ser.is_open)
    amp_text = f"Amplifier: {'Connected' if connected else 'Connecting…' if 'Connecting' in (last_status or '') else 'Disconnected'}"
    amp_style = badge_style('green' if connected else 'orange' if 'Connecting' in (last_status or '') else 'gray')

    if len(ts_buf) < 15 or len(ch_buf[ch]) < 15:
        empty = go.Figure()
        return (empty, empty, amp_text, amp_style, "Data Sync: Waiting", badge_style('orange'))

    xs = np.asarray(ch_buf[ch], dtype=np.float32)
    # build analysis window
    win_len = int(SAMPLE_RATE_HZ * ROLLING_WINDOW_SEC)
    x = xs[-win_len:]

    # filters (guard filtfilt padlen)
    y = x
    try:
        if x.size > min_padlen():
            y = signal.filtfilt(b_hp, a_hp, x)
            y = signal.filtfilt(b_lp, a_lp, y)
            y = signal.filtfilt(b_notch, a_notch, y)
    except Exception:
        y = x

    if y.size < 8:
        empty = go.Figure()
        return (empty, empty, amp_text, amp_style, "Data Sync: Buffering…", badge_style('orange'))

    band_vals = {name: bandpower(y, SAMPLE_RATE_HZ, rng) for name, rng in BANDS.items()}
    fig_band = go.Figure(data=go.Bar(x=list(band_vals.keys()), y=list(band_vals.values()), name=f"Ch {ch+1}"))
    fig_band.update_layout(title=f"Band Power — Ch {ch+1}", yaxis_title="µV²", height=300)

    freqs = np.fft.rfftfreq(len(y), d=1.0/SAMPLE_RATE_HZ)
    psd = (np.abs(np.fft.rfft(y)) ** 2) / max(len(y), 1)
    mask = freqs <= 40
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=freqs[mask], y=psd[mask], mode='lines', name=f"Ch {ch+1}"))
    fig_psd.update_layout(title="PSD", xaxis_title="Hz", yaxis_title="µV²/Hz", height=400)

    return (fig_band, fig_psd, amp_text, amp_style, "Data Sync: Streaming", badge_style('green'))

@app.callback(Output('live-eeg-graph', 'figure'),
              Input('interval-eeg', 'n_intervals'),
              Input('ch-dd', 'value'))
def update_live_eeg(_, ch_idx):
    ch = max(1, min(NUM_CH_EEG, int(ch_idx or 1))) - 1
    if len(ts_buf) < 2 or len(ch_buf[ch]) < 2:
        return go.Figure()
    ts = np.asarray(ts_buf, dtype=np.float64)
    xs = np.asarray(ch_buf[ch], dtype=np.float32)
    n  = min(len(ts), len(xs))
    ts, xs = ts[-n:], xs[-n:]

    t_rel = ts - ts[0]
    t_rel = t_rel - t_rel[-1] + PLOT_DURATION_SEC

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t_rel, y=xs, mode='lines', line={'width':1}, name=f"Ch {ch+1}"))
    fig.update_layout(height=300, showlegend=False,
                      title=f"Live EEG — Ch {ch+1}",
                      xaxis_title="Time (s)", yaxis_title="µV")
    return fig

if __name__ == "__main__":
    app.run(debug=False, port=PORT_HTTP, use_reloader=False)
