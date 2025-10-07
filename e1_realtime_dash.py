#!/usr/bin/env python3
# Elephant Brain Labs — E1 Real-Time EEG (First Channel Only)
# - Resilient connect (default COM13), status bar, Connect/Disconnect buttons
# - Decodes E1 24-bit frames; MATLAB-faithful amplitude conversion
#   EEG mV = raw * 0.000286 ; plotted as µV (= mV * 1000)
# - Displays ONLY the first EEG channel (index 0) with no custom name
# - Band power + PSD derived from the SAME displayed signal
# - Fixes time/length mismatch => waveform always renders
#
# pip install pyserial numpy dash plotly scipy

from __future__ import annotations
import os, io, sys, base64, threading, time, queue
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Generator

import numpy as np
import serial
from serial.tools import list_ports

import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input, State, callback_context
from scipy import signal
from scipy.integrate import trapezoid
from collections import deque

# ─── Compatibility for PyInstaller consoles (optional) ──────────────────────
if sys.stdout is None: sys.stdout = io.StringIO()
if sys.stderr is None: sys.stderr = io.StringIO()

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

# ─── SETTINGS ───────────────────────────────────────────────────────────────
PORT_HTTP          = 4548              # Dash UI port
DEFAULT_COM_PORT   = "COM13"           # Matches your MATLAB SerialCOM
SAMPLE_RATE_HZ     = 250               # E1 EEG rate
REFRESH_S          = 0.2               # chunking cadence like MATLAB
PLOT_DURATION_SEC  = 2.0               # seconds visible in EEG trace
ROLLING_WINDOW_SEC = 2.0               # window used for FFT/PSD/bands
NOTCH_HZ           = 50                # 50 or 60
HP_CUTOFF_HZ       = 5
LP_CUTOFF_HZ       = 45

# Bands for bar chart
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta":  (15, 18),
    "Gamma": (30, 40),
}

# ─── E1 low-level config & stream (MATLAB-accurate protocol) ────────────────
@dataclass
class E1Config:
    port: str = DEFAULT_COM_PORT
    baudrate: int = 460_800
    timeout_s: float = 1.0        # short for snappy retries
    mode: int = 0                 # 0=EEG, 2=Impedance, 3=Test
    samp_freq: int = SAMPLE_RATE_HZ
    num_ch_eeg: int = 32
    num_ch_acc: int = 6
    refresh_s: float = REFRESH_S

    # MATLAB conversion factors
    conv_eeg_mv: float = 0.000286                 # mV/LSB
    conv_eda_mv: float = 3.3/4096*1000            # mV/LSB
    conv_temp_c: float = 0.0078125                # °C/LSB

    # Aux indices (0-based)
    idx_ppg: Tuple[int, int] = (32, 33)
    idx_eda: int = 34
    idx_temp: int = 35
    idx_ctrl2: int = 37

class E1Stream:
    def __init__(self, cfg: E1Config):
        self.cfg = cfg
        self.ser: Optional[serial.Serial] = None
        self.num_ch_tot = cfg.num_ch_eeg + cfg.num_ch_acc

    def _send_go(self):
        # GO bytes = [170, mode*2 + 1, 85]
        self.ser.write(bytes([170, self.cfg.mode * 2 + 1, 85]))

    def _send_stop(self):
        # STOP bytes = [170, 0, 85]
        try: self.ser.write(bytes([170, 0, 85]))
        except Exception: pass

    def open(self) -> bool:
        try:
            self.ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=self.cfg.timeout_s)
            time.sleep(0.05)
            self._send_go()
            return True
        except Exception:
            self.ser = None
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self._send_stop()
            time.sleep(0.05)
            try: self.ser.close()
            except Exception: pass
        self.ser = None

    def __enter__(self):
        self.open();
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _bytes24_to_int32_signed(msb: np.ndarray, mid: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        val = (msb.astype(np.int32) << 16) | (mid.astype(np.int32) << 8) | lsb.astype(np.int32)
        neg = val >= (1 << 23)
        val[neg] -= (1 << 24)
        return val

    def _read_exact(self, nbytes: int) -> Optional[bytes]:
        buf = bytearray()
        # allow timeouts; return partial/None for robustness
        chunk = self.ser.read(nbytes)
        if not chunk:
            return None
        buf.extend(chunk)
        # if partial, let the caller skip this cycle
        return bytes(buf) if len(buf) == nbytes else None

    def stream_chunks(self) -> Generator[Tuple[np.ndarray, Dict[str, np.ndarray]], None, None]:
        cfg = self.cfg
        ch_tot = self.num_ch_tot
        ns = int(round(cfg.samp_freq * cfg.refresh_s))
        refresh_real = ns / float(cfg.samp_freq)

        rows = np.arange(ch_tot * 3).reshape(ch_tot, 3)
        t_local = np.linspace(0.0, refresh_real, ns + 1, dtype=np.float32)
        raw_block = np.zeros((ch_tot, ns + 1), dtype=np.int32)

        while True:
            need = ch_tot * 3 * ns
            buf = self._read_exact(need)
            if buf is None:
                # let caller handle retry/yield
                yield (t_local, {"RAW": None, "EEG_mv": None})
                continue

            temp = np.frombuffer(buf, dtype=np.uint8).reshape(ch_tot * 3, ns, order="F")
            msb = temp[rows[:, 0], :]
            mid = temp[rows[:, 1], :]
            lsb = temp[rows[:, 2], :]
            vals = self._bytes24_to_int32_signed(msb, mid, lsb)

            raw_block[:, 1:] = vals
            raw_block[:, 0]  = raw_block[:, 1]

            eeg_mv = raw_block[:cfg.num_ch_eeg, :].astype(np.float32) * cfg.conv_eeg_mv
            out = {"RAW": raw_block.copy(), "EEG_mv": eeg_mv}
            yield (t_local, out)

# ─── State for UI & processing ──────────────────────────────────────────────
ser_lock        = threading.Lock()
stop_event      = threading.Event()

# Connection status
desired_port    = DEFAULT_COM_PORT
last_status     = "Disconnected"
last_error      = ""
mode_current    = 0   # 0=EEG, 2=Impedance

# Shared stream object (re-created on port change)
stream: Optional[E1Stream] = None

# Data buffers for FIRST channel only (index 0)
plot_data   = []  # list[(timestamp, microvolts)]
rolling_buf = deque(maxlen=int(SAMPLE_RATE_HZ * ROLLING_WINDOW_SEC))

# Filters (used for PSD/bandpower; guarded so they won’t run too early)
nyq = 0.5 * SAMPLE_RATE_HZ
b_hp, a_hp       = signal.butter(4,  HP_CUTOFF_HZ/nyq, btype="high")
b_lp, a_lp       = signal.butter(8,  LP_CUTOFF_HZ/nyq, btype="low")
b_notch, a_notch = signal.iirnotch(NOTCH_HZ/nyq, 30)

def min_padlen() -> int:
    # scipy filtfilt padlen rule of thumb; we use the strictest among the chains
    return int(3 * (max(len(a_lp), len(b_lp)) - 1))

def bandpower(sig, fs, band):
    f, psd = signal.welch(sig, fs, nperseg=min(1024, len(sig)))
    idx = (f >= band[0]) & (f <= band[1])
    if not np.any(idx):
        return 0.0
    return trapezoid(psd[idx], f[idx])

# ─── Connection + streaming workers ─────────────────────────────────────────
def connection_worker():
    global stream, last_status, last_error
    while not stop_event.is_set():
        ok = False
        try:
            with ser_lock:
                if stream is None:
                    stream = E1Stream(E1Config(port=desired_port, mode=mode_current))
                # try open if needed
                if stream.ser is None or not stream.ser.is_open:
                    last_status = f"Connecting to {desired_port}…"
                    last_error  = ""
                    ok = stream.open()
                    if ok:
                        last_status = f"Connected ({desired_port})"
                else:
                    ok = True
        except Exception as e:
            last_error = str(e)
            last_status = f"Connecting to {desired_port}…"
            ok = False

        # slow down retries a bit
        time.sleep(0.5 if ok else 0.7)

def reader_worker():
    global plot_data, last_error, last_status
    while not stop_event.is_set():
        with ser_lock:
            s = stream
            ser_ok = (s is not None and s.ser is not None and s.ser.is_open)
        if not ser_ok:
            time.sleep(0.05);
            continue

        try:
            # Pull one chunk
            t_local, blk = next(s.stream_chunks())
            if blk["EEG_mv"] is None:
                # read timeout/partial; skip
                continue

            # FIRST EEG channel (index 0), convert mV → µV
            ch0_uv = blk["EEG_mv"][0, :] * 1000.0  # µV

            # Create aligned timestamps spanning this chunk
            now = time.time()
            dur = float(t_local[-1] - t_local[0])  # chunk duration
            n   = ch0_uv.shape[0]
            if n <= 0:
                continue

            dt = dur / max(n-1, 1)
            base = now - dur
            for i in range(n):
                plot_data.append((base + i*dt, float(ch0_uv[i])))

            # Keep last PLOT_DURATION_SEC seconds
            t_last = plot_data[-1][0]
            cutoff = t_last - PLOT_DURATION_SEC
            i0 = 0
            while i0 < len(plot_data) and plot_data[i0][0] < cutoff:
                i0 += 1
            if i0 > 0:
                del plot_data[:i0]

        except StopIteration:
            # stream ended; force reconnect
            with ser_lock:
                if s and s.ser:
                    s.close()
                stream = None
            last_status = "Disconnected"
            time.sleep(0.2)
        except Exception as e:
            last_error = str(e)
            time.sleep(0.05)

# Start workers
threading.Thread(target=connection_worker, daemon=True).start()
threading.Thread(target=reader_worker, daemon=True).start()

# ─── Dash App ───────────────────────────────────────────────────────────────
# Optional logo
try:
    with open(resource_path("logo.png"), "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    encoded_logo = ""

app = Dash(__name__)
app.title = "Elephant Brain Labs — E1 Real-Time EEG (Ch 1)"

badge_style = lambda color: {'padding':'6px','color':'white','backgroundColor':color,'marginBottom':'6px'}

def list_serial_options():
    opts = [{"label": p.device, "value": p.device} for p in list_ports.comports()]
    if not any(o["value"] == DEFAULT_COM_PORT for o in opts):
        opts = [{"label": DEFAULT_COM_PORT+" (default)", "value": DEFAULT_COM_PORT}] + opts
    return opts

app.layout = html.Div([
    html.Div([
        html.Img(src=f"data:image/png;base64,{encoded_logo}", style={'height':'56px','marginRight':'12px'}) if encoded_logo else html.Div(),
        html.Div([
            html.H2("Elephant Brain Labs", style={'color':'#1E90FF','margin':'0'}),
            html.H2("E1 Real-Time EEG (First Channel)", style={'margin':'0','fontWeight':'bold'})
        ])
    ], style={'display':'flex','alignItems':'center','gap':'10px','marginBottom':'10px'}),

    html.Div([
        dcc.Dropdown(id="port-dd", clearable=False, style={'width':'220px','display':'inline-block'}),
        html.Button("Refresh Ports", id="btn-refresh", n_clicks=0, style={'marginLeft':'8px'}),
        html.Button("Connect", id="btn-connect", n_clicks=0, style={'marginLeft':'12px'}),
        html.Button("Disconnect", id="btn-disconnect", n_clicks=0, style={'marginLeft':'8px'}),
        html.Span(id="status-text", style={'marginLeft':'16px','fontWeight':'bold'})
    ], style={'marginBottom':'8px'}),

    dcc.Store(id="store-port", data=DEFAULT_COM_PORT),

    # Status + graphs
    html.Div([
        html.Div("Amplifier: Disconnected", id="amp-status",  style=badge_style('gray')),
        html.Div("Data Sync: Waiting",      id="sync-status", style=badge_style('gray')),
    ]),

    dcc.Graph(id='band-power-graph'),
    dcc.Graph(id='psd-graph'),
    dcc.Graph(id='live-eeg-graph'),

    # timers
    dcc.Interval(id='interval-eeg',    interval=100,  n_intervals=0),   # waveform refresh
    dcc.Interval(id='interval-status', interval=800,  n_intervals=0),   # status/bands/psd
])

# Populate ports on load + refresh
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

# Buttons: connect/disconnect
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
    if "btn-disconnect" in trig:
        with ser_lock:
            if stream is not None:
                stream.close()
                stream = None
        last_status = "Disconnected"
    else:
        # force re-open on desired_port
        with ser_lock:
            if stream is not None:
                stream.close()
                stream = None
    return desired_port

# Status + bands + PSD
@app.callback(
    Output('band-power-graph', 'figure'),
    Output('psd-graph', 'figure'),
    Output('amp-status', 'children'),
    Output('amp-status', 'style'),
    Output('sync-status', 'children'),
    Output('sync-status', 'style'),
    Input('interval-status', 'n_intervals')
)
def update_status(_):
    # Connection badges
    with ser_lock:
        connected = (stream is not None and stream.ser is not None and stream.ser.is_open)
    amp_text = f"Amplifier: {'Connected' if connected else 'Connecting…' if 'Connecting' in last_status else 'Disconnected'}"
    amp_style = badge_style('green' if connected else 'orange' if 'Connecting' in last_status else 'gray')

    # Build rolling signal for analysis (same series as the live plot)
    if len(plot_data) < 15:
        empty = go.Figure()
        return (empty, empty,
                amp_text, amp_style,
                "Data Sync: Waiting", badge_style('orange'))

    xs = np.array([v for _, v in plot_data], dtype=np.float32)
    rolling_buf.extend(xs)
    x = np.asarray(rolling_buf, dtype=np.float32)

    # Guard for filtfilt padlen
    y = x
    try:
        if x.size > min_padlen():
            y = signal.filtfilt(b_hp, a_hp, x)
            y = signal.filtfilt(b_lp, a_lp, y)
            y = signal.filtfilt(b_notch, a_notch, y)
    except Exception:
        y = x

    win = y[-int(SAMPLE_RATE_HZ * ROLLING_WINDOW_SEC):]
    if win.size < 8:
        empty = go.Figure()
        return (empty, empty,
                amp_text, amp_style,
                "Data Sync: Buffering…", badge_style('orange'))

    band_vals = {name: bandpower(win, SAMPLE_RATE_HZ, rng) for name, rng in BANDS.items()}
    fig_band = go.Figure(data=go.Bar(x=list(band_vals.keys()), y=list(band_vals.values()), name="EEG"))
    fig_band.update_layout(title="Band Power (from displayed channel)", yaxis_title="µV²", height=300)

    freqs = np.fft.rfftfreq(len(win), d=1.0/SAMPLE_RATE_HZ)
    psd = (np.abs(np.fft.rfft(win)) ** 2) / max(len(win), 1)
    mask = freqs <= 40
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=freqs[mask], y=psd[mask], mode='lines', name="EEG"))
    fig_psd.update_layout(title="PSD", xaxis_title="Hz", yaxis_title="µV²/Hz", height=400)

    return (fig_band, fig_psd,
            amp_text, amp_style,
            "Data Sync: Streaming", badge_style('green'))

# Live EEG trace (FIRST channel only)
@app.callback(Output('live-eeg-graph', 'figure'), Input('interval-eeg', 'n_intervals'))
def update_live_eeg(_):
    if not plot_data:
        return go.Figure()

    # Use exactly the timestamps and values we buffered
    ts = np.array([t for t, _ in plot_data], dtype=np.float64)
    xs = np.array([v for _, v in plot_data], dtype=np.float32)  # µV

    # Align time axis to [0, PLOT_DURATION_SEC]
    t_rel = ts - ts[0]
    t_rel = t_rel - t_rel[-1] + PLOT_DURATION_SEC

    # No channel name, minimal legend (hidden)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t_rel, y=xs, mode='lines',
        line=dict(width=1), name="EEG"
    ))
    fig.update_layout(
        height=300, showlegend=False,
        title="Live EEG (First Channel)",
        xaxis_title="Time (s)", yaxis_title="µV"
    )
    return fig

if __name__ == "__main__":
    app.run(debug=False, port=PORT_HTTP, use_reloader=False)
