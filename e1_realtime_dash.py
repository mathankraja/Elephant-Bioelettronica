# e1_realtime_dash.py
# ─── Real-time EEG Visualizer for OT Bioelettronica E1 ─────────────────────
# - Opens COM, sends GO, decodes 24-bit samples, streams to Dash graphs
# - Replicates your eego-based UI (AFz-focused) for the E1
#
# pip install pyserial numpy plotly dash scipy

from __future__ import annotations
import os, io, sys, base64, threading, time, queue
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Generator

import numpy as np
import serial

import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input
from scipy import signal
from scipy.integrate import trapezoid
from collections import deque
import webbrowser
from threading import Timer

# ─── Compatibility for PyInstaller consoles (optional) ──────────────────────
if sys.stdout is None: sys.stdout = io.StringIO()
if sys.stderr is None: sys.stderr = io.StringIO()

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

# ─── SETTINGS (edit these) ──────────────────────────────────────────────────
PORT_HTTP          = 4548          # Dash UI port
COM_PORT           = "COM13"       # ← set your E1 COM port
CHANNEL_NAME       = "AFz"         # label only
CHANNEL_INDEX      = 2             # zero-based EEG index (0..31)
SAMPLE_RATE_HZ     = 250           # E1 EEG rate (matches your MATLAB cfg)
PLOT_DURATION_SEC  = 2
ROLLING_WINDOW_SEC = 2
NOTCH_HZ           = 50            # choose 50 or 60 depending on mains
HP_CUTOFF_HZ       = 5
LP_CUTOFF_HZ       = 45

# Bands for bar chart & NF ratio
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
    port: str = COM_PORT
    baudrate: int = 460_800
    timeout_s: float = 8.0
    mode: int = 0              # 0=EEG, 2=Impedance, 3=Test
    samp_freq: int = SAMPLE_RATE_HZ
    num_ch_eeg: int = 32
    num_ch_acc: int = 6
    refresh_s: float = 0.2     # chunking cadence like MATLAB

    # MATLAB conversion factors
    conv_eeg_mv: float = 0.000286
    conv_eda_mv: float = 3.3/4096*1000
    conv_temp_c: float = 0.0078125

    # Aux zero-based indices (kept here for completeness)
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
        self.ser.write(bytes([170, self.cfg.mode * 2 + 1, 85]))

    def _send_stop(self):
        try: self.ser.write(bytes([170, 0, 85]))
        except Exception: pass

    def open(self):
        self.ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=self.cfg.timeout_s)
        time.sleep(0.05)
        self._send_go()

    def close(self):
        if self.ser and self.ser.is_open:
            self._send_stop()
            time.sleep(0.05)
            self.ser.close()
        self.ser = None

    def __enter__(self): self.open(); return self
    def __exit__(self, exc_type, exc, tb): self.close()

    @staticmethod
    def _bytes24_to_int32_signed(msb: np.ndarray, mid: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        val = (msb.astype(np.int32) << 16) | (mid.astype(np.int32) << 8) | lsb.astype(np.int32)
        neg = val >= (1 << 23)
        val[neg] -= (1 << 24)
        return val

    def _read_exact(self, nbytes: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < nbytes:
            chunk = self.ser.read(nbytes - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

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
                break
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

# ─── Stream → queues for UI ─────────────────────────────────────────────────
data_queue = queue.Queue()
stop_event  = threading.Event()
plot_data   = []            # [(timestamp, microvolts)]
baseline_data = []
baseline_collecting = False
baseline_theta_beta = None

max_points = SAMPLE_RATE_HZ * PLOT_DURATION_SEC
rolling_buffer = deque(maxlen=SAMPLE_RATE_HZ * ROLLING_WINDOW_SEC)

# Filters
nyq = 0.5 * SAMPLE_RATE_HZ
b_hp, a_hp     = signal.butter(4,  HP_CUTOFF_HZ/nyq, btype="high")
b_lp, a_lp     = signal.butter(8,  LP_CUTOFF_HZ/nyq, btype="low")
b_notch, a_notch = signal.iirnotch(NOTCH_HZ/nyq, 30)

def bandpower(sig, fs, band):
    f, psd = signal.welch(sig, fs, nperseg=min(1024, len(sig)))
    idx = (f >= band[0]) & (f <= band[1])
    if not np.any(idx): return 0.0
    return trapezoid(psd[idx], f[idx])

def e1_worker():
    """
    Background thread: pulls chunks from E1, pushes single-channel samples
    (in microvolts) into data_queue for the UI.
    """
    cfg = E1Config(port=COM_PORT, samp_freq=SAMPLE_RATE_HZ, mode=0, refresh_s=0.2)
    try:
        with E1Stream(cfg) as s:
            print(f"✅ E1 connected on {cfg.port} at {cfg.baudrate} baud; streaming EEG …")
            for t, blk in s.stream_chunks():
                if stop_event.is_set(): break
                eeg_mv = blk["EEG_mv"]  # (32, n+1) in mV
                ch = eeg_mv[CHANNEL_INDEX, :] * 1000.0  # → microvolts
                now = time.time()
                # enqueue each sample with a timestamp close to 'now'
                # distribute timestamps across the chunk duration for smooth x-axis
                n = ch.shape[0]
                dt = (t[-1] - t[0]) / max(n-1, 1)
                base = now - (t[-1] - t[0])
                for i in range(n):
                    data_queue.put((base + i*dt, float(ch[i])))
    except Exception as e:
        print(f"❌ E1 worker error: {e}")

# ─── Dash App ───────────────────────────────────────────────────────────────
# Optional logo
try:
    with open(resource_path("logo.png"), "rb") as f:
        encoded_logo = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    encoded_logo = ""

app = Dash(__name__)
app.title = "E1 Real-Time EEG"

badge_style = lambda color: {'padding':'6px','color':'white','backgroundColor':color,'marginBottom':'6px'}

app.layout = html.Div([
    html.Div([
        html.Img(src=f"data:image/png;base64,{encoded_logo}", style={'height':'56px','marginRight':'12px'}) if encoded_logo else html.Div(),
        html.Div([
            html.H2("Elephant Brain Labs", style={'color':'#1E90FF','margin':'0'}),
            html.H2(f"Real-Time EEG ({CHANNEL_NAME} channel)", style={'margin':'0','fontWeight':'bold'})
        ])
    ], style={'display':'flex','alignItems':'center','gap':'10px','marginBottom':'10px'}),

    html.Button("Toggle Baseline Capture", id="btn-baseline", n_clicks=0, style={'marginBottom':'8px'}),

    dcc.Graph(id='neurofeedback-graph'),
    html.Div([
        html.Div("Amplifier: Disconnected", id="amp-status",  style=badge_style('gray')),
        html.Div("Data Sync: Waiting",      id="sync-status", style=badge_style('gray')),
    ]),

    dcc.Graph(id='band-power-graph'),
    dcc.Graph(id='psd-graph'),
    dcc.Graph(id='live-eeg-graph'),

    # timers
    dcc.Interval(id='interval-eeg',    interval=100,  n_intervals=0),  # waveform refresh
    dcc.Interval(id='interval-status', interval=1000, n_intervals=0),  # NF/PSD/Bands & badges
])

# ─── Neurofeedback: Theta/Beta ratio with baseline toggle ───────────────────
@app.callback(
    Output('neurofeedback-graph', 'figure'),
    Input('btn-baseline', 'n_clicks')
)
def update_neurofeedback(clicks):
    global baseline_data, baseline_collecting, baseline_theta_beta
    if len(plot_data) < 15:
        return go.Figure()

    samples = np.array([v for _, v in plot_data], dtype=np.float32)
    rolling_buffer.extend(samples)
    x = np.asarray(rolling_buffer, dtype=np.float32)

    if len(x) < 20:
        return go.Figure()

    # filter chain
    y = signal.filtfilt(b_hp, a_hp, x)
    y = signal.filtfilt(b_lp, a_lp, y)
    y = signal.filtfilt(b_notch, a_notch, y)
    win = y[-SAMPLE_RATE_HZ * PLOT_DURATION_SEC:]

    theta = bandpower(win, SAMPLE_RATE_HZ, (4, 8))
    beta  = bandpower(win, SAMPLE_RATE_HZ, (15, 18))
    ratio = (theta / beta) if beta > 1e-12 else 0.0

    # baseline toggle (odd clicks = collecting)
    if clicks % 2 == 1:
        baseline_collecting = True
        baseline_data.append(ratio)
    else:
        if baseline_collecting and len(baseline_data) > 0:
            baseline_theta_beta = float(np.mean(baseline_data))
        baseline_collecting = False
        baseline_data = []  # reset bin

    ref = baseline_theta_beta or 0.0
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=float(ratio),
        delta={'reference': ref, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={'axis': {'range': [0, max(2.0, ratio * 1.2)]}, 'bar': {'color': "royalblue"}},
        title={"text": f"Theta/Beta Ratio ({CHANNEL_NAME})"}
    ))
    fig.update_layout(height=240)
    return fig

# ─── Live EEG trace ─────────────────────────────────────────────────────────
@app.callback(Output('live-eeg-graph', 'figure'), Input('interval-eeg', 'n_intervals'))
def update_live_eeg(_):
    # drain queue
    drained = 0
    while not data_queue.empty():
        ts, val = data_queue.get()
        plot_data.append((ts, val))
        drained += 1
    # keep last N seconds
    if plot_data:
        t_last = plot_data[-1][0]
        cutoff = t_last - PLOT_DURATION_SEC
        # drop old
        i0 = 0
        while i0 < len(plot_data) and plot_data[i0][0] < cutoff:
            i0 += 1
        if i0 > 0:
            del plot_data[:i0]

    if len(plot_data) < 15:
        return go.Figure()

    ts = np.array([t for t, _ in plot_data], dtype=np.float64)
    xs = np.array([v for _, v in plot_data], dtype=np.float32)

    # filter chain on a rolling copy for stability
    rolling_buffer.extend(xs)
    x = np.asarray(rolling_buffer, dtype=np.float32)
    y = signal.filtfilt(b_hp, a_hp, x)
    y = signal.filtfilt(b_lp, a_lp, y)
    y = signal.filtfilt(b_notch, a_notch, y)

    # align to last len(xs) points
    display = y[-len(xs):]
    t_rel = ts - ts[-1] + PLOT_DURATION_SEC

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t_rel, y=display, mode='lines',
        line=dict(width=1), name=CHANNEL_NAME
    ))
    fig.update_layout(
        height=300, showlegend=True,
        title=f"Live EEG – {CHANNEL_NAME}",
        xaxis_title="Time (s)", yaxis_title="µV"
    )
    return fig

# ─── Band powers + PSD + status badges ──────────────────────────────────────
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
    if len(plot_data) < 15:
        empty = go.Figure()
        return (empty, empty,
                "Amplifier: Connecting…", badge_style('orange'),
                "Data Sync: Waiting",      badge_style('orange'))

    xs = np.array([v for _, v in plot_data], dtype=np.float32)
    rolling_buffer.extend(xs)
    x = np.asarray(rolling_buffer, dtype=np.float32)
    y = signal.filtfilt(b_hp, a_hp, x)
    y = signal.filtfilt(b_lp, a_lp, y)
    y = signal.filtfilt(b_notch, a_notch, y)
    win = y[-SAMPLE_RATE_HZ * PLOT_DURATION_SEC:]

    band_vals = {name: bandpower(win, SAMPLE_RATE_HZ, rng) for name, rng in BANDS.items()}
    fig_band = go.Figure(data=go.Bar(x=list(band_vals.keys()), y=list(band_vals.values()), name=CHANNEL_NAME))
    fig_band.update_layout(title=f"Band Power – {CHANNEL_NAME}", yaxis_title="µV²", height=300)

    # simple FFT PSD
    freqs = np.fft.rfftfreq(len(win), d=1.0/SAMPLE_RATE_HZ)
    psd = (np.abs(np.fft.rfft(win)) ** 2) / max(len(win), 1)
    mask = freqs <= 40
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=freqs[mask], y=psd[mask], mode='lines', name=CHANNEL_NAME))
    fig_psd.update_layout(title="PSD", xaxis_title="Hz", yaxis_title="µV²/Hz", height=400)

    return (fig_band, fig_psd,
            "Amplifier: Connected", badge_style('green'),
            "Data Sync: Streaming", badge_style('green'))

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # start E1 background reader
    stop_event.clear()
    threading.Thread(target=e1_worker, daemon=True).start()

    # open browser
    Timer(1, lambda: webbrowser.open_new(f"http://127.0.0.1:{PORT_HTTP}")).start()
    app.run(debug=True, port=PORT_HTTP, use_reloader=False)
