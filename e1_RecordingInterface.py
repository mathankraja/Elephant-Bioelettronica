#!/usr/bin/env python3
# ─── Elephant Brain Labs — E1 Real-Time EEG Recorder (Viewer + EDF+) ─────────
# Adds: filename input, Start/Stop recording (32 EEG channels, µV), Event 1..4 annotations
# Plus: Left header logo + favicon (tab icon)
# Records EEG ONLY to EDF+ (no sensors); live plots still show sensors
# Requires: pip install pyedflib numpy dash plotly pyserial
# Author: @Mathan K. Raja  • October 2025

import argparse, threading, time, atexit, os, datetime, base64
from collections import deque
from typing import Optional, List, Tuple
import numpy as np
import serial
from serial.tools import list_ports
from dash import Dash, html, dcc, Output, Input, State, callback_context
import plotly.graph_objects as go
import pyedflib

# --------------------- CLI ---------------------
p = argparse.ArgumentParser(
    description="Elephant Brain Labs — E1 Real-Time EEG Recorder (Viewer + EDF+)"
)
p.add_argument("--default_port", default="COM13", help="Default serial port (MATLAB used COM13)")
p.add_argument("--fs", type=int, default=250, help="Sample rate (Hz)")
p.add_argument("--refresh", type=float, default=0.2, help="Chunk seconds")
p.add_argument("--offset", type=float, default=1.0, help="EEG vertical spacing in mV")
p.add_argument("--units", choices=["mv","uv"], default="mv", help="EEG display units")
p.add_argument("--http_port", type=int, default=4548, help="Dash port")
args = p.parse_args()

# ------------------ Device layout ------------------
NUM_CH_EEG = 32
NUM_CH_ACC = 6
NUM_CH_TOT = NUM_CH_EEG + NUM_CH_ACC
BAUD = 460_800
TIMEOUT_S = 1.0

# Conversions (exactly as MATLAB)
CONV_EEG_MV  = 0.000286               # raw counts -> mV
CONV_EDA_MV  = 3.3/4096*1000.0        # ADC LSB -> mV
CONV_TEMP_C  = 0.0078125              # raw -> °C

# Aux indices (after 32 EEG channels)
IDX_PPG0 = 32
IDX_PPG1 = 33
IDX_EDA  = 34
IDX_TEMP = 35
IDX_CTRL2= 37

# ------------------ Serial Connection ------------------
ser: Optional[serial.Serial] = None
ser_lock = threading.Lock()
desired_port = args.default_port
auto_reconnect = True
mode_current = 0  # 0=EEG, 2=Impedance
last_status = "Disconnected"
last_error = ""

def _safe_open(port: str) -> bool:
    global ser, last_error, last_status
    try:
        s = serial.Serial(port, BAUD, timeout=TIMEOUT_S)
        time.sleep(0.05)
        s.write(bytes([170, mode_current*2 + 1, 85]))
        with ser_lock:
            ser = s
        last_status = f"Connected ({port})"
        last_error = ""
        return True
    except Exception as e:
        last_status = f"Connecting to {port}…"
        last_error = str(e)
        with ser_lock:
            if ser and ser.is_open:
                try: ser.close()
                except: pass
            ser = None
        return False

def _safe_close():
    global ser, last_status
    with ser_lock:
        if ser and ser.is_open:
            try: ser.write(bytes([170,0,85])); time.sleep(0.05)
            except: pass
            try: ser.close()
            except: pass
        ser = None
    last_status = "Disconnected"

def e1_go(mode: int):
    global mode_current
    mode_current = mode
    with ser_lock:
        if ser and ser.is_open:
            try: ser.write(bytes([170, mode*2 + 1, 85]))
            except: pass

def connection_worker():
    """Auto-reconnect loop."""
    while True:
        if auto_reconnect:
            with ser_lock:
                connected = ser is not None and ser.is_open
            if not connected:
                _safe_open(desired_port)
        time.sleep(0.5)

def read_exact(n: int):
    with ser_lock:
        s = ser
    if not s or not s.is_open:
        return None
    buf = bytearray()
    while len(buf) < n:
        chunk = s.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf) if buf else None

def bytes24_to_int32(msb, mid, lsb):
    val = (msb.astype(np.int32)<<16) | (mid.astype(np.int32)<<8) | lsb.astype(np.int32)
    neg = val >= (1<<23)
    val[neg] -= (1<<24)
    return val

# ------------------ Buffers ------------------
ns = int(round(args.fs * args.refresh))
ROLL_SEC = 3.0
max_cols = int(ROLL_SEC * args.fs) + (ns+1)
eeg_buf = np.zeros((NUM_CH_TOT, 0), dtype=np.int32)
imp_r_hist  = deque(maxlen=max_cols)
imp_eda_hist= deque(maxlen=max_cols)
stop_event = threading.Event()

# --------- Recording state (EEG-only) ---------
recording = False
rec_start_time: float = 0.0
rec_ch_buffers: List[List[float]] = [[] for _ in range(NUM_CH_EEG)]  # EEG in µV
rec_events: List[Tuple[float,str]] = []  # (onset_seconds, "Event N")
last_saved_path: Optional[str] = None

def _reset_recording():
    global rec_ch_buffers, rec_events, rec_start_time
    rec_ch_buffers = [[] for _ in range(NUM_CH_EEG)]  # EEG (µV)
    rec_events = []
    rec_start_time = time.time()

def _timestamped_edf_path(base: str) -> str:
    base = (base or "session").strip()
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(base)
    if ext.lower() not in (".edf", ".edf+"):
        ext = ".edf"
    return f"{root}_{ts}{ext}"

def _write_edf_plus(path: str):
    """
    Save EDF+ with EEG only:
      • 32 EEG channels in µV (from rec_ch_buffers)
      • EDF+ annotations from rec_events
    """
    fs = int(args.fs)

    # ---------- EEG ----------
    n_eeg = NUM_CH_EEG
    if n_eeg <= 0:
        raise RuntimeError("No EEG channels configured")

    min_len = min((len(x) for x in rec_ch_buffers), default=0)
    if min_len == 0:
        raise RuntimeError("No EEG data buffered to save")

    # Build list of 1-D contiguous arrays (one per channel)
    eeg_list = [np.ascontiguousarray(ch[:min_len], dtype=np.float64) for ch in rec_ch_buffers]

    # Robust per-channel physical ranges (avoid saturation/NaNs)
    ch_min = np.array([float(np.min(x)) for x in eeg_list], dtype=float)
    ch_max = np.array([float(np.max(x)) for x in eeg_list], dtype=float)
    span   = np.maximum(ch_max - ch_min, 1.0)
    pad    = 0.05 * span
    phys_mins = ch_min - pad
    phys_maxs = ch_max + pad

    signal_headers = [{
        "label": f"Ch{i+1}",
        "dimension": "uV",
        "sample_frequency": fs,     # correct key for pyedflib
        "physical_min": float(phys_mins[i]),
        "physical_max": float(phys_maxs[i]),
        "digital_min": -32768,
        "digital_max":  32767,
        "transducer": "",
        "prefilter": ""
    } for i in range(n_eeg)]

    # Ensure output dir
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Open EDF+ writer
    f = pyedflib.EdfWriter(path, n_eeg, file_type=pyedflib.FILETYPE_EDFPLUS)

    now_dt = datetime.datetime.now()
    try:
        f.setHeader({
            "technician": "Elephant Brain Labs",
            "recording_additional": "E1 32ch EEG",
            "equipment": "OT Bioelettronica Elephant",
            "patientname": "",
            "patient_additional": "",
            "patientcode": "",
            "admincode": "",
            "sex": "",
            "gender": "",
            "birthdate": "",
            "startdate": now_dt,
        })
    except Exception:
        pass
    try:
        if hasattr(f, "setStartdatetime"): f.setStartdatetime(now_dt)
        if hasattr(f, "setPatientCode"):   f.setPatientCode("")
        if hasattr(f, "setAdmincode"):     f.setAdmincode("")
        if hasattr(f, "setGender"):        f.setGender("")
        if hasattr(f, "setSex"):           f.setSex("")
        if hasattr(f, "setBirthdate"):     f.setBirthdate("")
    except Exception:
        pass

    f.setSignalHeaders(signal_headers)

    # Write samples: list of 1-D arrays (one per channel)
    try:
        f.writeSamples(eeg_list)
    except Exception:
        # Fallback: convert to int16 respecting phys_mins/maxs
        dig = []
        for i in range(n_eeg):
            pmn, pmx = phys_mins[i], phys_maxs[i]
            scale = (32767 - (-32768)) / (pmx - pmn)
            di = np.clip(np.round((eeg_list[i] - pmn) * scale - 32768), -32768, 32767).astype(np.int16)
            dig.append(di)
        if hasattr(f, "writeSamples"):
            f.writeSamples(dig)
        else:
            f.writeDigitalSamples(np.vstack(dig))

    # EDF+ annotations
    for onset, label in rec_events:
        try:
            f.writeAnnotation(float(onset), 0.0, str(label))
        except Exception:
            pass

    f.close()

def stream_worker():
    """Read data continuously if connected."""
    global eeg_buf, recording
    rows = np.arange(NUM_CH_TOT*3).reshape(NUM_CH_TOT,3)
    while not stop_event.is_set():
        with ser_lock:
            s_ok = ser is not None and ser.is_open
        if not s_ok:
            time.sleep(0.1); continue
        need = NUM_CH_TOT * 3 * ns
        raw = read_exact(need)
        if not raw or len(raw) < need:
            time.sleep(0.05); continue
        try:
            blk  = np.frombuffer(raw, dtype=np.uint8).reshape(NUM_CH_TOT*3, ns, order="F")
            msb, mid, lsb = blk[rows[:,0], :], blk[rows[:,1], :], blk[rows[:,2], :]
            vals = bytes24_to_int32(msb, mid, lsb)
        except Exception:
            continue

        out = np.zeros((NUM_CH_TOT, ns+1), dtype=np.int32)
        out[:,1:] = vals
        out[:,0]  = out[:,1]
        eeg_buf = np.hstack([eeg_buf, out]) if eeg_buf.size else out
        if eeg_buf.shape[1] > max_cols:
            eeg_buf = eeg_buf[:, -max_cols:]

        # If recording, append EEG (converted to µV)
        if recording:
            eeg_uv_block = out[:NUM_CH_EEG, :].astype(np.float32) * (CONV_EEG_MV * 1000.0)
            for j in range(eeg_uv_block.shape[1]):
                col = eeg_uv_block[:, j]
                for k in range(NUM_CH_EEG):
                    rec_ch_buffers[k].append(float(col[k]))

        # impedance calc unchanged (for live display)
        if mode_current == 2:
            eda_raw = out[IDX_EDA,:].astype(np.int32)
            InSine  = np.floor(eda_raw/4096).astype(np.float32)
            OutSine = (eda_raw % 4096).astype(np.float32)
            InSine  -= np.mean(InSine)
            OutSine  = np.mean(OutSine) - OutSine
            RawRMS  = float(np.sqrt(np.mean(InSine**2)))
            Rkohm   = (21.6*RawRMS) / max(1e-6, (525.0 - RawRMS))
            eda_mv  = OutSine * CONV_EDA_MV
            for _ in range(out.shape[1]): imp_r_hist.append(Rkohm)
            for v in eda_mv: imp_eda_hist.append(float(v))

threading.Thread(target=connection_worker, daemon=True).start()
threading.Thread(target=stream_worker, daemon=True).start()
atexit.register(lambda: (stop_event.set(), _safe_close()))

# ------------------ DASH UI ------------------
def list_serial_options():
    opts = [{"label": p.device, "value": p.device} for p in list_ports.comports()]
    if not any(o["value"] == args.default_port for o in opts):
        opts = [{"label": args.default_port+" (default)", "value": args.default_port}] + opts
    return opts

# Load header logo (left). Prefer assets/logo.png, else logo.png in CWD.
def _load_logo_b64() -> str:
    for candidate in ["assets/logo.png", "logo.png", "assets/logo.jpg", "logo.jpg"]:
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return ""

# App + favicon
app = Dash(__name__, assets_folder="assets")
app.title = "Elephant Brain Labs — E1 Real-Time EEG Recorder"
if os.path.exists(os.path.join("assets", "favicon.png")):
    app._favicon = "favicon.png"
elif os.path.exists(os.path.join("assets", "favicon.ico")):
    app._favicon = "favicon.ico"

encoded_logo = _load_logo_b64()

app.layout = html.Div([
    # Header with left logo + titles
    html.Div([
        (html.Img(
            src=f"data:image/png;base64,{encoded_logo}",
            style={'height': '56px', 'marginRight': '12px'}
         ) if encoded_logo else html.Div()),
        html.Div([
            html.H2("Elephant Brain Labs", style={'color':'#1E90FF','margin':'0'}),
            html.H3("E1 Real-Time EEG Recorder (32-ch, EDF+)", style={'margin':'0','fontWeight':'bold'})
        ])
    ], style={'display':'flex','alignItems':'center','gap':'10px','marginBottom':'10px'}),

    # Connection controls
    html.Div([
        dcc.Dropdown(id="port-dd", clearable=False, style={'width':'220px','display':'inline-block'}),
        html.Button("Refresh Ports", id="btn-refresh", n_clicks=0, style={'marginLeft':'8px'}),
        html.Button("Connect", id="btn-connect", n_clicks=0, style={'marginLeft':'12px'}),
        html.Button("Disconnect", id="btn-disconnect", n_clicks=0, style={'marginLeft':'8px'}),
        html.Button("EEG Mode", id="btn-eeg", n_clicks=0, style={'marginLeft':'16px'}),
        html.Button("Impedance Mode", id="btn-imp", n_clicks=0, style={'marginLeft':'8px'}),
        html.Span(id="status-text", style={'marginLeft':'16px','fontWeight':'bold'})
    ], style={'marginBottom':'8px'}),

    # Recording controls and events
    html.Div([
        html.Label("File base name"),
        dcc.Input(id="fname", type="text", value="session", style={'width':'220px','marginRight':'8px'}),
        html.Button("Start Recording", id="btn-start", n_clicks=0,
                    style={'marginRight':'6px','background':'#198754','color':'white'}),
        html.Button("Stop Recording",  id="btn-stop",  n_clicks=0,
                    style={'background':'#dc3545','color':'white','marginRight':'12px'}),
        html.Span("Recording: OFF", id="rec-status", style={'fontWeight':'bold','marginRight':'12px'}),
        html.Button("Event 1", id="ev1", n_clicks=0, style={'marginRight':'6px'}),
        html.Button("Event 2", id="ev2", n_clicks=0, style={'marginRight':'6px'}),
        html.Button("Event 3", id="ev3", n_clicks=0, style={'marginRight':'6px'}),
        html.Button("Event 4", id="ev4", n_clicks=0, style={'marginRight':'6px'}),
        html.Span(id="ev-count", style={'marginLeft':'8px'})
    ], style={'marginBottom':'10px'}),

    dcc.Store(id="store-port", data=args.default_port),
    dcc.Store(id="store-auto", data=True),
    dcc.Store(id="store-lastpath", data=""),

    dcc.Interval(id="interval-status", interval=500, n_intervals=0),

    dcc.Graph(id="graph-eeg", style={'height':'460px'}),
    dcc.Graph(id="graph-sensors", style={'height':'320px'}),
    dcc.Graph(id="graph-impedance", style={'height':'300px'}),
])

# ------------------ Callbacks ------------------
@app.callback(
    Output("port-dd","options"),
    Output("port-dd","value"),
    Input("btn-refresh","n_clicks"),
    State("store-port","data")
)
def refresh_ports(_n, current):
    opts = list_serial_options()
    vals = [o["value"] for o in opts]
    value = current if current in vals else (vals[0] if vals else args.default_port)
    return opts, value

@app.callback(
    Output("store-port","data"),
    Output("store-auto","data"),
    Input("btn-connect","n_clicks"),
    Input("btn-disconnect","n_clicks"),
    Input("btn-eeg","n_clicks"),
    Input("btn-imp","n_clicks"),
    State("port-dd","value"),
    State("store-port","data"),
    State("store-auto","data"),
    prevent_initial_call=True
)
def on_buttons(n_conn, n_disc, n_eeg, n_imp, dd_port, cur_port, auto):
    trig = [t["prop_id"] for t in callback_context.triggered][0]
    global desired_port, auto_reconnect
    desired_port = dd_port or cur_port or args.default_port
    if "btn-connect" in trig:
        auto_reconnect = True
        return desired_port, True
    elif "btn-disconnect" in trig:
        auto_reconnect = False
        _safe_close()
        return desired_port, False
    elif "btn-eeg" in trig:
        e1_go(0); auto_reconnect = True
        return desired_port, True
    elif "btn-imp" in trig:
        e1_go(2); auto_reconnect = True
        return desired_port, True
    return desired_port, auto

@app.callback(
    Output("status-text","children"),
    Input("interval-status","n_intervals"),
    State("store-port","data"),
    State("store-auto","data")
)
def status_tick(_n, port_val, auto_val):
    with ser_lock:
        connected = ser is not None and ser.is_open
    txt = last_status
    if not connected and auto_val:
        txt = f"Connecting to {port_val}…  " + (f"(last error: {last_error})" if last_error else "")
    return txt

# Recording controls & events
@app.callback(
    Output("rec-status", "children"),
    Output("ev-count", "children"),
    Output("store-lastpath", "data"),
    Input("btn-start", "n_clicks"),
    Input("btn-stop",  "n_clicks"),
    Input("ev1", "n_clicks"),
    Input("ev2", "n_clicks"),
    Input("ev3", "n_clicks"),
    Input("ev4", "n_clicks"),
    State("fname", "value"),
    State("store-lastpath", "data"),
    prevent_initial_call=True
)
def recording_controls(n_start, n_stop, n1, n2, n3, n4, base_name, last_path):
    global recording, last_saved_path
    trig = [t["prop_id"] for t in callback_context.triggered][0]

    # Start
    if "btn-start" in trig:
        if not recording:
            _reset_recording()
            recording = True
        return ("Recording: ON", f"Events: {len(rec_events)}", last_path or "")

    # Stop -> save EDF+
    if "btn-stop" in trig:
        if recording:
            recording = False
            out_path = _timestamped_edf_path(base_name or "session")
            try:
                _write_edf_plus(out_path)
                last_saved_path = out_path
                return (f"Recording: OFF (saved {os.path.basename(out_path)})",
                        f"Events: {len(rec_events)}",
                        out_path)
            except Exception as e:
                return (f"Recording: OFF (save failed: {e})",
                        f"Events: {len(rec_events)}",
                        last_path or "")
        return ("Recording: OFF", f"Events: {len(rec_events)}", last_path or "")

    # Events -> EDF+ annotations (only if recording)
    if recording and any(k in trig for k in ["ev1.n_clicks","ev2.n_clicks","ev3.n_clicks","ev4.n_clicks"]):
        label = "Event 1" if "ev1" in trig else "Event 2" if "ev2" in trig else "Event 3" if "ev3" in trig else "Event 4"
        onset = time.time() - rec_start_time
        rec_events.append((float(onset), label))
        return ("Recording: ON", f"Events: {len(rec_events)}", last_path or "")

    # no-op
    return ("Recording: ON" if recording else "Recording: OFF",
            f"Events: {len(rec_events)}",
            last_path or "")

@app.callback(
    Output("graph-eeg","figure"),
    Output("graph-sensors","figure"),
    Output("graph-impedance","figure"),
    Input("interval-status","n_intervals")
)
def draw(_n):
    fig_empty = go.Figure()
    if eeg_buf.shape[1] < 5:
        return fig_empty, fig_empty, fig_empty

    cols  = eeg_buf.shape[1]
    t_rel = np.linspace(-cols/args.fs, 0, cols)
    eeg_mV = eeg_buf[:NUM_CH_EEG,:].astype(np.float32) * CONV_EEG_MV
    scale = 1000.0 if args.units == "uv" else 1.0
    y_label = "µV" if args.units == "uv" else "mV"
    offset_disp = args.offset * scale

    fig_eeg = go.Figure()
    for k in range(NUM_CH_EEG):
        y = eeg_mV[k] * scale + k * offset_disp
        fig_eeg.add_trace(go.Scattergl(x=t_rel, y=y, mode='lines', line={'width':1}, name=f"Ch{k+1}"))
    fig_eeg.update_layout(
        title=f"EEG (32ch, stacked) — {y_label}",
        xaxis_title="Time (s)",
        yaxis_title=f"{y_label} (offset)",
        showlegend=False
    )

    ppg0   = eeg_buf[IDX_PPG0,:].astype(np.float32)
    ppg1   = eeg_buf[IDX_PPG1,:].astype(np.float32)
    eda_raw= eeg_buf[IDX_EDA,:].astype(np.int32)
    out_s  = (eda_raw % 4096).astype(np.float32)
    out_s  = (np.mean(out_s) - out_s) * CONV_EDA_MV
    temp_c = eeg_buf[IDX_TEMP,:].astype(np.float32) * CONV_TEMP_C

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=ppg0, mode='lines', name="PPG0 (au)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=ppg1, mode='lines', name="PPG1 (au)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=out_s, mode='lines', name="EDA (mV)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=temp_c, mode='lines', name="Temp (°C)"))
    fig_sens.update_layout(title="Sensors (autoscale)", xaxis_title="Time (s)", yaxis_title="au / mV / °C",
                           legend={'orientation':'h'})

    if len(imp_r_hist) > 0:
        r = np.array(imp_r_hist, dtype=np.float32)
        t_imp = np.linspace(-len(r)/args.fs, 0, len(r))
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Scatter(x=t_imp, y=r, mode='lines', name="R (kΩ)"))
        fig_imp.update_layout(title="Impedance (EDA-derived)", xaxis_title="Time (s)", yaxis_title="kΩ")
    else:
        fig_imp = fig_empty

    return fig_eeg, fig_sens, fig_imp

if __name__ == "__main__":
    app.run(port=args.http_port, debug=False, use_reloader=False)
