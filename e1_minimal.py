#!/usr/bin/env python3
# ─── Elephant Brain Labs — E1 Real-Time Amplifier Viewer ─────────────────────
# Simple live visualizer for OT Bioelettronica E1 amplifier
# - Streams 32-ch EEG and auxiliary sensors (PPG, EDA, Temp)
# - Shows connection status, auto-reconnect, and mode (EEG / Impedance)
# - MATLAB-faithful amplitude conversions, no filters, autoscaling
#
# pip install pyserial numpy dash plotly
# python e1_minimal.py
#
# Author: Elephant Brain Labs • October 2025

import argparse, threading, time, atexit, os
from collections import deque
from typing import Optional
import numpy as np
import serial
from serial.tools import list_ports
from dash import Dash, html, dcc, Output, Input, State, callback_context
import plotly.graph_objects as go
import base64

# --------------------- CLI ---------------------
p = argparse.ArgumentParser(description="Elephant Brain Labs — E1 Real-Time Amplifier Viewer")
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
CONV_EEG_MV  = 0.000286
CONV_EDA_MV  = 3.3/4096*1000.0
CONV_TEMP_C  = 0.0078125

# Aux indices
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

def stream_worker():
    """Read data continuously if connected."""
    global eeg_buf
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

# optional logo
encoded_logo = ""
for candidate in ["logo.png", "assets/logo.png"]:
    if os.path.exists(candidate):
        with open(candidate, "rb") as f:
            encoded_logo = base64.b64encode(f.read()).decode()

app = Dash(__name__)
app.title = "Elephant Brain Labs — E1 Real-Time Amplifier Viewer"

app.layout = html.Div([
    html.Div([
        html.Img(src=f"data:image/png;base64,{encoded_logo}", style={'height':'56px','marginRight':'12px'}) if encoded_logo else html.Div(),
        html.Div([
            html.H2("Elephant Brain Labs", style={'color':'#1E90FF','margin':'0'}),
            html.H3("E1 Real-Time Amplifier Viewer", style={'margin':'0','fontWeight':'bold'})
        ])
    ], style={'display':'flex','alignItems':'center','gap':'10px','marginBottom':'10px'}),

    html.Div([
        dcc.Dropdown(id="port-dd", clearable=False, style={'width':'220px','display':'inline-block'}),
        html.Button("Refresh Ports", id="btn-refresh", n_clicks=0, style={'marginLeft':'8px'}),
        html.Button("Connect", id="btn-connect", n_clicks=0, style={'marginLeft':'12px'}),
        html.Button("Disconnect", id="btn-disconnect", n_clicks=0, style={'marginLeft':'8px'}),
        html.Button("EEG Mode", id="btn-eeg", n_clicks=0, style={'marginLeft':'16px'}),
        html.Button("Impedance Mode", id="btn-imp", n_clicks=0, style={'marginLeft':'8px'}),
        html.Span(id="status-text", style={'marginLeft':'16px','fontWeight':'bold'})
    ], style={'marginBottom':'8px'}),

    dcc.Store(id="store-port", data=args.default_port),
    dcc.Store(id="store-auto", data=True),
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
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=ppg0, mode='lines', name="PPG0 (AU)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=ppg1, mode='lines', name="PPG1 (AU)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=out_s, mode='lines', name="EDA Out (mV)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=temp_c, mode='lines', name="Temp (°C)"))
    fig_sens.update_layout(title="Sensors (autoscale)", xaxis_title="Time (s)", yaxis_title="AU / mV / °C",
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
