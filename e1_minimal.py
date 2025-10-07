#!/usr/bin/env python3
# e1_minimal.py — Minimal E1 live viewer (MATLAB-faithful amplitude conversion)
# Default COM port set to COM13 (same as MATLAB SerialCOM)

import argparse, threading, time, atexit
from collections import deque
from typing import Optional
import numpy as np
import serial
from dash import Dash, html, dcc, Output, Input, State, callback_context
import plotly.graph_objects as go

# --------------------- CLI ---------------------
p = argparse.ArgumentParser(description="Minimal E1 live viewer (MATLAB-faithful)")
p.add_argument("--port", default="COM13",  # ← default matches MATLAB
               help="Serial port (default: COM13)")
p.add_argument("--fs", type=int, default=250, help="Sample rate (Hz), default 250")
p.add_argument("--refresh", type=float, default=0.2, help="Chunk seconds, default 0.2")
p.add_argument("--offset", type=float, default=1.0, help="EEG vertical spacing in mV (MATLAB Offset=1)")
p.add_argument("--units", choices=["mv","uv"], default="mv",
               help="Display EEG in 'mv' (MATLAB-true) or 'uv' (×1000). Default mv.")
p.add_argument("--http_port", type=int, default=4548, help="Dash port")
args = p.parse_args()

# ------------------ Device layout ------------------
NUM_CH_EEG = 32
NUM_CH_ACC = 6
NUM_CH_TOT = NUM_CH_EEG + NUM_CH_ACC
BAUD = 460_800
TIMEOUT_S = 8.0

# Conversions (EXACTLY as in your MATLAB script)
CONV_EEG_MV  = 0.000286            # mV/LSB
CONV_EDA_MV  = 3.3/4096*1000.0     # mV/LSB
CONV_TEMP_C  = 0.0078125           # °C/LSB

# Aux indices (zero-based to match Python arrays)
IDX_PPG0 = 32
IDX_PPG1 = 33
IDX_EDA  = 34
IDX_TEMP = 35
IDX_CTRL2= 37

# ------------------ Serial / protocol ------------------
ser: Optional[serial.Serial] = None
mode_current = 0  # 0=EEG, 2=Impedance
stop_event = threading.Event()

def e1_go(mode: int):
    global mode_current
    if ser and ser.is_open:
        ser.write(bytes([170, mode*2 + 1, 85]))
        mode_current = mode

def e1_stop():
    if ser and ser.is_open:
        try: ser.write(bytes([170, 0, 85]))
        except Exception: pass

def open_serial():
    global ser
    if not ser or not ser.is_open:
        ser = serial.Serial(args.port, BAUD, timeout=TIMEOUT_S)
        time.sleep(0.05)
        e1_go(0)  # default EEG

def close_serial():
    global ser
    if ser and ser.is_open:
        e1_stop()
        time.sleep(0.05)
        ser.close()
    ser = None

atexit.register(close_serial)

def read_exact(n: int):
    buf = bytearray()
    while len(buf) < n and not stop_event.is_set():
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

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

imp_r_hist  = deque(maxlen=max_cols)   # R[kΩ]
imp_eda_hist= deque(maxlen=max_cols)   # EDA Out (mV)

# ------------------ Worker ------------------
def stream_worker():
    global eeg_buf
    rows = np.arange(NUM_CH_TOT*3).reshape(NUM_CH_TOT,3)
    while not stop_event.is_set():
        try:
            if not (ser and ser.is_open):
                time.sleep(0.2); continue
            need = NUM_CH_TOT * 3 * ns
            raw = read_exact(need)
            if raw is None:
                time.sleep(0.05); continue
            blk  = np.frombuffer(raw, dtype=np.uint8).reshape(NUM_CH_TOT*3, ns, order="F")
            msb, mid, lsb = blk[rows[:,0], :], blk[rows[:,1], :], blk[rows[:,2], :]
            vals = bytes24_to_int32(msb, mid, lsb)

            out = np.zeros((NUM_CH_TOT, ns+1), dtype=np.int32)
            out[:,1:] = vals
            out[:,0]  = out[:,1]   # MATLAB continuity trick

            eeg_buf = np.hstack([eeg_buf, out]) if eeg_buf.size else out
            if eeg_buf.shape[1] > max_cols:
                eeg_buf = eeg_buf[:, -max_cols:]

            if mode_current == 2:
                eda_raw = out[IDX_EDA,:].astype(np.int32)
                InSine  = np.floor(eda_raw/4096).astype(np.float32)
                OutSine = (eda_raw % 4096).astype(np.float32)
                InSine  = InSine - np.mean(InSine)
                OutSine = np.mean(OutSine) - OutSine  # flipped like MATLAB
                RawRMS  = float(np.sqrt(np.mean(InSine**2)))
                Rkohm   = (21.6*RawRMS) / max(1e-6, (525.0 - RawRMS))
                eda_mv  = OutSine * CONV_EDA_MV
                for _ in range(out.shape[1]):
                    imp_r_hist.append(Rkohm)
                for v in eda_mv:
                    imp_eda_hist.append(float(v))
        except Exception as e:
            print("worker error:", e)
            time.sleep(0.2)

open_serial()
stop_event.clear()
threading.Thread(target=stream_worker, daemon=True).start()

# ------------------ UI ------------------
app = Dash(__name__)
app.title = "E1 — Minimal (MATLAB amplitudes)"

app.layout = html.Div([
    html.H3("E1 Minimal Viewer (MATLAB-faithful amplitudes, no filters)"),
    html.Div([
        html.Button("EEG Mode", id="btn-eeg", n_clicks=0, style={'marginRight':'8px'}),
        html.Button("Impedance Mode", id="btn-imp", n_clicks=0, style={'marginRight':'8px'}),
        html.Button("Stop", id="btn-stop", n_clicks=0),
        html.Span("  ", id="status-text", style={'marginLeft':'16px','fontWeight':'bold'})
    ], style={'marginBottom':'8px'}),

    dcc.Store(id="amp-mode", data=0),
    dcc.Graph(id="graph-eeg", style={'height':'460px'}),
    dcc.Graph(id="graph-sensors", style={'height':'320px'}),
    dcc.Graph(id="graph-impedance", style={'height':'300px'}),
    dcc.Interval(id="tick", interval=200, n_intervals=0)
])

@app.callback(
    Output("amp-mode","data"),
    Output("status-text","children"),
    Input("btn-eeg","n_clicks"),
    Input("btn-imp","n_clicks"),
    Input("btn-stop","n_clicks"),
    State("amp-mode","data"),
    prevent_initial_call=True
)
def on_buttons(n_eeg, n_imp, n_stop, current):
    changed = [p['prop_id'] for p in callback_context.triggered][0]
    if "btn-eeg" in changed:
        open_serial(); e1_go(0)
        imp_r_hist.clear(); imp_eda_hist.clear()
        return 0, "Amplifier: EEG mode"
    elif "btn-imp" in changed:
        open_serial(); e1_go(2)
        return 2, "Amplifier: Impedance mode"
    elif "btn-stop" in changed:
        e1_stop(); close_serial()
        return current, "Amplifier: Stopped"
    return current, "OK"

@app.callback(
    Output("graph-eeg","figure"),
    Output("graph-sensors","figure"),
    Output("graph-impedance","figure"),
    Input("tick","n_intervals"),
    State("amp-mode","data")
)
def update(_n, mode):
    fig_empty = go.Figure()
    if eeg_buf.shape[1] < 5:
        return fig_empty, fig_empty, fig_empty

    cols  = eeg_buf.shape[1]
    t_rel = np.linspace(-cols/args.fs, 0, cols)

    # ---- EEG amplitude conversion (MATLAB-true): mV = raw*0.000286
    eeg_mV = eeg_buf[:NUM_CH_EEG,:].astype(np.float32) * CONV_EEG_MV

    # optional display in µV if requested
    scale = 1000.0 if args.units == "uv" else 1.0
    y_label = "µV" if args.units == "uv" else "mV"
    offset_disp = args.offset * scale   # keep offset in mV but display-scaled

    fig_eeg = go.Figure()
    for k in range(NUM_CH_EEG):
        y = eeg_mV[k] * scale + k * offset_disp
        fig_eeg.add_trace(go.Scattergl(x=t_rel, y=y, mode='lines', line={'width':1}, name=f"Ch{k+1}"))
    fig_eeg.update_layout(
        title=f"EEG (32ch, stacked) — units: {y_label}",
        xaxis_title="Time (s)",
        yaxis_title=f"{y_label} (offset by channel)",
        showlegend=False
    )

    # ---- Sensors (same math as MATLAB) ----
    ppg0   = eeg_buf[IDX_PPG0,:].astype(np.float32)           # AU
    ppg1   = eeg_buf[IDX_PPG1,:].astype(np.float32)           # AU
    eda_raw= eeg_buf[IDX_EDA,:].astype(np.int32)
    out_s  = (eda_raw % 4096).astype(np.float32)
    out_s  = (np.mean(out_s) - out_s) * CONV_EDA_MV           # mV, flipped
    temp_c = eeg_buf[IDX_TEMP,:].astype(np.float32) * CONV_TEMP_C

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=ppg0, mode='lines', name="PPG0 (AU)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=ppg1, mode='lines', name="PPG1 (AU)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=out_s, mode='lines', name="EDA Out (mV)"))
    fig_sens.add_trace(go.Scattergl(x=t_rel, y=temp_c, mode='lines', name="Temp (°C)"))
    fig_sens.update_layout(
        title="Sensors (autoscale)",
        xaxis_title="Time (s)",
        yaxis_title="AU / mV / °C",
        legend={'orientation':'h'}
    )

    # ---- Impedance panel (R[kΩ] from InSine RMS), identical math
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
