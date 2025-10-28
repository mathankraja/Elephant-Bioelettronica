#!/usr/bin/env python3
# Elephant Brain Labs — E1 Serial Simulator
# - Emulates OT Bioelettronica E1 over a COM port
# - Understands GO/STOP control bytes and streams 24-bit packed frames
# - Channels: 32 EEG + 2 PPG + 1 EDA (composite In/Out) + 1 Temp + 1 Ctrl2 = 38 total

# Author: @ Mathan K. Raja  Elephant Brain Labs • October 2025

import argparse, time, math, threading, sys
import numpy as np
import serial

# ---------------- CLI ----------------
ap = argparse.ArgumentParser("E1 Serial Simulator")
ap.add_argument("--port", required=True, help="Serial port to open (e.g., COM12)")
ap.add_argument("--baud", type=int, default=460800)
ap.add_argument("--fs", type=int, default=250, help="Sample rate Hz")
ap.add_argument("--refresh", type=float, default=0.2, help="Chunk seconds")
ap.add_argument("--eda_center", type=float, default=2048.0)
ap.add_argument("--alpha_uv", type=float, default=40.0, help="Alpha amplitude in µV")
ap.add_argument("--noise_uv", type=float, default=10.0, help="Noise stdev in µV")
args = ap.parse_args()

# ---------------- Layout & constants ----------------
NUM_CH_EEG = 32
NUM_CH_ACC = 6  # we don't synth accel; we use 2 PPG + EDA + Temp + Ctrl2 among these 6
NUM_CH_TOT = NUM_CH_EEG + NUM_CH_ACC  # 38
IDX_PPG0 = 32
IDX_PPG1 = 33
IDX_EDA  = 34
IDX_TEMP = 35
IDX_CTRL2= 37

# MATLAB conversions (for inverse mapping where needed)
CONV_EEG_MV = 0.000286
CONV_EDA_MV = 3.3/4096*1000.0
CONV_TEMP_C = 0.0078125

GO_HDR   = 170
STOP_CMD = bytes([GO_HDR, 0, 85])

# State
mode_lock = threading.Lock()
mode = 0  # 0 EEG, 2 Impedance
running = False
ser = None

# ---------------- Helpers ----------------
def _pack_24_signed(x_int32: np.ndarray) -> bytes:
    """
    Pack int32 values (range: -2^23 .. 2^23-1) into 24-bit two's complement.
    Output order: for sample j, channels 0..(N-1), bytes MSB, MID, LSB (big-endian by 3 bytes).
    """
    x = x_int32.copy()
    # Convert negatives to 24-bit two's complement
    neg = x < 0
    x[neg] = (1 << 24) + x[neg]
    msb = (x >> 16) & 0xFF
    mid = (x >> 8)  & 0xFF
    lsb = x & 0xFF
    return np.vstack([msb, mid, lsb]).astype(np.uint8).T.tobytes()

def _gen_chunk(t0: float, n: int, fs: int) -> np.ndarray:
    """
    Generate one (NUM_CH_TOT x n) int32 chunk (already "decoded" 24-bit values).
    We'll pack to 24-bit later. This mirrors your viewer's 'out[:,1:]' part.
    """
    t = np.linspace(0.0, (n-1)/fs, n, dtype=np.float64)
    # EEG in µV
    eeg_uv = np.zeros((NUM_CH_EEG, n), dtype=np.float32)
    for ch in range(NUM_CH_EEG):
        phase = 2*math.pi*(ch/NUM_CH_EEG)
        freq  = 10.0 + 0.15*math.sin(0.1*(t0) + ch*0.2)
        eeg_uv[ch] = (
            args.alpha_uv * np.sin(2*np.pi*freq*t + phase) +
            np.random.normal(0.0, args.noise_uv, n)
        ).astype(np.float32)
    # Convert µV -> raw counts using inverse of viewer conversion:
    # raw = (EEG_mV) / 0.000286 ; EEG_mV = EEG_µV / 1000
    eeg_raw = np.round((eeg_uv/1000.0) / CONV_EEG_MV)
    eeg_raw = np.clip(eeg_raw, -(1<<23), (1<<23)-1).astype(np.int32)

    # PPG AU
    pulse = 1.2
    ppg0 = (5000*np.maximum(0.0, np.sin(2*np.pi*pulse*t)) +
            500*np.sin(2*np.pi*2*pulse*t)).astype(np.int32)
    ppg1 = (4000*np.maximum(0.0, np.sin(2*np.pi*1.1*t + 1.0)) +
            400*np.sin(2*np.pi*2.2*t + 0.3)).astype(np.int32)

    # EDA composite: raw = InSine*4096 + OutSine
    insine = 260 + 80*np.sin(2*np.pi*0.2*t) + 10*np.random.randn(n)
    insine = np.clip(np.floor(insine), 0, 525).astype(np.int32)
    outs   = args.eda_center + 300*np.sin(2*np.pi*0.15*t + 0.7) + 50*np.random.randn(n)
    outs   = np.clip(np.floor(outs), 0, 4095).astype(np.int32)
    eda_raw = (insine * 4096 + outs).astype(np.int32)

    # Temperature °C -> raw
    temp_c  = 32.0 + 0.2*np.sin(2*np.pi*0.01*t) + 0.05*np.random.randn(n)
    temp_raw= np.clip(np.round(temp_c / CONV_TEMP_C), 0, (1<<23)-1).astype(np.int32)

    # Ctrl2 saw
    ctrl2 = np.mod(np.arange(n)*100, 10000).astype(np.int32)

    out = np.zeros((NUM_CH_TOT, n), dtype=np.int32)
    out[:NUM_CH_EEG,:] = eeg_raw
    out[IDX_PPG0,:]    = ppg0
    out[IDX_PPG1,:]    = ppg1
    out[IDX_EDA,:]     = eda_raw
    out[IDX_TEMP,:]    = temp_raw
    out[IDX_CTRL2,:]   = ctrl2
    return out

def _write_chunk_bytes(out: np.ndarray, ser: serial.Serial):
    """
    Serialize 'out' (NUM_CH_TOT x n int32) to the exact byte order your client expects:
    For sample j in [0..n-1]:
        For channel c in [0..NUM_CH_TOT-1]:
            write 3 bytes: MSB, MID, LSB (24-bit two's complement)
    """
    n = out.shape[1]
    # interleave per-sample/per-channel
    # We'll build one sample at a time to keep memory small
    for j in range(n):
        triplets = _pack_24_signed(out[:, j].astype(np.int32))
        # triplets is (NUM_CH_TOT * 3) bytes in MSB,MID,LSB order per channel
        ser.write(triplets)

# ---------------- Serial RX (control) ----------------
def control_reader():
    global running, mode
    buf = bytearray()
    while True:
        try:
            b = ser.read(1)
            if not b:
                continue
            buf += b
            # We look for 3-byte control frames anywhere in the stream
            while len(buf) >= 3:
                if buf[0] != GO_HDR:
                    buf.pop(0); continue
                frame = bytes(buf[:3])
                del buf[:3]
                if frame == STOP_CMD:
                    running = False
                    # print("[SIM] STOP")
                else:
                    # GO: [170, mode*2 + 1, 85]
                    if frame[2] != 85:
                        continue
                    mbyte = frame[1]
                    new_mode = (mbyte - 1) // 2
                    if new_mode in (0,2):
                        with mode_lock:
                            mode = new_mode
                        running = True
                        # print(f"[SIM] GO mode={mode}")
        except Exception:
            time.sleep(0.01)

# ---------------- Stream loop ----------------
def stream_writer():
    global running
    ns = max(1, int(round(args.fs * args.refresh)))
    period = ns / float(args.fs)
    t0 = time.time()
    while True:
        t_now = time.time()
        # Only stream when 'running' is True (after GO)
        if running:
            out = _gen_chunk(t_now, ns, args.fs)
            # If in impedance mode we can (optionally) tweak signals
            with mode_lock:
                m = mode
            if m == 2:
                # Slightly boost insine ripple by nudging EDA encoding
                # (already included in generator; left here to show knob)
                pass
            _write_chunk_bytes(out, ser)
        # pace
        sleep_left = (t0 + (int((t_now - t0)/period)+1)*period) - time.time()
        time.sleep(max(0.0, sleep_left))

# ---------------- Main ----------------
def main():
    global ser, running
    ser = serial.Serial(args.port, args.baud, timeout=0.01)
    print(f"[SIM] Opened {args.port} @ {args.baud}. Waiting for GO (AA, mode*2+1, 55)…")
    running = False  # wait for GO
    threading.Thread(target=control_reader, daemon=True).start()
    stream_writer()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
