#!/usr/bin/env python3
# app.py
# BB84 Visual Simulator — Lightweight modes, Eve beam, pulsing Alice/Bob,
# teaching mode, layout toggle, exports, QBER charts.
#
# Expects bb84_backend.run_bb84(...) to exist and return:
# alice_bits_raw, alice_bases, bob_bases, sifted_alice, sifted_bob, qber_sifted, qber_final, logs (opt)
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
from typing import Optional, List
from bb84_backend import run_bb84
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
# Optional PNG export
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

# -------------------------
# Page setup & CSS
# -------------------------
st.set_page_config(page_title="BB84 Visual Simulator", layout="wide")
if "use_real_device" not in st.session_state:
    st.session_state.use_real_device = False

if "shots_on_device" not in st.session_state:
    st.session_state.shots_on_device = 1024 # Or whatever default value you prefer
st.markdown(
    """
    <style>
      body { background: #0b1220; color: #e5e7eb; }
      .pill {display:inline-block; padding:4px 10px; border-radius:999px; background:#111827; color:#e5e7eb; margin-right:6px; font-size:12px}
      .keychip {display:inline-block; padding:6px 10px; border-radius:10px; margin:4px; background:#0b1220; color:#e2e8f0; border:1px solid #172039}
      .inspect {background:#0b1220; border:1px solid #172039; border-radius:12px; padding:10px;}
      .small {font-size:12px; color:#b6c2d9}
      hr {border-color:#172039}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("BB84 Quantum Key Distribution —  Visual Simulator")
st.caption("Two visualization modes: Lightweight. Teaching & layout toggles included.")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Simulation Controls")

preset = st.sidebar.selectbox("Scenario Presets", [
    "Custom (manual)", "Clean Fiber (demo)", "Noisy Fiber", "Aggressive Eve", "Hardware Demo"
])
preset_defaults = {
    "Custom (manual)": {"n_qubits": 32, "eve_prob": 0.25, "noise": True, "channel_noise_per_km": 0.2, "shots": 1},
    "Clean Fiber (demo)": {"n_qubits": 32, "eve_prob": 0.0, "noise": False, "channel_noise_per_km": 0.05, "shots": 1},
    "Noisy Fiber": {"n_qubits": 48, "eve_prob": 0.05, "noise": True, "channel_noise_per_km": 0.8, "shots": 1},
    "Aggressive Eve": {"n_qubits": 40, "eve_prob": 0.6, "noise": True, "channel_noise_per_km": 0.3, "shots": 1},
    "Hardware Demo": {"n_qubits": 16, "eve_prob": 0.15, "noise": True, "channel_noise_per_km": 0.1, "shots": 4},
}
defaults = preset_defaults.get(preset, preset_defaults["Custom (manual)"])

n_qubits = st.sidebar.slider("Number of Qubits", 1, 32, min(defaults["n_qubits"], 32))
eve_prob = st.sidebar.slider("Eve Intercept Probability", 0.0, 1.0, defaults["eve_prob"], step=0.01)
add_noise = st.sidebar.checkbox("Add channel/device noise", defaults["noise"])
channel_noise_per_km = st.sidebar.number_input("Channel noise per km (%)", 0.0, 5.0, defaults["channel_noise_per_km"], step=0.1)
error_correction = st.sidebar.checkbox("Enable Error Correction (EC)", value=True)
privacy_amplification = st.sidebar.checkbox("Enable Privacy Amplification (PA)", value=False)
# -------------------------
# Backend selection
# -------------------------
backend_choice = st.sidebar.selectbox(
    "Choose backend",
    ["Custom Classical Simulator", "Aer Simulator", "Real Quantum Device"]
)

use_custom_sim = backend_choice == "Custom Simulator"
use_aer = backend_choice == "Aer Simulator"
use_real_device = backend_choice == "Real Quantum Device"

if use_real_device:
    try:
        # Initialize IBM Q service with your chosen instance
        service = QiskitRuntimeService()
        backend_name = "ibm_brisbane"
        # Fetch all operational real devices
        real_devices = service.backends(operational=True, simulator=False)
        if not real_devices:
            st.sidebar.warning("No real devices available. Falling back to simulator.")
            use_real_device = False
        else:
            # Pick the device with the fewest pending jobs
            backend = min(real_devices, key=lambda b: b.status().pending_jobs)

            # Fetch device configuration
            max_qubits = backend.configuration().n_qubits
            st.sidebar.info(
                f"⚠️ Real device: {backend.name} (max {max_qubits} qubits). "
                "Results may differ from simulator due to noise and queue times."
            )

            # Check qubit count
            if n_qubits > max_qubits:
                st.sidebar.error(
                    f"Selected qubits ({n_qubits}) exceed device capacity ({max_qubits}). "
                    "Reduce qubits to run on hardware."
                )
                st.stop()
            # User input for shots
            shots = st.sidebar.number_input(
                "Hardware shots (for real device)",
                min_value=1,
                max_value=128,
                value=defaults["shots"],
                step=1,
                key="hardware_shots"
            )

            # Create Sampler for measurement
            # Use runtime session to create sampler
            sampler = None
            runtime_context = None
            try:
                sampler = Sampler(mode=backend) # Pass the backend, no 'session'
                
            except Exception as e:
                st.sidebar.error(f"Failed to create sampler on real device: {e}")
                use_real_device = False


    except Exception as e:
        st.sidebar.error(f"Failed to access real device: {e}")
        st.stop()
else:
    max_qubits = 1024  # arbitrary large number for simulator
    shots = defaults["shots"]

seed = st.sidebar.number_input("Random Seed", value=42, step=1)
save_keys_switch = st.sidebar.checkbox("Save keys to result", value=True)
show_summary_ui = st.sidebar.checkbox("Show simulation summary panel", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Alice (optional manual input)")
alice_bits_input = st.sidebar.text_input("Alice bits (0/1, e.g. 010101)", value="")
alice_bases_input = st.sidebar.text_input("Alice bases (0/1 or Z/X or +/-/x)", value="")

st.sidebar.markdown("---")
visual_mode = "Lightweight"
layout_mode = "Dashboard"
teaching_mode = st.sidebar.checkbox("Show Eve teaching details (measured bases/bits)", value=False)
show_eve_traces = st.sidebar.checkbox("Show Eve traces (detailed)", value=False)
animation_speed = st.sidebar.slider("Animation speed (ms per frame, lower = faster, higher = slower)", 40, 400, 180, step=10)
vis_distance = st.sidebar.slider(
    "Visualization Distance (km)",
    0, 500, 40, step=5
)
run_clicked = st.sidebar.button("▶ Run Simulation", use_container_width=True)

# -------------------------
# Helpers & visual settings
# -------------------------
BIT_COLOR = {0: "#60a5fa", 1: "#fb923c"}  # blue / orange
BASIS_SYMBOL = {0: "+", 1: "×"}
BASIS_MARKER = {0: "circle", 1: "diamond"}

def parse_bits_str(s: str) -> Optional[List[int]]:
    s = s.strip().replace(" ", "")
    if not s: return None
    if any(ch not in "01" for ch in s): return None
    return [int(c) for c in s]

def parse_bases_str(s: str, expected_len: Optional[int] = None) -> Optional[List[int]]:
    if not s: return None
    mapping = {"0":0,"1":1,"z":0,"Z":0,"+":0,"x":1,"X":1,"×":1,"*":1}
    out = []
    for ch in s:
        if ch not in mapping: return None
        out.append(mapping[ch])
    if expected_len is not None and len(out) != expected_len:
        return None
    return out

def eve_mask(n, p, seed_val):
    rng = np.random.default_rng(int(seed_val) + 1337)
    return rng.random(n) < p

def compute_scale(n_qubits):
    n = max(1, int(n_qubits))
    scale = max(1.0, n / 32.0)
    font_size = int(max(10, 20 / scale))
    marker_size = int(max(10, 30 / scale))
    glow_size = int(max(20, 80 / scale))
    slider_dtick = int(max(1, round(n / 16)))
    return {"font_size": font_size, "marker_size": marker_size, "glow_size": glow_size, "dtick": slider_dtick}

def simple_particle(x, y, bit, basis, name, size=16, highlight=False, dimmed=False):
    base_color = "#ef4444" if highlight else BIT_COLOR[int(bit)]
    color = f"rgba({int(int(base_color[1:3],16)*0.6)},{int(int(base_color[3:5],16)*0.6)},{int(int(base_color[5:7],16)*0.6)},0.6)" if dimmed else base_color
    symbol = BASIS_MARKER[int(basis)]
    return go.Scatter(x=[x], y=[y], mode="markers+text",
                      marker=dict(size=size, color=color, symbol=symbol, line=dict(width=1, color="#e5e7eb")),
                      text=[f"{bit}|{BASIS_SYMBOL[int(basis)]}"], textposition="bottom center",
                      name=name, hovertemplate="<b>%{text}</b>", showlegend=False)

def eve_beam(xa, ya, xb, yb, active=True, width=6):
    # beam drawn as a thick semi-transparent line between Alice and Eve or Eve and Bob for effect
    if not active:
        return None
    return go.Scatter(x=[xa, xb], y=[ya, yb], mode="lines",
                      line=dict(width=width, color="#ef4444", dash='dash'), hoverinfo="skip", showlegend=False)

def eve_flash(x, y, on=True, size=64):
    return go.Scatter(x=[x], y=[y], mode="markers",
                      marker=dict(size=size if on else 0.001, opacity=0.45 if on else 0,
                                  color="#ef4444", symbol="circle-open", line=dict(width=4, color="#ef4444")),
                      hoverinfo="skip", showlegend=False, visible=on)

def pulsing_node_text(x, y, label, pulse=False, base_font=18):
    size = base_font + 10 if pulse else base_font
    return go.Scatter(x=[x], y=[y], mode="text",
                      text=[f"<span style='font-size:{size}px'><b>{label}</b></span>"],
                      hoverinfo="skip", showlegend=False)

def visualize_outcome(alice_bit, alice_basis, bob_basis, intercepted, noise_prob, rng):
    # local mini-sim for visuals consistent with earlier code
    if intercepted:
        eve_basis = int(rng.integers(0, 2))
        eve_bit = alice_bit if eve_basis == alice_basis else int(rng.integers(0, 2))
        resent_bit, resent_basis = eve_bit, eve_basis
    else:
        resent_bit, resent_basis = alice_bit, alice_basis
    if bob_basis == resent_basis:
        bob_bit = resent_bit
    else:
        bob_bit = int(rng.integers(0, 2))
    if rng.random() < noise_prob:
        bob_bit ^= 1
    kept = (bob_basis == alice_basis)
    error = (kept and (bob_bit != alice_bit))
    return {"resent_bit": resent_bit, "resent_basis": resent_basis, "bob_bit": bob_bit, "kept": kept, "error": error,
            "eve_basis": (resent_basis if intercepted else None), "eve_bit": (resent_bit if intercepted else None)}

# -------------------------
# Quick guide
# -------------------------
with st.expander("BB84 Quick Guide (4 steps) — click to expand", expanded=False):
    st.markdown("""
    **Prepare**: Alice encodes bits (0/1) into qubits using random bases (Z or X: + / ×).  
    **Transmit**: Eve may intercept (intercept-resend). LightWeight mode highlights intercept events.  
    **Measure**: Bob measures in random bases.  
    **Sift**: Alice & Bob reveal bases and keep positions where bases matched (sifted key).  
    **QBER**: Fraction of mismatches in sifted key (didactic alarm ≈ 11%).
    """)

st.markdown("<hr/>", unsafe_allow_html=True)

# -------------------------
# Run simulation
# -------------------------
if run_clicked:
    # validate manual inputs
    sampler = None
    use_real_device = st.session_state.use_real_device
    shots = st.session_state.shots_on_device
    user_bits = parse_bits_str(alice_bits_input) if alice_bits_input else None
    user_bases = None
    if alice_bases_input:
        user_bases = parse_bases_str(alice_bases_input, expected_len=len(user_bits) if user_bits else None)
        if user_bases is None:
            st.sidebar.error("Alice bases invalid or length mismatch with bits. Use 0/1 or Z/X or +/-/x.")
            st.stop()

    effective_n = len(user_bits) if user_bits else n_qubits
    # --- real device / simulator warning ---
    if use_real_device:
        try:
            service = QiskitRuntimeService()
            real_devices = service.backends(operational=True, simulator=False)
            if not real_devices:
                raise RuntimeError("No real devices available. Using Aer simulator.")

            backend = min(real_devices, key=lambda b: b.status().pending_jobs)
            
            # This is the correct way to create the Sampler in V2.
            sampler = Sampler(mode=backend) 
            
            # 🛑 REMOVE THIS LINE: it's what's causing your error.
            # sampler.options.shots = shots

        except Exception as e:
            st.sidebar.error(f"Failed to create sampler on real device: {e}")
            use_real_device = False
    else:
        if use_custom_sim:
            st.info("Running on default classical math simulator (no Aer). Results are idealized with basic noise model.")
        elif use_aer:
            st.info("Running on Qiskit Aer simulator (ideal gates, readout/depolarizing noise optional).")

    with st.spinner("Running BB84 backend..."):
        result = run_bb84(
            n_qubits=effective_n,
            eve_prob=float(eve_prob),
            run_on_real_device=use_real_device,
            run_on_aer=use_aer,
            run_on_custom=use_custom_sim,
            shots_on_device=int(shots),
            sampler=sampler,
            noise=bool(add_noise),
            ec=bool(error_correction),
            pa=bool(privacy_amplification),
            pa_target_len=None,
            seed=int(seed),
            save_keys=True,
            alice_bits_user=user_bits,
            alice_bases_user=user_bases,
            aer_dep=0.005,
            aer_ro=0.02,
        )
    
    # pull arrays
    alice_bits = result.get("alice_bits_raw", [])
    alice_bases = result.get("alice_bases", [])
    bob_bases = result.get("bob_bases", [])
    sifted_alice = result.get("sifted_alice", [])
    sifted_bob = result.get("sifted_bob", [])
    qber_sifted = float(result.get("qber_sifted", 0.0))
    qber_final = float(result.get("qber_final", qber_sifted))
    if "ec_stats" in result:
        # Recalculate final QBER after EC if available
        corrected_len = result.get("final_key_length", len(sifted_alice))
        if corrected_len > 0:
            qber_final = 0.0  # post-EC keys should match

    # Eve info
    eve_indices = result.get("eve_indices", [])
    eve_bases = result.get("eve_bases", [])
    # Compute eve_bits if saved in final_key_bits (optional)
    eve_bits = result.get("eve_measured_bits", []) 

    if not (alice_bits and alice_bases and bob_bases):
        st.error("Backend didn't return required arrays (alice_bits_raw, alice_bases, bob_bases).")
        st.stop()

    # visuals params
    #vis_distance = st.sidebar.slider("Visualization Distance (km)", 0, 500, 40, step=5)
    noise_prob = (vis_distance * channel_noise_per_km) / 100.0 if add_noise else 0.0

    # deterministic Eve mask and precompute visual outcomes
    e_mask = eve_mask(len(alice_bits), eve_prob, seed)
    rng = np.random.default_rng(int(seed) + 2025)
    vis_outcomes = []
    max_vis_qubits = 32
    vis_range = min(len(alice_bits), max_vis_qubits)
    for i in range(vis_range):
        vis_outcomes.append(visualize_outcome(alice_bits[i], alice_bases[i], bob_bases[i], bool(e_mask[i]), noise_prob, rng))

    # coords & scaling
    xA, xE, xB = 0.06, 0.50, 0.94
    yLine = 0.5
    scale = compute_scale(len(alice_bits))
    marker_size = scale["marker_size"]
    glow_size = scale["glow_size"]
    dtick = scale["dtick"]

    # build animation frames
    frames = []
    # Slow down when qubits are small, speed up when large
    if len(alice_bits) <= 8:
        frames_per_qubit = 20   # nice and slow
    elif len(alice_bits) <= 32:
        frames_per_qubit = 12
    else:
        frames_per_qubit = max(6, 12 - len(alice_bits)//64)

    frame_names = []
    wire = go.Scatter(x=[xA, xB], y=[yLine, yLine], mode="lines",
                      line=dict(width=12 if visual_mode=="Cinematic" else 8, color="#0f172a"),
                      hoverinfo="skip", showlegend=False)
    # node label placeholders (pulses toggled per frame)
    base_font = scale["font_size"] + 6

    for i in range(len(alice_bits)):
        a_bit, a_basis = alice_bits[i], alice_bases[i]
        b_basis = bob_bases[i]
        out = vis_outcomes[i]
        intercepted = bool(e_mask[i])

        xs = np.linspace(xA, xB, frames_per_qubit)
        ys = np.full(frames_per_qubit, yLine)
        eve_idx = frames_per_qubit // 2

        for f in range(frames_per_qubit):
            traces = []
            # determine node pulse: Alice pulses at send frame (f==0), Bob pulses at last frame
            alice_pulse = (f == 0)
            bob_pulse = (f == frames_per_qubit - 1)
            # choose bit & basis before/after Eve
            if f <= eve_idx:
                bit, basis = a_bit, a_basis
            else:
                bit, basis = out["resent_bit"], out["resent_basis"]
            label = f"{bit} | {BASIS_SYMBOL[int(basis)]}"

            # particle (lightweight or cinematic + highlight if intercepted after Eve)
            highlight = intercepted and f > eve_idx
            traces.append(simple_particle(xs[f], ys[f], bit, basis, name=label, size=max(12, marker_size), highlight=intercepted and f > eve_idx))
            if intercepted and f == eve_idx:
                # flash at Eve's position
                traces.append(eve_flash(xE, yLine, on=True, size=marker_size*2))
                # simple line from Alice->Eve and Eve->Bob
                traces.append(eve_beam(xs[0], ys[0], xE, yLine, active=True, width=marker_size*0.5))
                traces.append(eve_beam(xE, yLine, xs[-1], ys[-1], active=True, width=marker_size*0.35))
                # optionally show Eve measured bit if teaching mode
                if teaching_mode:
                    eve_text = f"Eve measured: {out.get('eve_bit')}|{BASIS_SYMBOL[int(out.get('eve_basis') or 0)]}" if out.get('eve_bit') is not None else "Eve measured: ?"
                    traces.append(go.Scatter(
                        x=[xE], y=[yLine+0.08], mode="text",
                        text=[f"<span style='font-size:{base_font-2}px; color:#fca5a5'>{eve_text}</span>"],
                        hoverinfo="skip", showlegend=False
                    ))
            # node labels (Alice, Eve, Bob), pulsing text
            traces.append(pulsing_node_text(xA, yLine, "Alice", pulse=alice_pulse, base_font=base_font))
            traces.append(pulsing_node_text(xE, yLine, "Eve", pulse=(f==eve_idx), base_font=base_font))
            traces.append(pulsing_node_text(xB, yLine, "Bob", pulse=bob_pulse, base_font=base_font))

            # final Bob annotations on last frame
            if f == frames_per_qubit - 1:
                kept = out["kept"]; error = out["error"]
                traces.append(go.Scatter(x=[xB], y=[yLine+0.14], mode="text",
                                         text=[f"<span style='font-size:{base_font}px'>Bob basis: <b>{BASIS_SYMBOL[int(b_basis)]}</b></span>"],
                                         hoverinfo="skip", showlegend=False))
                if kept and not error:
                    badge_html = f"<span style='font-size:{base_font}px'>✅ <b>Kept (match)</b></span>"
                elif kept and error:
                    badge_html = f"<span style='font-size:{base_font}px'>❌ <b>Error (mismatch)</b></span>"
                else:
                    badge_html = f"<span style='font-size:{base_font}px'>🗑️ <b>Discarded</b></span>"
                traces.append(go.Scatter(x=[xB], y=[yLine-0.14], mode="text", text=[badge_html], hoverinfo="skip", showlegend=False))

            # optionally display Eve trace lines in teaching mode for the whole path
            if (show_eve_traces or teaching_mode) and f == eve_idx:
                # show a faint marker at the point where Eve measured and resend (visual cue)
                traces.append(go.Scatter(x=[xE], y=[yLine], mode="markers+text",
                                         marker=dict(size=glow_size*0.4, color="#fca5a5", symbol="x"),
                                         text=[f"{out.get('eve_bit') if out.get('eve_bit') is not None else '?'}|{BASIS_SYMBOL[int(out.get('eve_basis') or 0)]}"],
                                         textposition="top center", showlegend=False, hoverinfo="skip"))

            fname = f"q{i}_f{f}"
            frame_names.append(fname)
            frames.append(go.Frame(name=fname, data=[wire] + traces,
                                   layout=go.Layout(
                                       annotations=[dict(x=0.5, y=0.975, xref='paper', yref='paper', showarrow=False,
                                                         text=(f"<span style='font-size:{base_font+2}px'><b>Qubit {i+1}/{len(alice_bits)}</b> — Eve p={eve_prob:.2f} | Dist={vis_distance} km | Noise≈{noise_prob*100:.2f}%</span>"))],
                                       xaxis=dict(range=[0, 1], showticklabels=False, zeroline=False, showgrid=False),
                                       yaxis=dict(range=[0, 1], showticklabels=False, zeroline=False, showgrid=False, scaleanchor='x'),
                                       margin=dict(l=40, r=40, t=100, b=60),
                                   )))

    if not frames:
        st.warning("No frames generated for animation.")
        st.stop()

    data0 = frames[0].data

    # slider steps (prune labels if too many)
    slider_steps = []
    for idx, fname in enumerate(frame_names):
        label = fname if len(frame_names) <= 80 else f"f{idx}"
        slider_steps.append(dict(method="animate", label=label,
                                 args=[[fname], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]))

    # adapt per-frame duration to total qubits
    per_frame=int(animation_speed)
    # compose figure
    fig = go.Figure(
        data=data0,
        layout=go.Layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            font=dict(color="#e5e7eb"),
            updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.12,
                              buttons=[
                                  dict(label="▶ Play", method="animate",
                                       args=[frame_names, {"frame": {"duration": per_frame, "redraw": True}, "fromcurrent": True, "transition": {"duration": 60}}]),
                                  dict(label="⏹ Stop", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                              ])],
            sliders=[dict(active=0, pad={"t": 16}, currentvalue={"prefix": "Frame: ", "visible": True, "xanchor": "right"}, steps=slider_steps)]
        ),
        frames=frames
    )
    fig.update_layout(margin=dict(l=30, r=30, t=90, b=60))
    fig.update_xaxes(tickmode="linear", dtick=dtick)

    # show layout: Dashboard (wide left) or Sidebar (balanced)
    if layout_mode == "Dashboard":
        col_anim, col_info = st.columns([2.6, 1.0])
    
    with col_anim:
        st.markdown("### BB84 Encoding Table")
        data = [
        {"Bit": "0", "Z-basis": "|0⟩", "X-basis": "|+⟩"},
        {"Bit": "1", "Z-basis": "|1⟩", "X-basis": "|−⟩"}
        ]
        st.table(data)
        
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("### Inspector")
        st.markdown("<div class='inspect'>Play or scrub. Table shows the first 12 qubits' outcomes; enable Teaching Mode to show Eve's measured basis/bit where applicable.</div>", unsafe_allow_html=True)
        rows = []
        
        for i in range(min(12, len(alice_bits))):
            o = vis_outcomes[i]
            eve_status = "Yes" if e_mask[i] else "No"
            rows.append({
                "#": i+1,
                "A bit|basis": f"{alice_bits[i]}|{BASIS_SYMBOL[int(alice_bases[i])] }",
                "Eve?": "Yes" if e_mask[i] else "No",
                "Resent": f"{o['resent_bit']}|{BASIS_SYMBOL[int(o['resent_basis'])]}",
                "Bob basis": BASIS_SYMBOL[int(bob_bases[i])],
                "Bob bit": o['bob_bit'],
                "Kept": "✓" if o['kept'] else "–",
                "Error": "✗" if o['error'] else " ",
            })
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Keys & metrics
    if len(sifted_alice) == 0:
        st.warning("No sifted key produced (few bases matched). Try increasing qubits.")

    with st.expander("Keys & Metrics", expanded=True):
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            st.markdown("**Alice (Sender)**")
            st.write("Bits (first 64):")
            st.markdown("".join([f"<span class='keychip'>{b}</span>" for b in alice_bits[:64]]), unsafe_allow_html=True)
            st.write("Bases (0=Z,1=X):")
            st.markdown("".join([f"<span class='keychip'>{b}</span>" for b in alice_bases[:64]]), unsafe_allow_html=True)
        with c2:
            st.markdown("**Bob (Receiver)**")
            st.write("Bases (0=Z,1=X):")
            st.markdown("".join([f"<span class='keychip'>{b}</span>" for b in bob_bases[:64]]), unsafe_allow_html=True)
            st.write("Sifted Alice (first 64):")
            st.markdown("".join([f"<span class='keychip'>{b}</span>" for b in sifted_alice[:64]]), unsafe_allow_html=True)
            st.write("Sifted Bob (first 64):")
            st.markdown("".join([f"<span class='keychip'>{b}</span>" for b in sifted_bob[:64]]), unsafe_allow_html=True)
        with c3:
            st.markdown("**Metrics**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Sifted Key Length", len(sifted_alice))
            m2.metric("QBER (sifted)", f"{qber_sifted*100:.2f}%")
            if error_correction:
                m3.metric("Residual QBER (after EC)", f"{qber_final*100:.2f}%")
            if privacy_amplification:
                st.write(f"Final key length after PA: {result.get('final_key_length', len(sifted_alice))}")

            st.markdown("<div class='small'>Didactic QBER alarm threshold: 11%</div>", unsafe_allow_html=True)
            if qber_final > 0.4:
                st.warning("⚠️ QBER above 11% — intercepted or very noisy channel (insecure).")
            else:
                st.success("✅ QBER below threshold — key may be secure (subject to EC/PA).")
            st.markdown("### Measurement Probability")
            st.latex(r"P(Bob = Alice) = \begin{cases}1 & \text{if bases match} \\ 0.5 & \text{otherwise} \end{cases}")

    # QBER Evolution bar
    stages, qbers = ["Sifted"], [qber_sifted]
    if error_correction:
        stages.append("After EC"); qbers.append(qber_final)
    if privacy_amplification:
        stages.append("After PA"); qbers.append(0.0)

    bar = go.Figure()
    bar.add_trace(go.Bar(x=stages, y=qbers, text=[f"{v*100:.2f}%" for v in qbers], textposition='auto'))
    bar.update_layout(title="QBER Evolution", yaxis=dict(range=[0, 1], title="QBER"),
                      xaxis_title="Stage", plot_bgcolor="#0b1220", paper_bgcolor="#0b1220",
                      font=dict(color="#e5e7eb"))
    st.plotly_chart(bar, use_container_width=True)

    # QBER vs Distance
    def simulate_qber_distance(n_qubits, max_distance, eve_prob, runs=3, channel_noise_per_km=0.002):
        distances = np.arange(0, max_distance + 1, 10)
        qber_no_eve, qber_with_eve = [], []
        for d in distances:
            r_no, r_e = [], []
            for _ in range(runs):
                res_no = run_bb84(n_qubits=n_qubits, eve_prob=0.0, noise=add_noise,
                                  ec=error_correction, pa=privacy_amplification, seed=seed)
                res_e = run_bb84(n_qubits=n_qubits, eve_prob=eve_prob, noise=add_noise,
                                 ec=error_correction, pa=privacy_amplification, seed=seed)
                q_no = min(1.0, float(res_no.get("qber_sifted", 0.0)) + d * channel_noise_per_km)
                q_e  = min(1.0, float(res_e.get("qber_sifted", 0.0)) + d * channel_noise_per_km)
                r_no.append(q_no); r_e.append(q_e)
            qber_no_eve.append(float(np.mean(r_no))); qber_with_eve.append(float(np.mean(r_e)))
        return distances, qber_no_eve, qber_with_eve

    distances, q_no, q_e = simulate_qber_distance(n_qubits=n_qubits, max_distance=200, eve_prob=eve_prob, runs=3,
                                                  channel_noise_per_km=channel_noise_per_km/100.0)

    line = go.Figure()
    line.add_trace(go.Scatter(x=distances, y=q_no, mode='lines+markers', name="No Eve"))
    line.add_trace(go.Scatter(x=distances, y=q_e, mode='lines+markers', name="With Eve"))
    line.update_layout(title="QBER vs Distance", xaxis_title="Distance (km)", yaxis_title="QBER",
                       yaxis=dict(range=[0,1]), plot_bgcolor="#0b1220", paper_bgcolor="#0b1220",
                       font=dict(color="#e5e7eb"))
    st.plotly_chart(line, use_container_width=True)
    # -------------------------
    # Backend summary & Eve traces
    # -------------------------
    # Display summary
    if show_summary_ui:
        # Determine security status based on QBER threshold
        secure_threshold = 0.11  # 11% didactic threshold
        is_secure = qber_final <= secure_threshold and result.get("secure", False)

        status_icon = "✅" if is_secure else "⚠️"
        status_color = "green" if is_secure else "red"

        st.markdown(f"""
        <div style='border:2px solid {status_color}; border-radius:12px; padding:15px; background:#111827;'>
            <h3 style='color:{status_color}'>{status_icon} BB84 Simulation Summary</h3>
            <p><b>Total qubits:</b> {effective_n}</p>
            <p><b>Eve intercept probability:</b> {eve_prob:.2f}</p>
            <p><b>Sifted key length:</b> {len(sifted_alice)}</p>
            <p><b>Sifted QBER:</b> {qber_sifted:.2%}</p>
            <p><b>Final key length:</b> {result.get("final_key_length", len(sifted_alice))}</p>
            <p><b>Final QBER:</b> {qber_final:.2%}</p>
            <p><b>Channel secure:</b> {"✅ Secure" if is_secure else "❌ Insecure"}</p>
        </div>
        """, unsafe_allow_html=True)


    # Optional: show Eve traces if teaching mode
    if teaching_mode and eve_indices:
        st.subheader("Eve Interception Highlights")
        st.write(f"Eve intercepted {len(eve_indices)} qubits ({len(eve_indices)/effective_n:.2%})")
        st.write("Intercepted qubit indices:", eve_indices)
        st.write("Eve bases:", eve_bases)
        st.write("Eve measured bits:", eve_bits)

    # Protocol logs
    with st.expander("Protocol Logs"):
        logs = result.get("logs", [])
        if logs:
            st.text("\n".join(logs))
        else:
            st.text("Key exchange complete. See metrics above.")

else:
    st.info("Adjust parameters on the left and click ▶ Run Simulation. Use lightweight mode for the full animated experience.")
