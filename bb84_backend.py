#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bb84_backend.py
Core BB84 simulation logic, CLI, and utilities.

Features:
- Optional user-provided Alice bits & bases.
- AerSimulator or IBM runtime Sampler (defensive parsing).
- Optional Aer noise model.
- Cascade-lite error correction and BLAKE2s privacy amplification.
- Saves arrays needed for visualization when save_keys=True.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import random
import argparse
import json
import math
import os
import sys
import hashlib

# qiskit imports (optional at runtime — will error if not installed)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
    HAS_QISKIT = True
except Exception:
    HAS_QISKIT = False


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: Optional[int]):
    if seed is None:
        return
    np.random.seed(int(seed))
    random.seed(int(seed))

def rng_randint(rng, low, high=None, size=None):
    """
    Unified randint function for both RandomState and Generator.
    """
    if hasattr(rng, "integers"):  # New Generator API
        if high is None:
            high = low
            low = 0
        return rng.integers(low, high, size=size)  # Use rng.integers directly
    else:  # Old RandomState API
        if high is None:
            high = low
            low = 0
        return rng.randint(low, high, size=size)  # Use rng.randint directly

def _parity(bits: List[int]) -> int:
    return sum(bits) & 1
# -------------------------
# Helper functions
# -------------------------
def _bits_to_bytes(bits: List[int]) -> bytes:
    if not bits:
        return b""
    byte_len = (len(bits) + 7) // 8
    as_int = 0
    for b in bits:
        as_int = (as_int << 1) | (int(b) & 1)
    return as_int.to_bytes(byte_len, "big", signed=False)

# -------------------------
# Qiskit helpers (if available)
# -------------------------
def build_basic_noise_model(p_dep: float = 0.005, p_ro: float = 0.02):
    if not HAS_QISKIT:
        raise RuntimeError("Qiskit not available for noise model.")
    nm = NoiseModel()
    dep1 = depolarizing_error(p_dep, 1)
    ro = ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
    nm.add_all_qubit_quantum_error(dep1, ['h', 'x'])
    nm.add_all_qubit_readout_error(ro)
    return nm

def generate_bits_and_bases(n: int):
    bits = np.random.randint(2, size=n)
    bases = np.random.randint(2, size=n)
    return bits, bases

def prepare_qubits(bits: np.ndarray, bases: np.ndarray):
    if not HAS_QISKIT:
        raise RuntimeError("Qiskit not installed. prepare_qubits requires qiskit.")
    qubits = []
    for bit, base in zip(bits, bases):
        qc = QuantumCircuit(1, 1)
        if base == 0:
            if bit == 1:
                qc.x(0)
        else:
            if bit == 0:
                qc.h(0)
            else:
                qc.x(0); qc.h(0)
        qubits.append(qc)
    return qubits

def eavesdrop(qubits, eve_prob=0.3, sim_backend=None, seed: Optional[int] = None):
    """
    Intercept-resend: measure in a random basis and resend.
    Returns new qubits plus Eve’s measured bases/bits for visualization.
    """
    if not HAS_QISKIT:
        raise RuntimeError("Qiskit not installed. eavesdrop requires qiskit.")
    if sim_backend is None:
        sim_backend = AerSimulator()
    rng = random.Random(seed)
    eve_qubits, eve_measured_bits, eve_measured_bases, eve_indices = [], [], [], []
    for i, qc in enumerate(qubits):
        qc_copy = qc.copy()
        if rng.random() < eve_prob:
            measure_basis = rng.choice([0, 1])
            qc_copy2 = qc_copy.copy()
            if measure_basis == 1:
                qc_copy2.h(0)
            qc_copy2.measure(0, 0)
            t = transpile(qc_copy2, sim_backend)
            job = sim_backend.run(t, shots=1)
            result = job.result()
            counts = result.get_counts()
            bitstring = max(counts, key=counts.get)
            measured_bit = int(bitstring[-1])
            # record Eve measurement
            eve_indices.append(i)
            eve_measured_bits.append(measured_bit)
            eve_measured_bases.append(measure_basis)
            # resend
            qc_new = QuantumCircuit(1, 1)
            if measure_basis == 0:
                if measured_bit == 1:
                    qc_new.x(0)
            else:
                if measured_bit == 0:
                    qc_new.h(0)
                else:
                    qc_new.x(0); qc_new.h(0)
            eve_qubits.append(qc_new)
        else:
            eve_qubits.append(qc_copy)
    return eve_qubits, eve_indices, eve_measured_bits, eve_measured_bases

def measure_with_aer(qubits, bob_bases, sim_backend=None):
    if not HAS_QISKIT:
        raise RuntimeError("Qiskit not installed. measure_with_aer requires qiskit.")
    if sim_backend is None:
        sim_backend = AerSimulator()
    circuits = []
    for qc, base in zip(qubits, bob_bases):
        qc_m = qc.copy()
        if base == 1:
            qc_m.h(0)
        qc_m.measure(0, 0)
        circuits.append(qc_m)
    tcs = transpile(circuits, sim_backend)
    job = sim_backend.run(tcs, shots=1)
    res = job.result()
    results = []
    for idx in range(len(circuits)):
        counts = res.get_counts(idx)
        bitstring = max(counts, key=counts.get)
        measured_bit = int(bitstring[-1])
        results.append(measured_bit)
    return results

# Robust parsing for Sampler result items
def _extract_single_bit_from_sampler_result(item) -> int:
    data = getattr(item, "data", None)
    if data is None:
        raise RuntimeError("Sampler result item has no .data")

    # Try common shapes
    for attr in ("quasi_dists", "quasi_dist", "meas", "c", "counts"):
        obj = getattr(data, attr, None)
        if obj is None:
            continue

        # quasi-dists (list[dict] or dict)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            dist = obj[0]
            key = max(dist, key=dist.get)
            return int(str(key)[-1], 2) if isinstance(key, str) else (int(key) & 1)
        if isinstance(obj, dict):
            if obj:
                key = max(obj, key=obj.get)
                return int(str(key)[-1], 2) if isinstance(key, str) else (int(key) & 1)

        # sampler may expose a method
        if hasattr(obj, "get_counts"):
            counts = obj.get_counts()
            if isinstance(counts, dict) and counts:
                bitstring = max(counts, key=counts.get)
                return int(bitstring[-1])

    # Last resort: data.get_counts()
    if hasattr(data, "get_counts"):
        counts = data.get_counts()
        if isinstance(counts, dict) and counts:
            bitstring = max(counts, key=counts.get)
            return int(bitstring[-1])

    raise RuntimeError("Unable to parse Sampler result; unexpected schema")

def measure_with_sampler(qubits, bob_bases, sampler: "Sampler", shots: int = 1):
    circuits = []
    for qc, base in zip(qubits, bob_bases):
        qc_m = qc.copy()
        if base == 1:
            qc_m.h(0)
        qc_m.measure(0, 0)
        circuits.append(qc_m)

    job = sampler.run(circuits, shots=1)
    results = []
    try:
        for item in job.result():
            try:
                if shots == 1:
                    results.append(_extract_single_bit_from_sampler_result(item))
                else:
                    data = getattr(item, "data", None)
                    counts = None
                    for reg_name in ("meas", "c", "counts"):
                        reg = getattr(data, reg_name, None)
                        if reg is None:
                            continue
                        counts = reg.get_counts() if hasattr(reg, "get_counts") else reg
                        if isinstance(counts, dict) and counts:
                            break
                    if counts:
                        bitstring = max(counts, key=counts.get)
                        results.append(int(bitstring[-1]))
                    else:
                        results.append(0)
            except Exception:
                results.append(0)
    except Exception as e:
        raise RuntimeError(f"Sampler run/parse failed: {e}")
    return results

# -------------------------
# Post-processing: EC & PA
# -------------------------
def _binary_search_fix(alice: List[int], bob: List[int], idxs: List[int]) -> int:
    queries = 0
    L, R = 0, len(idxs)
    while R - L > 1:
        mid = (L + R) // 2
        queries += 1
        if _parity([alice[i] for i in idxs[L:mid]]) != _parity([bob[i] for i in idxs[L:mid]]):
            R = mid
        else:
            L = mid
    bob[idxs[L]] ^= 1
    return queries

def cascade_lite(
    alice: List[int],
    bob: List[int],
    block_size: int = 16,
    passes: int = 2,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int], dict]:
    rng = random.Random(seed)
    n = len(alice)
    alice_corr = alice[:]
    bob_corr = bob[:]
    total_queries = 0
    if n == 0:
        return alice_corr, bob_corr, {"parity_queries": 0, "passes": passes, "block_size": block_size}

    for _ in range(passes):
        perm = list(range(n))
        rng.shuffle(perm)
        blocks = [perm[i:i+block_size] for i in range(0, n, block_size)]
        for idxs in blocks:
            if not idxs:
                continue
            total_queries += 1
            a_par = _parity([alice_corr[i] for i in idxs])
            b_par = _parity([bob_corr[i] for i in idxs])
            while a_par != b_par:
                total_queries += _binary_search_fix(alice_corr, bob_corr, idxs)
                total_queries += 1
                a_par = _parity([alice_corr[i] for i in idxs])
                b_par = _parity([bob_corr[i] for i in idxs])
    stats = {"parity_queries": total_queries, "passes": passes, "block_size": block_size}
    return alice_corr, bob_corr, stats

def hash_pa(bits: List[int], out_bits: int = 128, salt: Optional[bytes] = None) -> Tuple[List[int], bytes]:
    """
    BLAKE2s PA. Returns (out_bits, salt). Salt is exactly 8 bytes to keep outputs portable.
    """
    if out_bits <= 0:
        return [], b""
    if salt is None:
        salt = os.urandom(8)  # fixed 8B
    else:
        salt = bytes(salt[:8])
    key_bytes = _bits_to_bytes(bits)
    h = hashlib.blake2s(key_bytes, digest_size=32, salt=salt).digest()
    out_bytes = math.ceil(out_bits / 8)
    raw = int.from_bytes(h[:out_bytes], "big")
    out = [(raw >> i) & 1 for i in reversed(range(out_bits))]
    return out, salt

# -------------------------
# Sifting / QBER / helpers
# -------------------------
def sift_keys(alice_bits, alice_bases, bob_results, bob_bases):
    sifted_alice, sifted_bob = [], []
    for a_bit, a_base, b_bit, b_base in zip(alice_bits, alice_bases, bob_results, bob_bases):
        if int(a_base) == int(b_base):
            sifted_alice.append(int(a_bit))
            sifted_bob.append(int(b_bit))
    return sifted_alice, sifted_bob

def calculate_qber(key1, key2):
    if len(key1) == 0:
        return 1.0
    errors = sum(a != b for a, b in zip(key1, key2))
    return errors / len(key1)
# -------------------------
# Classical math-based simulator (no Qiskit)
# -------------------------
# States
zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)

# Gates
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)

def apply_pauli_x(state: np.ndarray) -> np.ndarray:
    """Apply Pauli-X (bit flip) to a 2D state vector."""
    return X @ state

def apply_gate(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return np.dot(gate, state)

def prepare_state(bit: int, basis: int) -> np.ndarray:
    """Prepare a qubit mathematically."""
    state = zero if bit == 0 else one
    if basis == 1:  # X basis
        state = apply_gate(state, H)
    return state

def measure_state(state: np.ndarray, basis: int, rng: np.random.Generator) -> int:
    """Measure a state in Z or X basis."""
    if basis == 1:  # rotate to Z basis
        state = apply_gate(state, H)
    probs = np.abs(state) ** 2
    return int(rng.choice([0, 1], p=probs))

# -------------------------
# Orchestration: run_bb84
# -------------------------
def run_bb84(
    n_qubits: int = 32,
    eve_prob: float = 0.0,
    run_on_real_device: bool = False,
    run_on_aer=False,
    run_on_custom=False,
    noise: bool = False,
    ec: bool = False,
    pa: bool = False,
    pa_target_len: Optional[int] = None,
    seed: Optional[int] = None,
    save_keys: bool = True,
    alice_bits_user: Optional[List[int]] = None,
    alice_bases_user: Optional[List[int]] = None,
    shots_on_device: int = 1,
    sampler=None,
    aer_dep: float = 0.005,
    aer_ro: float = 0.02,
    distance_km: Optional[float] = None,
    channel_noise_rate: float = 0.02,
) -> Dict[str, Any]:
    """
    Full BB84 run with optional real-device toggle and fallback to simulator.
    Returns result dict; if save_keys=True includes arrays for UI visualization.
    """
    set_seed(seed)
    fallback_to_simulator = False
    fallback_message = ""

    # --- Alice bits & bases ---
    if alice_bits_user is not None and len(alice_bits_user) > 0:
        alice_bits = np.array([int(b) & 1 for b in alice_bits_user], dtype=int)
        n_qubits = len(alice_bits)
        if alice_bases_user is not None and len(alice_bases_user) == n_qubits:
            alice_bases = np.array([int(b) & 1 for b in alice_bases_user], dtype=int)
        else:
            alice_bases = np.random.randint(2, size=n_qubits)
    else:
        alice_bits, alice_bases = generate_bits_and_bases(n_qubits)

    # Prepare quantum circuits
    qubits = None
    try:
        if HAS_QISKIT:
            qubits = prepare_qubits(alice_bits, alice_bases)
    except Exception:
        qubits = None

    # --- Noise adjustment ---
    effective_ro = aer_ro
    if distance_km is not None:
        effective_ro = min(max(distance_km * channel_noise_rate, 0.0), 1.0)

    # Eve interception
    eve_indices, eve_measured_bits, eve_measured_bases = [], [], []
    if eve_prob > 0.0:
        if qubits is not None and HAS_QISKIT:
            qubits, eve_indices, eve_measured_bits, eve_measured_bases = eavesdrop(
                qubits, eve_prob=eve_prob, sim_backend=AerSimulator(), seed=seed
            )
        else:
            rng = np.random.RandomState(seed)
            for i in range(n_qubits):
                if rng.rand() < eve_prob:
                    eve_indices.append(i)
                    eve_basis = int(rng_randint(rng, 0, 2))
                    if eve_basis == int(alice_bases[i]):
                        eve_bit = int(alice_bits[i])
                    else:
                        eve_bit = int(rng_randint(rng, 0, 2))
                    eve_measured_bases.append(eve_basis)
                    eve_measured_bits.append(eve_bit)

    # Bob chooses bases
    bob_bases = np.random.randint(2, size=n_qubits)

    # --- Measurement ---
    bob_results = None
    if run_on_real_device and HAS_QISKIT and qubits is not None:
        try:
            service = QiskitRuntimeService()  # use your actual instance name

            real_devices = service.backends(operational=True, simulator=False)
            if not real_devices:
                raise RuntimeError("No real devices available. Using Aer simulator.")
            
            # choose the one with fewest pending jobs
            backend = min(real_devices, key=lambda b: b.status().pending_jobs)
            sampler = Sampler(mode=backend)
            options = Options(backend_name=backend.name)
            bob_results = measure_with_sampler(qubits, bob_bases, sampler, shots=shots_on_device)
            fallback_to_simulator = False
        except Exception as e:
            # Fallback to Aer
            fallback_to_simulator = True
            fallback_message = str(e)
            noise_model = build_basic_noise_model(aer_dep, effective_ro) if (noise and HAS_QISKIT) else None
            bob_results = measure_with_aer(qubits, bob_bases, sim_backend=AerSimulator(noise_model=noise_model))
    elif run_on_aer:
        # Local Aer or classical fallback
        if HAS_QISKIT and qubits is not None:
            noise_model = build_basic_noise_model(aer_dep, effective_ro) if (noise and HAS_QISKIT) else None
            bob_results = measure_with_aer(qubits, bob_bases, sim_backend=AerSimulator(noise_model=noise_model))
        else:
            # Classical fallback
            bob_results = []
            rng = np.random.RandomState(seed)
            for i in range(n_qubits):
                if eve_prob and (i in eve_indices):
                    idx_in_eve = eve_indices.index(i)
                    resent_bit = int(eve_measured_bits[idx_in_eve])
                    resent_basis = int(eve_measured_bases[idx_in_eve])
                else:
                    resent_bit = int(alice_bits[i])
                    resent_basis = int(alice_bases[i])
                if int(bob_bases[i]) == int(resent_basis):
                    bob_bit = resent_bit
                else:
                    bob_bit = int(rng_randint(rng, 0, 2))
                if noise and (rng.rand() < effective_ro):
                    bob_bit ^= 1
                bob_results.append(bob_bit)
            bob_results = list(bob_results)
    else:
        rng = np.random.default_rng(seed)

    # Ensure Python lists (we'll keep these consistent for save_keys later)
    alice_bits = (
        alice_bits_user if alice_bits_user is not None
        else rng_randint(rng, 0, 2, size=n_qubits).tolist()
    )
    alice_bases = (
        alice_bases_user if alice_bases_user is not None
        else rng_randint(rng,0, 2, size=n_qubits).tolist()
    )
    bob_bases = rng_randint(rng, 0, 2, size=n_qubits).tolist()

    bob_results = []
    vis_outcomes = []
    # Eve telemetry for UI
    eve_indices = []
    eve_measured_bits = []
    eve_measured_bases = []

    for i in range(n_qubits):
        # Alice prepares |0>,|1> (Z) or |+>,|-> (X) via H
        state = prepare_state(alice_bits[i], alice_bases[i])

        # Eve: intercept-resend (measure in random basis, then re-prepare)
        if rng.random() < eve_prob:
            e_basis = int(rng_randint(rng,0, 2))
            e_bit = measure_state(state, e_basis, rng)
            state = prepare_state(e_bit, e_basis)

            eve_indices.append(i)
            eve_measured_bases.append(e_basis)
            eve_measured_bits.append(int(e_bit))

        
        # Optional channel noise: flip qubit *before* Bob measures
        if noise and (rng.random() < channel_noise_rate):
            state = apply_pauli_x(state)

        # Bob measures in his basis
        b_bit = measure_state(state, bob_bases[i], rng)

        # Optional readout noise: flip measurement with prob effective_ro
        if noise and (rng.random() < effective_ro):
            b_bit ^= 1

        bob_results.append(int(b_bit))

        vis_outcomes.append({
            "resent_bit": int(alice_bits[i]),
            "resent_basis": int(alice_bases[i]),
            "bob_bit": int(b_bit),
            "kept": int(bob_bases[i]) == int(alice_bases[i]),
            "error": (int(b_bit) != int(alice_bits[i])) and (int(bob_bases[i]) == int(alice_bases[i])),
        })

    # Stash these so the common tail can include them in result/save_keys
    # (We continue to the common sifting/EC/PA code below.)
    

    # --- Sifting ---
    sifted_alice, sifted_bob = sift_keys(alice_bits, alice_bases, bob_results, bob_bases)
    qber_sifted = calculate_qber(sifted_alice, sifted_bob)

    result: Dict[str, Any] = {
        "n_qubits": n_qubits,
        "eve_prob": float(eve_prob),
        "qber_sifted": float(qber_sifted),
        "sifted_key_length": len(sifted_alice),
        "channel_noise_rate": float(effective_ro),
        "fallback_to_simulator": fallback_to_simulator,
        "fallback_message": fallback_message
    }

    # Logs
    logs: List[str] = []
    logs.append(f"Alice prepared {n_qubits} qubits")
    logs.append(f"Eve intercept probability: {eve_prob:.2f}")
    logs.append("Bob chose random bases")
    logs.append(f"Sifted key length: {len(sifted_alice)} (QBER={qber_sifted:.3f})")
    logs.append(f"Channel noise rate used: {effective_ro:.6f}")
    if fallback_to_simulator:
        logs.append(f"⚠️ Real device failed, falling back to Aer simulator: {fallback_message}")
        # Mark simulator type in logs
    if not run_on_aer and not run_on_real_device:
        logs.append("✅ Default classical math simulator used (no Aer/IBM backend).")
    
    # Save arrays for UI
    if save_keys:
        result.update({
            "alice_bits_raw": list(map(int, alice_bits)),
            "alice_bases": list(map(int, alice_bases)),
            "bob_bases": list(map(int, bob_bases)),
            "sifted_alice": list(map(int, sifted_alice)),
            "sifted_bob": list(map(int, sifted_bob)),
            "eve_indices": eve_indices,
            "eve_measured_bits": eve_measured_bits,
            "eve_measured_bases": eve_measured_bases,
            "vis_outcomes": vis_outcomes,
            "logs": logs
        })

    # --- Error correction ---
    corrected_alice = list(map(int, sifted_alice))
    corrected_bob = list(map(int, sifted_bob))
    qber_final = qber_sifted

    if ec:
        corrected_alice, corrected_bob, ec_stats = cascade_lite(corrected_alice, corrected_bob, seed=seed)
        qber_final = calculate_qber(corrected_alice, corrected_bob)
        result["ec_stats"] = ec_stats
        result["residual_qber_ec"] = float(qber_final)
        logs.append(
            f"Error correction: parity_queries={ec_stats.get('parity_queries',0)}, passes={ec_stats.get('passes',0)}"
        )
        logs.append(f"Residual QBER after EC: {qber_final:.3f}")
    else:
        logs.append("Error correction: disabled")

    result["secure"] = qber_final <= 0.11
    if result["secure"]:
        logs.append("✅ Security check: QBER below threshold → secure channel.")
    else:
        logs.append("⚠️ Security check: QBER above threshold → insecure channel.")

    # --- Privacy amplification ---
    if pa:
        source_key = corrected_alice
        if pa_target_len is None:
            pa_target_len = max(32, len(source_key) // 2)
        final_key, salt = hash_pa(source_key, out_bits=pa_target_len)
        result["final_key_length"] = len(final_key)
        result["pa_salt_hex"] = salt.hex()
        if save_keys:
            result["final_key_bits"] = list(map(int, final_key))
        logs.append(f"Privacy amplification: out_bits={len(final_key)} (salt={salt.hex()})")
    else:
        result["final_key_length"] = len(corrected_alice)
        if save_keys:
            result["final_key_bits"] = corrected_alice
        logs.append("Privacy amplification: disabled")

    return result

# -------------------------
# CLI for quick runs
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="BB84 QKD Simulator (Research-Grade)")
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--eve-prob", type=float, default=0.0)
    parser.add_argument("--real", action="store_true", help="Use real IBM quantum device")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--ec", action="store_true")
    parser.add_argument("--pa", action="store_true")
    parser.add_argument("--pa-bits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-keys", action="store_true")
    parser.add_argument("--alice-bits", type=str, default=None)
    parser.add_argument("--alice-bases", type=str, default=None)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--distance-km", type=float, default=None)
    parser.add_argument("--channel-noise-rate", type=float, default=0.02)
    args = parser.parse_args()

    def parse_bits_str(s: Optional[str]) -> Optional[List[int]]:
        if not s:
            return None
        s = "".join(ch for ch in s if ch in "01")
        if not s:
            return None
        return [int(c) for c in s]

    def parse_bases_str(s: Optional[str]) -> Optional[List[int]]:
        if not s:
            return None
        m = {"0":0,"1":1,"z":0,"Z":0,"+":0,"x":1,"X":1,"×":1,"*":1}
        out = [m[ch] for ch in s if ch in m]
        return out or None

    alice_bits_user = parse_bits_str(args.alice_bits)
    alice_bases_user = parse_bases_str(args.alice_bases)

    res = run_bb84(
        n_qubits=min(args.qubits, 32),
        eve_prob=args.eve_prob,
        run_on_real_device=args.real,
        run_on_aer=False,   # explicitly force NOT Aer
        run_on_custom=False,
        noise=args.noise,
        ec=args.ec,
        pa=args.pa,
        pa_target_len=args.pa_bits,
        seed=args.seed,
        save_keys=args.save_keys,
        alice_bits_user=alice_bits_user,
        alice_bases_user=alice_bases_user,
        shots_on_device=max(1, args.shots),
        distance_km=args.distance_km,
        channel_noise_rate=args.channel_noise_rate,
    )

    # ALERT for fallback
    if res.get("fallback_to_simulator"):
        print(f"\n⚠️ IBM real device failed, falling back to Aer simulator:")
        print(f"   Reason: {res.get('fallback_message')}\n")

    # -------------------------
    # Human-readable summary
    # -------------------------
    print("=== BB84 QKD Simulation Summary ===")
    print(f"Total qubits prepared: {res.get('n_qubits', 0)}")
    print(f"Eve intercept probability: {res.get('eve_prob', 0):.2f}")
    print(f"Sifted key length: {res.get('sifted_key_length', 0)}")
    print(f"Residual QBER: {res.get('residual_qber_ec', res.get('qber_sifted',0)):.3f}")
    print(f"Channel secure: {'✅ Secure' if res.get('secure') else '❌ Insecure'}")
    if args.pa:
        print(f"Final key length (after PA): {res.get('final_key_length',0)}")
    print(f"Channel noise rate used: {res.get('channel_noise_rate',0):.6f}")
    print("===================================\n")

    # Full JSON output (optional)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()