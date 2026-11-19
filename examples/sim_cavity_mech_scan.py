###################################################################################
#  Copyright (c) 2026 by National Synchrotron Radiation Laboratory, University of Science and Technology of China
#  All rights reserved.
#  Authors: Ming Liu, mliu1018@ustc.edu.cn
###################################################################################
"""
Cavity pulse simulation with mechanical (Lorentz) modes - the llrflibs-rl
port of ``LLRFLibsPy/example/example_sim_cavity_mech2.py``.

It simulates the full open-loop RF chain (RF source -> I/Q modulator ->
amplifier -> cavity with beam current, microphonics and multi-mode mechanical
detuning) and draws the classic mech2 view:

    top panel    |Vc|   cavity voltage amplitude (MV)
    middle panel        cavity phase (deg)
    bottom panel        detuning driven by the mechanical modes (Hz)

plus (in a second figure) the I/Q traces of every internal chain signal.

Usage (from the repository root)::

    D:/Software/Anaconda/envs/rl/python.exe examples/sim_cavity_mech_scan.py --out gallery/mech_scan.png
    D:/Software/Anaconda/envs/rl/python.exe examples/sim_cavity_mech_scan.py --show --sim-len 16384
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from llrflibs.rf_control import ss_discrete
from llrflibs.rf_sim import cav_ss_mech, sim_scav_step

from llrflibs_rl.configs.cavity import nominal_sc_cavity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mech2-style cavity + mechanical-mode simulation")
    parser.add_argument("--sim-len", type=int, default=2048 * 8, help="total simulated samples")
    parser.add_argument("--pul-len", type=int, default=2048 * 6, help="pulse repetition length")
    parser.add_argument("--fsrc", type=float, default=-460.0,
                        help="RF source offset from carrier, Hz (mech2 default -460)")
    parser.add_argument("--amp-k", type=float, default=12e6,
                        help="amplifier voltage gain (mech2: 12e6 V/V = ~161.6 dB)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the microphonics noise")
    parser.add_argument("--out", default=None, help="PNG output path (default gallery/mech_scan.png)")
    parser.add_argument("--show", action="store_true", help="block and show the interactive windows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # ---- cavity parameters (canonical 1.3 GHz SC example, with mech modes) --
    cavity = nominal_sc_cavity()
    Ts = 1e-6
    f0, beta, roQ, QL = 1.3e9, cavity.beta, cavity.roQ_or_RoQ, cavity.QL
    RL = 0.5 * roQ * QL
    wh = np.pi * f0 / QL
    ib = cavity.beam_current
    t_fill, t_flat = cavity.filling_len, cavity.filling_len + cavity.flattop_len
    mech_modes = {
        "f": [280.0, 341.0, 460.0, 487.0, 618.0],
        "Q": [40.0, 20.0, 50.0, 80.0, 100.0],
        "K": [2.0, 0.8, 2.0, 0.6, 0.2],
    }
    print(f"cavity: wh={wh/(2*np.pi):.1f} Hz, RL={RL:.3e} Ohm, beam={ib*1e3:.1f} mA, "
          f"filling={t_fill}, flattop-end={t_flat}")

    # ---- thresholds of the chain ---------------------------------------------
    buf_size = args.sim_len
    base_pul = np.zeros(buf_size, dtype=complex)
    base_pul[:t_flat] = 1.0
    beam_pul = np.zeros(buf_size, dtype=complex)
    beam_pul[t_fill:t_flat] = ib

    status, Am, Bm, Cm, Dm = cav_ss_mech(mech_modes)
    status, Ad, Bd, Cd, Dd, _ = ss_discrete(Am, Bm, Cm, Dm, Ts=Ts, method="zoh",
                                            plot=False, plot_pno=10000)
    state_m = np.matrix(np.zeros(Bd.shape))
    state_vc = 0.0

    rng = np.random.default_rng(args.seed)
    dw0 = 2.0 * np.pi * 100.0
    dw = 0.0

    sig_src = np.zeros(args.sim_len, dtype=complex)
    sig_iqm = np.zeros(args.sim_len, dtype=complex)
    sig_amp = np.zeros(args.sim_len, dtype=complex)
    sig_vc = np.zeros(args.sim_len, dtype=complex)
    sig_vr = np.zeros(args.sim_len, dtype=complex)
    sig_dw = np.zeros(args.sim_len, dtype=float)

    pha_src = 0.0
    buf_id = -1
    for i in range(args.sim_len):
        # RF source (offset carrier) -> I/Q modulator pulse -> amplifier
        pha_src += 2.0 * np.pi * args.fsrc * Ts
        S0 = np.exp(1j * pha_src)
        buf_id += 1
        if buf_id >= args.pul_len:
            buf_id = 0
        S1 = S0 * base_pul[min(buf_id, buf_size - 1)]
        S2 = S1 * args.amp_k

        # microphonics (white, 10 Hz rms) + cavity with mechanical modes
        dw_micr = 2.0 * np.pi * rng.standard_normal() * 10.0
        status, vc, vr, dw, state_m = sim_scav_step(
            wh, dw, dw0 + dw_micr, S2, -RL * beam_pul[min(buf_id, buf_size - 1)],
            state_vc, Ts, beta=beta,
            state_m0=state_m, Am=Ad, Bm=Bd, Cm=Cd, Dm=Dd, mech_exe=True,
        )
        if not status:
            raise RuntimeError(f"sim_scav_step failed at sample {i}")
        state_vc = vc

        sig_src[i], sig_iqm[i], sig_amp[i] = S0, S1, S2
        sig_vc[i] = complex(np.asarray(vc).reshape(-1)[0])
        sig_vr[i] = complex(np.asarray(vr).reshape(-1)[0])
        sig_dw[i] = float(np.asarray(dw).reshape(-1)[0])

    # ---- figure 1: internal chain I/Q traces (mech2 style) -------------------
    plt.figure(figsize=(11, 4.5))
    for sig, name in ((sig_src, "src"), (sig_iqm, "IQ"), (sig_amp, "amp"), (sig_vc, "Vc")):
        plt.plot(np.real(sig), label=f"Re({name})")
        plt.plot(np.imag(sig), "--", label=f"Im({name})")
    plt.legend(ncol=2, fontsize=8)
    plt.xlabel("Time (Ts)")
    plt.ylabel("Signal (V)")
    plt.title("RF chain I/Q traces")
    plt.grid(alpha=0.3)

    # ---- figure 2: the three-panel mech2 view --------------------------------
    fig2 = plt.figure(figsize=(11, 8))
    ax1 = fig2.add_subplot(3, 1, 1)
    ax1.plot(np.abs(sig_vc) * 1e-6, color="#58a6ff")
    ax1.set_ylabel("|Vc| (MV)")
    ax1.set_title("Cavity pulse with mechanical (Lorentz) modes")
    ax1.grid(alpha=0.3)

    ax2 = fig2.add_subplot(3, 1, 2)
    ax2.plot(np.angle(sig_vc) * 180.0 / np.pi, color="#3fb950")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(alpha=0.3)

    ax3 = fig2.add_subplot(3, 1, 3)
    ax3.plot(sig_dw / (2.0 * np.pi), color="#d29922")
    ax3.set_ylabel("Detuning (Hz)")
    ax3.set_xlabel("Time (Ts)")
    ax3.grid(alpha=0.3)

    fig2.tight_layout()

    # ---- report a few numbers the user can compare ----------------------------
    flat = slice(t_fill, min(t_flat, args.sim_len))
    vc_flat = np.abs(sig_vc[flat])
    print(f"flattop |Vc| mean = {vc_flat.mean() * 1e-6:.3f} MV (target 25 MV)")
    print(f"detuning: mean {sig_dw[flat].mean() / (2*np.pi):.1f} Hz, "
          f"rms {np.std(sig_dw[flat]) / (2*np.pi):.1f} Hz, "
          f"peak-peak {np.ptp(sig_dw[flat]) / (2*np.pi):.1f} Hz")

    out = Path(args.out) if args.out else Path("gallery") / "mech_scan.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out, dpi=110)
    print(f"[mech_scan] wrote {out}")
    if args.show:
        plt.show(block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
