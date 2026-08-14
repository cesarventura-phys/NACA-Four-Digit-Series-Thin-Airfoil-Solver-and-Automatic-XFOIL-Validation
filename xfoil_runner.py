"""
xfoil_runner.py
Drives XFOIL non-interactively via subprocess, replaying the exact command
sequences that are known to work by hand:

  Polar file (one XFOIL session):
      NACA <code>
      GDES
      <enter>              (back to main menu)
      OPER
      ITER <n>
      PACC
      <polar_file>
      <enter>               (skip the dump-file prompt)
      ASEQ <amin> <amax> <astep>
      <enter>               (exit OPER)
      QUIT

  CPWR file (a second, completely separate XFOIL session):
      NACA <code>
      GDES
      <enter>
      OPER
      ITER <n>
      A <alpha>
      CPWR <cpwr_file>
      <enter>
      QUIT

Assumes the XFOIL executable lives in (or is reachable from) the working
directory this script is run from -- matches "run it from the same
directory as XFOIL.exe".
"""

import os
import subprocess
import re


# ------------------------------------------------------------
# NACA 4-digit code parsing
# ------------------------------------------------------------
def parse_naca4(code):
    """
    Parse a NACA 4-digit code string (e.g. "2412") into thin-airfoil
    parameters used by this solver, plus thickness (which the thin-airfoil
    theory ignores but XFOIL needs to build real geometry).

    Returns
    -------
    dict with keys: naca (str), M, P, T
        M : max camber, fraction of chord (first digit / 100)
        P : position of max camber, fraction of chord (second digit / 10)
        T : max thickness, fraction of chord (last two digits / 100)
    """
    code = str(code).strip()
    if not re.fullmatch(r'\d{4}', code):
        raise ValueError(f"'{code}' is not a valid NACA 4-digit code (need exactly 4 digits)")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:4]) / 100.0

    if p == 0.0 and m != 0.0:
        raise ValueError(
            f"NACA {code}: max camber position digit is 0 but camber digit is nonzero; "
            "this is not a valid NACA 4-digit definition"
        )

    return {'naca': code, 'M': m, 'P': p, 'T': t}


# ------------------------------------------------------------
# XFOIL script construction
# ------------------------------------------------------------
def build_polar_script(naca_code, alpha_min, alpha_max, alpha_step,
                        polar_file, iter_limit=400, viscous_re=None):
    """Command script for the polar (Cl vs alpha) run."""
    lines = []
    lines.append(f"NACA {naca_code}")
    lines.append("GDES")
    lines.append("")                       # back to main menu
    lines.append("OPER")
    if viscous_re is not None:
        lines.append(f"VISC {viscous_re}")
    lines.append(f"ITER {iter_limit}")
    lines.append("PACC")
    lines.append(polar_file)
    lines.append("")                       # skip dump-file prompt
    lines.append(f"ASEQ {alpha_min} {alpha_max} {alpha_step}")
    lines.append("")                       # exit OPER
    lines.append("QUIT")
    lines.append("")
    return "\n".join(lines)


def build_cpwr_script(naca_code, alpha_cp, cpwr_file, iter_limit=400, viscous_re=None):
    """Command script for the single-alpha Cp-distribution (CPWR) run."""
    lines = []
    lines.append(f"NACA {naca_code}")
    lines.append("GDES")
    lines.append("")                       # back to main menu
    lines.append("OPER")
    if viscous_re is not None:
        lines.append(f"VISC {viscous_re}")
    lines.append(f"ITER {iter_limit}")
    lines.append(f"A {alpha_cp}")
    lines.append(f"CPWR {cpwr_file}")
    lines.append("")                       # exit OPER
    lines.append("QUIT")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------
# Running XFOIL
# ------------------------------------------------------------
def run_xfoil(script_text, xfoil_exe="xfoil.exe", work_dir=".", timeout=120):
    """
    Feed `script_text` to XFOIL over stdin and capture its console output.

    Returns
    -------
    subprocess.CompletedProcess
    """
    exe_path = os.path.join(work_dir, xfoil_exe) if os.path.dirname(xfoil_exe) == "" \
        else xfoil_exe

    try:
        result = subprocess.run(
            [exe_path],
            input=script_text,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find '{xfoil_exe}' in '{os.path.abspath(work_dir)}'. "
            "Run this script from the same directory as the XFOIL executable, "
            "or pass xfoil_exe='path/to/xfoil'."
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"XFOIL did not finish within {timeout}s. It may be stuck waiting on a "
            f"prompt this script didn't anticipate. Partial output:\n{e.stdout}"
        )
    return result


def _check_output(path, label, result, verbose):
    ok = os.path.exists(path) and os.path.getsize(path) > 0
    if verbose:
        print(f"  {label} written : {ok}  ({path})")
    if not ok:
        if verbose:
            print(f"\n--- XFOIL output for {label} (for troubleshooting) ---")
            print(result.stdout[-3000:])
            if result.stderr:
                print("--- stderr ---")
                print(result.stderr[-1500:])
        raise RuntimeError(f"XFOIL did not produce {label} at {path}. See log above.")


def run_full_case(naca_code, alpha_min=-5.0, alpha_max=15.0, alpha_step=1.0,
                   alpha_cp=5.0, xfoil_exe="xfoil.exe", work_dir=".",
                   iter_limit=400, viscous_re=None, timeout=120, verbose=True):
    """
    End-to-end: parse the NACA code, run XFOIL once for the polar sweep and
    once (a fresh process, matching the known-working manual workflow) for
    the CPWR snapshot at a single alpha.

    Returns
    -------
    dict with keys:
        'geom'        : output of parse_naca4()
        'polar_file'  : path to the generated XFOIL polar file
        'cpwr_file'   : path to the generated XFOIL CPWR file
        'alpha_cp'    : the alpha (deg) the CPWR file was generated at
        'polar_log'   : XFOIL's raw stdout from the polar run
        'cpwr_log'    : XFOIL's raw stdout from the CPWR run
    """
    geom = parse_naca4(naca_code)
    naca = geom['naca']

    polar_file = f"polar_{naca}.txt"
    cpwr_file = f"cp_{naca}_a{alpha_cp:g}.txt"

    # Remove stale outputs so XFOIL doesn't stop to ask "overwrite? y/n"
    for f in (polar_file, cpwr_file):
        fp = os.path.join(work_dir, f)
        if os.path.exists(fp):
            os.remove(fp)

    if verbose:
        print(f"--- Running XFOIL for NACA {naca} ---")
        print(f"  mode          : {'viscous, Re=' + str(viscous_re) if viscous_re else 'inviscid'}")
        print(f"  iter limit    : {iter_limit}")

    # ---- Run 1: polar sweep ----
    if verbose:
        print(f"\n[1/2] Polar sweep: alpha {alpha_min} to {alpha_max} deg, step {alpha_step}")
    polar_script = build_polar_script(
        naca, alpha_min, alpha_max, alpha_step, polar_file,
        iter_limit=iter_limit, viscous_re=viscous_re,
    )
    polar_result = run_xfoil(polar_script, xfoil_exe=xfoil_exe, work_dir=work_dir, timeout=timeout)
    polar_path = os.path.join(work_dir, polar_file)
    _check_output(polar_path, "polar file", polar_result, verbose)

    # ---- Run 2: CPWR at a single alpha (fresh XFOIL process) ----
    if verbose:
        print(f"\n[2/2] Cp distribution at alpha = {alpha_cp} deg")
    cpwr_script = build_cpwr_script(
        naca, alpha_cp, cpwr_file, iter_limit=iter_limit, viscous_re=viscous_re,
    )
    cpwr_result = run_xfoil(cpwr_script, xfoil_exe=xfoil_exe, work_dir=work_dir, timeout=timeout)
    cpwr_path = os.path.join(work_dir, cpwr_file)
    _check_output(cpwr_path, "CPWR file", cpwr_result, verbose)

    return {
        'geom': geom,
        'polar_file': polar_path,
        'cpwr_file': cpwr_path,
        'alpha_cp': alpha_cp,
        'polar_log': polar_result.stdout,
        'cpwr_log': cpwr_result.stdout,
    }