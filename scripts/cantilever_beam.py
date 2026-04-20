"""
cantilever_beam.py
------------------------
Performs basic strength and serviceability checks for a cantilever beam
with a single point load at the free end, given a rectangular solid cross-section.

Checks performed:
  1. Flexural (bending) stress:  sigma_max <= F_b
  2. Shear stress:               tau_max   <= F_v
  3. Tip deflection:             delta_max <= delta_allow  (e.g. l/240)

All internal calculations use consistent SI base units:
  Force  → N
  Length → mm
  Stress → MPa  (= N/mm²)
  E      → MPa  (converted from GPa on input)

Usage:
  python scripts/cantilever_beam.py                      # uses default initial_values.json
  python scripts/cantilever_beam.py my_problem.json      # uses a custom JSON file
"""

import json
import sys
import re


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_problem(filepath: str) -> dict:
    """Load and return the problem definition from a JSON file."""
    with open(filepath, "r") as fh:
        return json.load(fh)


def parse_deflection_limit(limit_str: str, length_mm: float) -> float:
    """
    Parse a deflection limit string of the form 'l/N' and return the
    numerical allowable deflection in mm.

    Examples
    --------
    'l/240' with length_mm=1500  →  6.25 mm
    'l/360' with length_mm=2000  →  5.56 mm

    Parameters
    ----------
    limit_str  : str    e.g. 'l/240'  (case-insensitive)
    length_mm  : float  span / cantilever length in mm

    Returns
    -------
    float  allowable deflection in mm
    """
    # Match optional leading 'l' or 'L', a forward slash, then an integer divisor
    match = re.fullmatch(r"[lL]/(\d+)", limit_str.strip())
    if not match:
        raise ValueError(
            f"Unrecognised deflection limit format: '{limit_str}'. "
            "Expected format: 'l/<integer>', e.g. 'l/240'."
        )
    divisor = int(match.group(1))       # extract the denominator (e.g. 240)
    return length_mm / divisor          # compute allowable deflection in mm


# ---------------------------------------------------------------------------
# Section property calculations
# ---------------------------------------------------------------------------

def compute_section_properties(w: float, h: float) -> dict:
    """
    Compute geometric cross-section properties for a solid rectangle.

    Parameters
    ----------
    w : float  width  (mm)
    h : float  height (mm)

    Returns
    -------
    dict with keys:
        area_mm2       – cross-sectional area  (mm²)
        I_mm4          – second moment of area (mm⁴)
        c_mm           – distance from neutral axis to extreme fibre (mm)
        S_mm3          – section modulus = I/c  (mm³)
    """
    area = w * h                        # A = w·h  (mm²)
    I    = (w * h**3) / 12.0           # I = w·h³/12  (mm⁴)
    c    = h / 2.0                     # c = h/2  (mm), neutral axis at mid-height
    S    = I / c                       # S = I/c  (mm³)
    return {"area_mm2": area, "I_mm4": I, "c_mm": c, "S_mm3": S}


# ---------------------------------------------------------------------------
# Demand calculations
# ---------------------------------------------------------------------------

def compute_reactions(F_N: float, L_mm: float) -> dict:
    """
    Compute support reactions at the fixed (wall) end of a cantilever beam
    with a single point load F at the free end.

    Equilibrium:
        ΣFy = 0  →  R_y = F
        ΣM  = 0  →  M_wall = F · L

    Parameters
    ----------
    F_N   : float  applied point load  (N)
    L_mm  : float  cantilever length   (mm)

    Returns
    -------
    dict with keys:
        R_y_N     – vertical reaction at wall   (N)
        M_wall_Nmm– moment reaction at wall     (N·mm)
    """
    R_y     = F_N                   # vertical reaction equals applied load
    M_wall  = F_N * L_mm            # moment reaction = F × full span length
    return {"R_y_N": R_y, "M_wall_Nmm": M_wall}

# ---------------------------------------------------------------------------
# Stress and deflection demand calculations
# ---------------------------------------------------------------------------

def compute_max_flexural_stress(M_max_Nmm: float, S_mm3: float) -> float:
    """
    Compute the maximum flexural (bending) stress using the flexure formula:
        sigma_max = M_max / S

    This is the normal stress at the extreme fibre (top or bottom face) at
    the fixed end, where the bending moment is greatest.

    Parameters
    ----------
    M_max_Nmm : float  maximum bending moment  (N·mm)
    S_mm3     : float  section modulus          (mm³)

    Returns
    -------
    float  maximum bending stress (MPa = N/mm²)
    """
    return M_max_Nmm / S_mm3        # sigma = M/S  (N/mm² = MPa)


def compute_max_shear_stress(V_N: float, area_mm2: float) -> float:
    """
    Compute the maximum horizontal shear stress for a rectangular section.

    The shear stress distribution is parabolic and peaks at the neutral axis:
        tau_max = (3/2) · V / A

    The factor 3/2 is specific to rectangular cross-sections and arises from
    integrating the parabolic Q/b distribution across the depth.

    Parameters
    ----------
    V_N      : float  maximum shear force  (N)
    area_mm2 : float  cross-sectional area (mm²)

    Returns
    -------
    float  maximum shear stress (MPa = N/mm²)
    """
    return 1.5 * V_N / area_mm2     # tau = 3/2 · V/A  (N/mm² = MPa)


def compute_tip_deflection(F_N: float, L_mm: float,
                           E_MPa: float, I_mm4: float) -> float:
    """
    Compute the maximum tip deflection of a cantilever beam with a point load
    at the free end, derived from double-integration of the moment equation:

        delta_max = F · L³ / (3 · E · I)

    Boundary conditions applied: v(L)=0 (no displacement at wall),
    v'(L)=0 (no rotation at wall).

    Parameters
    ----------
    F_N   : float  applied point load      (N)
    L_mm  : float  cantilever length       (mm)
    E_MPa : float  modulus of elasticity   (MPa = N/mm²)
    I_mm4 : float  second moment of area   (mm⁴)

    Returns
    -------
    float  maximum tip deflection (mm)
    """
    return (F_N * L_mm**3) / (3.0 * E_MPa * I_mm4)  # delta = FL³/(3EI)


# ---------------------------------------------------------------------------
# Main design checks program
# ---------------------------------------------------------------------------

def main(json_filepath: str = "./scripts/initial_values.json", verbose: bool = False):
    """
    Load a cantilever beam problem from JSON, run all three checks, and
    return results. Optionally print a detailed report to the terminal.

    Parameters
    ----------
    json_filepath : str   path to the problem JSON file
    verbose       : bool  when True, print the full step-by-step report;
                          when False (default), run silently

    Returns
    -------
    demands  : list[float]  [max_flex_stress_MPa, max_shear_stress_MPa,
                             max_deflection_mm]
    passes   : list[bool]   [flex_ok, shear_ok, deflection_ok]
    all_pass : bool         True only when every individual check passes
    """

    # ------------------------------------------------------------------
    # 1. Load problem definition
    # ------------------------------------------------------------------
    prob    = load_problem(json_filepath)
    mat     = prob["material"]
    geo     = prob["geometry"]
    loading = prob["loading"]
    svc     = prob["serviceability"]

    # ------------------------------------------------------------------
    # 2. Extract and convert inputs to consistent units (N, mm, MPa)
    # ------------------------------------------------------------------
    F_b_MPa = mat["allowable_bending_stress_MPa"]   # allowable bending stress (MPa)
    F_v_MPa = mat["allowable_shear_stress_MPa"]     # allowable shear stress   (MPa)
    E_MPa   = mat["modulus_of_elasticity_GPa"] * 1_000.0  # GPa → MPa (×1000)

    w_mm    = geo["width_mm"]                       # cross-section width  (mm)
    h_mm    = geo["height_mm"]                      # cross-section height (mm)
    L_mm    = geo["length_mm"]                      # cantilever length    (mm)

    F_N     = loading["point_load_kN"] * 1_000.0   # kN → N (×1000)

    # ------------------------------------------------------------------
    # 3. Compute section properties
    # ------------------------------------------------------------------
    sec = compute_section_properties(w_mm, h_mm)

    # ------------------------------------------------------------------
    # 4. Compute support reactions
    # ------------------------------------------------------------------
    rxn = compute_reactions(F_N, L_mm)
    V_N         = rxn["R_y_N"]          # max shear = reaction = F (constant along beam)
    M_max_Nmm   = rxn["M_wall_Nmm"]     # max moment at fixed end

    # ------------------------------------------------------------------
    # 5. Compute stress and deflection demands
    # ------------------------------------------------------------------
    sigma_max_MPa = compute_max_flexural_stress(M_max_Nmm, sec["S_mm3"])
    tau_max_MPa   = compute_max_shear_stress(V_N, sec["area_mm2"])
    delta_max_mm  = compute_tip_deflection(F_N, L_mm, E_MPa, sec["I_mm4"])

    # ------------------------------------------------------------------
    # 6. Compute deflection limit from string (e.g. "l/240")
    # ------------------------------------------------------------------
    delta_allow_mm = parse_deflection_limit(svc["allowable_deflection"], L_mm)

    # ------------------------------------------------------------------
    # 7. Evaluate pass/fail for each check
    # ------------------------------------------------------------------
    flex_ok      = sigma_max_MPa <= F_b_MPa     # bending stress within limit?
    shear_ok     = tau_max_MPa   <= F_v_MPa     # shear stress within limit?
    deflect_ok   = delta_max_mm  <= delta_allow_mm  # deflection within limit?
    all_pass     = flex_ok and shear_ok and deflect_ok  # all three must pass

    # ------------------------------------------------------------------
    # 8. Print detailed report (only when verbose=True)
    # ------------------------------------------------------------------
    if verbose:
        PASS = "✅ PASS"
        FAIL = "❌ FAIL"
        div  = "─" * 58

        print(div)
        print("  CANTILEVER BEAM CHECK REPORT")
        print(div)
        print(f"  Material : {mat['name']}")
        print(f"  Section  : {w_mm} mm (w) × {h_mm} mm (h)")
        print(f"  Span     : {L_mm} mm")
        print(f"  Load     : {F_N / 1000:.2f} kN (point load at free end)")
        print()

        print("  MATERIAL PROPERTIES")
        print(f"    F_b  (allowable bending stress) = {F_b_MPa:.2f} MPa")
        print(f"    F_v  (allowable shear stress)   = {F_v_MPa:.3f} MPa")
        print(f"    E    (modulus of elasticity)     = {E_MPa / 1000:.1f} GPa  "
              f"({E_MPa:,.0f} MPa)")
        print()

        print("  CROSS-SECTION PROPERTIES")
        print(f"    A  = {sec['area_mm2']:>12,.1f} mm²")
        print(f"    I  = {sec['I_mm4']:>12,.0f} mm⁴   ({sec['I_mm4'] / 1e6:.2f} × 10⁶ mm⁴)")
        print(f"    c  = {sec['c_mm']:>12.1f} mm")
        print(f"    S  = {sec['S_mm3']:>12,.0f} mm³   ({sec['S_mm3'] / 1e3:.1f} × 10³ mm³)")
        print()

        print("  SUPPORT REACTIONS (fixed end)")
        print(f"    R_y      = {V_N / 1000:.3f} kN")
        print(f"    M_wall   = {M_max_Nmm:.3e} N·mm  =  {M_max_Nmm / 1e6:.3f} kN·m")
        print()

        print("  CHECK 1 — FLEXURAL STRESS")
        print("    sigma_max = M_max / S")
        print(f"              = {M_max_Nmm:.3e} N·mm / {sec['S_mm3']:,.0f} mm³")
        print(f"              = {sigma_max_MPa:.3f} MPa")
        print(f"    Capacity  = F_b = {F_b_MPa:.2f} MPa")
        print(f"    Ratio     = {sigma_max_MPa / F_b_MPa:.3f}  ({sigma_max_MPa / F_b_MPa * 100:.1f}% utilized)")
        print(f"    Result    → {PASS if flex_ok else FAIL}")
        print()

        print("  CHECK 2 — SHEAR STRESS")
        print("    tau_max = (3/2) · V / A")
        print(f"            = 1.5 × {V_N:.0f} N / {sec['area_mm2']:,.1f} mm²")
        print(f"            = {tau_max_MPa:.4f} MPa")
        print(f"    Capacity  = F_v = {F_v_MPa:.3f} MPa")
        print(f"    Ratio     = {tau_max_MPa / F_v_MPa:.3f}  ({tau_max_MPa / F_v_MPa * 100:.1f}% utilized)")
        print(f"    Result    → {PASS if shear_ok else FAIL}")
        print()

        print("  CHECK 3 — TIP DEFLECTION")
        print("    delta_max  = F·L³ / (3·E·I)")
        print(f"               = {F_N:.0f} × {L_mm:.0f}³ / (3 × {E_MPa:,.0f} × {sec['I_mm4']:,.0f})")
        print(f"               = {delta_max_mm:.3f} mm")
        print(f"    Limit      = {svc['allowable_deflection']} = "
              f"{L_mm:.0f} / {int(L_mm / delta_allow_mm)} = {delta_allow_mm:.3f} mm")
        print(f"    Ratio      = {delta_max_mm / delta_allow_mm:.3f}  "
              f"({delta_max_mm / delta_allow_mm * 100:.1f}% utilized)")
        print(f"    Result     → {PASS if deflect_ok else FAIL}")
        print()

        print(div)
        overall_str = "✅ ALL CHECKS PASS" if all_pass else "❌ ONE OR MORE CHECKS FAILED"
        print(f"  OVERALL: {overall_str}")
        print(div)

    # ------------------------------------------------------------------
    # 9. Return structured results
    # ------------------------------------------------------------------
    demands  = [sigma_max_MPa, tau_max_MPa, delta_max_mm]
    passes   = [flex_ok, shear_ok, deflect_ok]
    return demands, passes, all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Accept an optional positional argument for the JSON filepath and an
    # optional --verbose flag to enable printed output.
    #   python cantilever_beam.py                        # silent
    #   python cantilever_beam.py --verbose              # verbose, default JSON
    #   python cantilever_beam.py my.json --verbose      # verbose, custom JSON
    args     = sys.argv[1:]                                 # all CLI arguments
    verbose  = "--verbose" in args                          # True if flag is present
    non_flag = [a for a in args if not a.startswith("--")] # positional args only
    filepath = "./scripts/" + non_flag[0] if non_flag else "./scripts/initial_values.json"
    main(filepath, verbose=verbose)