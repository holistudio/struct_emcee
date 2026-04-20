"""
cantilever_beam.py
------------------
Performs NDS-informed strength and serviceability checks for a cantilever
beam with a point load at the free end and a solid rectangular cross-section.

Checks performed:
  1. Flexural (bending) stress  : sigma_max  <= F'_b  (adjusted allowable)
  2. Shear stress               : tau_max    <= F'_v  (adjusted allowable)
  3. Tip deflection             : delta_max  <= delta_allow  (e.g. l/240)

Loads modelled:
  - Applied tip point load F
  - Self-weight of the beam as a UDL (computed from density and cross-section)

Capacity adjustments (NDS ASD, sawn lumber):
  - F'_b = F_b · CD · CM · Ct · CF · Cr · CL
  - F'_v = F_v · CD · CM · Ct
  - E'   = E   · CM · Ct          (CD does NOT apply to stiffness)

  CF  — Size Factor          : auto-looked-up from NDS Table 4A by nominal depth
  CL  — Beam Stability Factor: computed per NDS 3.3.3 (LTB, assuming no bracing)

All internal calculations use consistent SI base units:
  Force  → N
  Length → mm
  Stress → MPa  (= N/mm²)
  E      → MPa  (converted from GPa on input)

Usage:
  python scripts/cantilever_beam.py                      # uses default initial_values.json
  python scripts/cantilever_beam.py my_problem.json      # uses a custom JSON file
  python scripts/cantilever_beam.py --verbose            # print detailed report
"""

import json
import math
import re
import sys


# ---------------------------------------------------------------------------
# NDS Table 4A — Size Factor CF for Fb
# Dimension lumber (2"–4" thick), visually graded.
# Key = nominal depth in inches; value = CF multiplier for Fb.
# Source: NDS 2018 Supplement Table 4A footnotes / NDS Section 4.3.6
# ---------------------------------------------------------------------------
CF_TABLE_FB = {
    2:  1.50,
    3:  1.50,
    4:  1.50,
    5:  1.40,
    6:  1.30,
    8:  1.20,
    10: 1.10,   
    12: 1.00,
    14: 0.90,
    16: 0.90,   # 14" and larger all use 0.90
}


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
    allowable deflection in mm.

    Parameters
    ----------
    limit_str  : str    e.g. 'l/240'  (case-insensitive)
    length_mm  : float  cantilever length in mm

    Returns
    -------
    float  allowable deflection in mm
    """
    match = re.fullmatch(r"[lL]/(\d+)", limit_str.strip())
    if not match:
        raise ValueError(
            f"Unrecognised deflection limit: '{limit_str}'. "
            "Expected 'l/<integer>', e.g. 'l/240'."
        )
    divisor = int(match.group(1))   # e.g. 240
    return length_mm / divisor      # allowable deflection in mm


# ---------------------------------------------------------------------------
# NDS adjustment factor helpers
# ---------------------------------------------------------------------------

def lookup_CF(nominal_depth_in: int) -> float:
    """
    Return the NDS size factor CF for bending (Fb) based on the nominal
    depth of the member in inches (NDS Table 4A, dimension lumber 2"–4" thick).

    For nominal depths larger than 16", the code falls back to 0.90 (the
    same value used for 14"+ in the standard).

    Parameters
    ----------
    nominal_depth_in : int  nominal depth in inches (e.g. 10 for a 4×10)

    Returns
    -------
    float  CF multiplier for Fb
    """
    # Find the largest table key that is <= the requested depth
    valid_keys = sorted(k for k in CF_TABLE_FB if k <= nominal_depth_in)
    if not valid_keys:
        # Depth smaller than any table entry — return the smallest available
        return CF_TABLE_FB[min(CF_TABLE_FB)]
    return CF_TABLE_FB[valid_keys[-1]]  # exact match or next-lower key


def compute_CL(L_u_mm: float, h_mm: float, w_mm: float,
               F_b_star_MPa: float, E_min_MPa: float,
               c: float = 0.95) -> float:
    """
    Compute the NDS Beam Stability Factor CL for a sawn lumber bending member
    with no lateral bracing along the compression edge (NDS Section 3.3.3).

    Steps
    -----
    1. Effective unbraced length:  Le = 1.84 · Lu          (cantilever, tip load)
    2. Beam slenderness ratio:     RB = sqrt(Le·h / w²)
    3. Critical buckling stress:   FbE = 1.20·Emin / RB²
    4. Stability factor:           CL via NDS interaction formula

    Parameters
    ----------
    L_u_mm      : float  unbraced length (= full cantilever span)  (mm)
    h_mm        : float  section depth                              (mm)
    w_mm        : float  section width                              (mm)
    F_b_star_MPa: float  F_b* = Fb with all factors except CL applied (MPa)
    E_min_MPa   : float  lower-bound modulus for stability          (MPa)
    c           : float  NDS constant = 0.95 for sawn lumber

    Returns
    -------
    float  CL beam stability factor (0 < CL <= 1)
    """
    # Step 1: effective unbraced length for cantilever with tip point load
    # NDS Appendix G: Le/Lu = 1.84 for this loading and support condition
    Le_mm = 1.84 * L_u_mm              # Le = 1.84·Lu  (mm)

    # Step 2: beam slenderness ratio (dimensionless)
    # RB = sqrt(Le·h / w²) — deeper/longer → higher RB → more buckling risk
    RB = math.sqrt((Le_mm * h_mm) / (w_mm ** 2))

    # NDS limit: RB must not exceed 50
    if RB > 50:
        raise ValueError(
            f"Beam slenderness ratio RB = {RB:.2f} exceeds NDS limit of 50. "
            "Add lateral bracing or choose a wider/shorter section."
        )

    # Step 3: critical elastic buckling stress for beams
    # Analogous to Euler column buckling; uses Emin (conservative lower bound)
    F_bE_MPa = (1.20 * E_min_MPa) / (RB ** 2)

    # Step 4: beam stability factor — same interaction form as column CP
    alpha = F_bE_MPa / F_b_star_MPa    # ratio of buckling stress to adjusted Fb*
    term  = (1.0 + alpha) / (2.0 * c)  # intermediate term
    CL    = term - math.sqrt(term ** 2 - alpha / c)

    return CL, RB, Le_mm, F_bE_MPa     # return intermediates for reporting


# ---------------------------------------------------------------------------
# Section property calculations
# ---------------------------------------------------------------------------

def compute_section_properties(w: float, h: float) -> dict:
    """
    Compute cross-section properties for a solid rectangle.

    Parameters
    ----------
    w : float  width  (mm)
    h : float  height (mm)

    Returns
    -------
    dict  area_mm2, I_mm4, c_mm, S_mm3
    """
    area = w * h                    # A = w·h            (mm²)
    I    = (w * h ** 3) / 12.0     # I = w·h³/12        (mm⁴)
    c    = h / 2.0                  # c = h/2            (mm)
    S    = I / c                    # S = I/c            (mm³)
    return {"area_mm2": area, "I_mm4": I, "c_mm": c, "S_mm3": S}


# ---------------------------------------------------------------------------
# Demand calculations
# ---------------------------------------------------------------------------

def compute_self_weight(density_kg_m3: float, area_mm2: float) -> float:
    """
    Compute the self-weight of the beam as a uniformly distributed load (UDL).

    w_sw = rho · A · g

    Parameters
    ----------
    density_kg_m3 : float  material density     (kg/m³)
    area_mm2      : float  cross-sectional area (mm²)

    Returns
    -------
    float  self-weight UDL in N/mm
    """
    area_m2      = area_mm2 * 1e-6          # mm² → m²
    w_sw_N_per_m = density_kg_m3 * area_m2 * 9.81  # N/m
    return w_sw_N_per_m / 1_000.0           # N/m → N/mm


def compute_reactions(F_N: float, w_sw_N_mm: float, L_mm: float) -> dict:
    """
    Compute fixed-end support reactions for a cantilever with a tip point
    load F and a UDL w_sw along the full length.

    ΣFy = 0  →  R_y = F + w_sw·L
    ΣM  = 0  →  M_wall = F·L + w_sw·L²/2

    Parameters
    ----------
    F_N       : float  tip point load  (N)
    w_sw_N_mm : float  self-weight UDL (N/mm)
    L_mm      : float  span length     (mm)

    Returns
    -------
    dict  R_y_N, M_wall_Nmm, V_max_N (= R_y at wall), M_max_Nmm (= M_wall)
    """
    R_y    = F_N + w_sw_N_mm * L_mm                    # vertical reaction   (N)
    M_wall = F_N * L_mm + (w_sw_N_mm * L_mm ** 2) / 2.0  # moment reaction (N·mm)
    return {
        "R_y_N":       R_y,
        "M_wall_Nmm":  M_wall,
        "V_max_N":     R_y,        # max shear occurs at the wall
        "M_max_Nmm":   M_wall,     # max moment occurs at the wall
    }


# ---------------------------------------------------------------------------
# Stress and deflection demand calculations
# ---------------------------------------------------------------------------

def compute_max_flexural_stress(M_max_Nmm: float, S_mm3: float) -> float:
    """
    Maximum bending stress via the flexure formula:  sigma = M / S

    Parameters
    ----------
    M_max_Nmm : float  maximum bending moment (N·mm)
    S_mm3     : float  section modulus        (mm³)

    Returns
    -------
    float  sigma_max (MPa)
    """
    return M_max_Nmm / S_mm3


def compute_max_shear_stress(V_max_N: float, area_mm2: float) -> float:
    """
    Maximum shear stress for a rectangular section (parabolic distribution):
        tau_max = (3/2) · V / A

    Parameters
    ----------
    V_max_N  : float  maximum shear force  (N)
    area_mm2 : float  cross-sectional area (mm²)

    Returns
    -------
    float  tau_max (MPa)
    """
    return 1.5 * V_max_N / area_mm2


def compute_tip_deflection(F_N: float, w_sw_N_mm: float,
                           L_mm: float, E_MPa: float, I_mm4: float) -> dict:
    """
    Compute tip deflection with superposition of point load and UDL components.

    Point load:   delta_F  = F·L³ / (3·E·I)
    UDL (self-wt):delta_sw = w·L⁴ / (8·E·I)
    Total:        delta    = delta_F + delta_sw

    Parameters
    ----------
    F_N       : float  tip point load       (N)
    w_sw_N_mm : float  self-weight UDL      (N/mm)
    L_mm      : float  cantilever length    (mm)
    E_MPa     : float  modulus of elasticity(MPa)
    I_mm4     : float  second moment of area(mm⁴)

    Returns
    -------
    dict  delta_F_mm, delta_sw_mm, delta_total_mm
    """
    EI         = E_MPa * I_mm4                          # flexural rigidity (N·mm²)
    delta_F    = (F_N * L_mm ** 3) / (3.0 * EI)        # point load contribution (mm)
    delta_sw   = (w_sw_N_mm * L_mm ** 4) / (8.0 * EI)  # UDL contribution       (mm)
    delta_total = delta_F + delta_sw                    # total tip deflection   (mm)
    return {
        "delta_F_mm":     delta_F,
        "delta_sw_mm":    delta_sw,
        "delta_total_mm": delta_total,
    }


# ---------------------------------------------------------------------------
# Main design checks program
# ---------------------------------------------------------------------------

def main(json_filepath: str = "./scripts/initial_values.json", verbose: bool = False):
    """
    Load a cantilever beam problem from JSON, apply NDS adjustment factors
    (including auto-computed CF and CL), run all three checks, and return results.

    Parameters
    ----------
    json_filepath : str   path to the problem JSON file
    verbose       : bool  print detailed report when True

    Returns
    -------
    demands  : list[float]  [sigma_max_MPa, tau_max_MPa, delta_total_mm]
    passes   : list[bool]   [flex_ok, shear_ok, deflect_ok]
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
    adj     = prob["adjustment_factors"]

    # ------------------------------------------------------------------
    # 2. Extract and convert inputs → consistent units (N, mm, MPa)
    # ------------------------------------------------------------------
    # Reference (unadjusted) material properties
    F_b_MPa     = mat["reference_bending_stress_MPa"]          # (MPa)
    F_v_MPa     = mat["reference_shear_stress_MPa"]            # (MPa)
    E_MPa       = mat["modulus_of_elasticity_GPa"] * 1_000.0   # GPa → MPa
    E_min_MPa   = mat["modulus_of_elasticity_min_MPa"]         # already MPa
    density     = mat["density_kg_per_m3"]                     # (kg/m³)

    # Geometry — nominal (for CF lookup) and actual (for calculations)
    nom_depth_in = geo["nominal_depth_in"]          # nominal depth  (inches, integer)
    w_mm         = geo["actual_width_mm"]           # actual width   (mm)
    h_mm         = geo["actual_height_mm"]          # actual height  (mm)
    L_mm         = geo["length_mm"]                 # span           (mm)

    # Applied load
    F_N = loading["point_load_kN"] * 1_000.0       # kN → N

    # User-specified adjustment factors
    CD = adj["CD"]["value"]     # load duration factor
    CM = adj["CM"]["value"]     # wet service factor
    Ct = adj["Ct"]["value"]     # temperature factor
    Cr = adj["Cr"]["value"]     # repetitive member factor

    # ------------------------------------------------------------------
    # 3. Auto-compute CF from nominal depth lookup table
    # ------------------------------------------------------------------
    CF = lookup_CF(nom_depth_in)    # size factor for Fb (NDS Table 4A)

    # ------------------------------------------------------------------
    # 4. Compute section properties
    # ------------------------------------------------------------------
    sec = compute_section_properties(w_mm, h_mm)

    # ------------------------------------------------------------------
    # 5. Compute self-weight UDL
    # ------------------------------------------------------------------
    w_sw = compute_self_weight(density, sec["area_mm2"])    # N/mm

    # ------------------------------------------------------------------
    # 6. Compute support reactions (include self-weight)
    # ------------------------------------------------------------------
    rxn       = compute_reactions(F_N, w_sw, L_mm)
    V_max_N   = rxn["V_max_N"]      # max shear at wall (N)
    M_max_Nmm = rxn["M_max_Nmm"]    # max moment at wall (N·mm)

    # ------------------------------------------------------------------
    # 7. Compute stress and deflection demands
    # ------------------------------------------------------------------
    sigma_max_MPa = compute_max_flexural_stress(M_max_Nmm, sec["S_mm3"])
    tau_max_MPa   = compute_max_shear_stress(V_max_N, sec["area_mm2"])

    defl = compute_tip_deflection(F_N, w_sw, L_mm, E_MPa, sec["I_mm4"])
    delta_total_mm = defl["delta_total_mm"]

    # ------------------------------------------------------------------
    # 8. Compute adjusted design values
    #    F'_b = Fb · CD · CM · Ct · CF · Cr · CL
    #    F'_v = Fv · CD · CM · Ct
    #    E'   = E  · CM · Ct          (CD never applies to modulus)
    # ------------------------------------------------------------------

    # F_b* = Fb with all factors EXCEPT CL (needed to compute CL itself)
    F_b_star = F_b_MPa * CD * CM * Ct * CF * Cr

    # CL: beam stability factor (LTB), computed from F_b* and Emin
    CL, RB, Le_mm, F_bE_MPa = compute_CL(
        L_u_mm       = L_mm,
        h_mm         = h_mm,
        w_mm         = w_mm,
        F_b_star_MPa = F_b_star,
        E_min_MPa    = E_min_MPa,
    )

    # Final adjusted allowable values
    F_b_prime = F_b_star * CL          # adjusted allowable bending stress (MPa)
    F_v_prime = F_v_MPa * CD * CM * Ct # adjusted allowable shear stress   (MPa)
    E_prime   = E_MPa * CM * Ct        # adjusted modulus for deflection    (MPa)

    # Deflection allowable
    delta_allow_mm = parse_deflection_limit(svc["allowable_deflection"], L_mm)

    # ------------------------------------------------------------------
    # 9. Evaluate pass/fail
    # ------------------------------------------------------------------
    flex_ok    = sigma_max_MPa  <= F_b_prime      # bending check
    shear_ok   = tau_max_MPa    <= F_v_prime      # shear check
    deflect_ok = delta_total_mm <= delta_allow_mm  # serviceability check
    all_pass   = flex_ok and shear_ok and deflect_ok

    # ------------------------------------------------------------------
    # 10. Print report (only when verbose=True)
    # ------------------------------------------------------------------
    if verbose:
        PASS = "✅ PASS"
        FAIL = "❌ FAIL"
        div  = "─" * 64
        div2 = "·" * 64

        print(div)
        print("  CANTILEVER BEAM CHECK REPORT  (NDS ASD — Sawn Lumber)")
        print(div)
        print(f"  Material  : {mat['name']}")
        print(f"  Section   : {w_mm} mm (w) × {h_mm} mm (h)  "
              f"[nominal {geo['nominal_width_in']}×{nom_depth_in}]")
        print(f"  Span      : {L_mm:,.0f} mm")
        print(f"  Tip load  : {F_N / 1_000:.2f} kN")
        print()

        # -- Reference material properties ---------------------------------
        print("  REFERENCE DESIGN VALUES  (NDS Table 4A, dry, normal duration)")
        print(f"    F_b  = {F_b_MPa:.2f} MPa   (bending)")
        print(f"    F_v  = {F_v_MPa:.3f} MPa  (shear)")
        print(f"    E    = {E_MPa / 1_000:.1f} GPa  ({E_MPa:,.0f} MPa)")
        print(f"    Emin = {E_min_MPa:,.0f} MPa  (stability modulus)")
        print()

        # -- Adjustment factors --------------------------------------------
        print("  NDS ADJUSTMENT FACTORS")
        print(f"    CD  (load duration)          = {CD:.2f}  ← user-specified")
        print(f"    CM  (wet service)            = {CM:.2f}  ← user-specified")
        print(f"    Ct  (temperature)            = {Ct:.2f}  ← user-specified")
        print(f"    CF  (size factor, auto)      = {CF:.2f}  ← from nominal {nom_depth_in}\" depth lookup")
        print(f"    Cr  (repetitive member)      = {Cr:.2f}  ← user-specified")
        print()

        # -- LTB sub-calculation -------------------------------------------
        print("  LATERAL-TORSIONAL BUCKLING  (NDS 3.3.3, no bracing)")
        print(f"    Unbraced length  Lu  = {L_mm:,.0f} mm")
        print(f"    Effective length Le  = 1.84 × {L_mm:,.0f} = {Le_mm:,.0f} mm")
        print(f"    Slenderness      RB  = √(Le·h/w²) = √({Le_mm:,.0f}×{h_mm}/{w_mm}²)")
        print(f"                         = {RB:.3f}  (NDS limit ≤ 50  ✓)")
        print(f"    F_b*             = Fb·CD·CM·Ct·CF·Cr = {F_b_star:.3f} MPa")
        print(f"    FbE  = 1.20·Emin / RB²  = 1.20×{E_min_MPa:,.0f} / {RB:.3f}²")
        print(f"         = {F_bE_MPa:.2f} MPa")
        print(f"    α    = FbE / F_b* = {F_bE_MPa:.2f} / {F_b_star:.3f} = {F_bE_MPa / F_b_star:.3f}")
        print(f"    CL   = {CL:.4f}")
        print()

        # -- Adjusted capacities -------------------------------------------
        print("  ADJUSTED ALLOWABLE DESIGN VALUES")
        print("    F'_b = Fb·CD·CM·Ct·CF·Cr·CL")
        print(f"         = {F_b_MPa:.2f}×{CD}×{CM}×{Ct}×{CF:.2f}×{Cr:.2f}×{CL:.4f}")
        print(f"         = {F_b_prime:.3f} MPa")
        print("    F'_v = Fv·CD·CM·Ct")
        print(f"         = {F_v_MPa:.3f}×{CD}×{CM}×{Ct}")
        print(f"         = {F_v_prime:.4f} MPa")
        print(f"    E'   = E·CM·Ct  = {E_MPa:,.0f}×{CM}×{Ct} = {E_prime:,.0f} MPa")
        print()

        # -- Section properties --------------------------------------------
        print("  CROSS-SECTION PROPERTIES")
        print(f"    A  = {sec['area_mm2']:>12,.1f} mm²")
        print(f"    I  = {sec['I_mm4']:>12,.0f} mm⁴  ({sec['I_mm4'] / 1e6:.3f} × 10⁶ mm⁴)")
        print(f"    c  = {sec['c_mm']:>12.1f} mm")
        print(f"    S  = {sec['S_mm3']:>12,.0f} mm³  ({sec['S_mm3'] / 1e3:.1f} × 10³ mm³)")
        print()

        # -- Self-weight ---------------------------------------------------
        print("  SELF-WEIGHT")
        print(f"    w_sw = ρ·A·g = {density:.0f}×{sec['area_mm2']:.1f}×10⁻⁶×9.81")
        print(f"         = {w_sw * 1_000:.4f} N/m  =  {w_sw:.6f} N/mm")
        print()

        # -- Reactions -----------------------------------------------------
        print("  SUPPORT REACTIONS  (fixed end, point load + self-weight)")
        print(f"    R_y     = F + w_sw·L = {F_N:.0f} + {w_sw:.4f}×{L_mm:.0f}")
        print(f"            = {rxn['R_y_N']:.1f} N  =  {rxn['R_y_N'] / 1000:.4f} kN ↑")
        print("    M_wall  = F·L + w_sw·L²/2")
        print(f"            = {F_N:.0f}×{L_mm:.0f} + {w_sw:.4f}×{L_mm:.0f}²/2")
        print(f"            = {M_max_Nmm:.3e} N·mm  =  {M_max_Nmm / 1e6:.4f} kN·m")
        print()

        # -- Check 1: Bending ----------------------------------------------
        print(div2)
        print("  CHECK 1 — FLEXURAL (BENDING) STRESS")
        print("    sigma_max = M_max / S")
        print(f"              = {M_max_Nmm:.4e} / {sec['S_mm3']:,.0f}")
        print(f"              = {sigma_max_MPa:.4f} MPa  (demand)")
        print(f"    F'_b      = {F_b_prime:.4f} MPa  (adjusted capacity)")
        print(f"    Ratio     = {sigma_max_MPa / F_b_prime:.4f}  "
              f"({sigma_max_MPa / F_b_prime * 100:.1f}% utilized)")
        print(f"    Result    → {PASS if flex_ok else FAIL}")
        print()

        # -- Check 2: Shear ------------------------------------------------
        print("  CHECK 2 — SHEAR STRESS")
        print("    tau_max = (3/2)·V / A")
        print(f"            = 1.5 × {V_max_N:.1f} / {sec['area_mm2']:,.1f}")
        print(f"            = {tau_max_MPa:.4f} MPa  (demand)")
        print(f"    F'_v    = {F_v_prime:.4f} MPa  (adjusted capacity)")
        print(f"    Ratio   = {tau_max_MPa / F_v_prime:.4f}  "
              f"({tau_max_MPa / F_v_prime * 100:.1f}% utilized)")
        print(f"    Result  → {PASS if shear_ok else FAIL}")
        print()

        # -- Check 3: Deflection -------------------------------------------
        print("  CHECK 3 — TIP DEFLECTION  (superposition)")
        print("    δ_F   = F·L³ / (3·E'·I)")
        print(f"          = {F_N:.0f}×{L_mm:.0f}³ / (3×{E_prime:,.0f}×{sec['I_mm4']:,.0f})")
        print(f"          = {defl['delta_F_mm']:.4f} mm")
        print("    δ_sw  = w_sw·L⁴ / (8·E'·I)")
        print(f"          = {w_sw:.6f}×{L_mm:.0f}⁴ / (8×{E_prime:,.0f}×{sec['I_mm4']:,.0f})")
        print(f"          = {defl['delta_sw_mm']:.4f} mm")
        print(f"    δ_tot = {defl['delta_F_mm']:.4f} + {defl['delta_sw_mm']:.4f}")
        print(f"          = {delta_total_mm:.4f} mm  (demand)")
        print(f"    Limit = {svc['allowable_deflection']} = "
              f"{L_mm:.0f} / {int(L_mm / delta_allow_mm)} = {delta_allow_mm:.4f} mm")
        print(f"    Ratio = {delta_total_mm / delta_allow_mm:.4f}  "
              f"({delta_total_mm / delta_allow_mm * 100:.1f}% utilized)")
        print(f"    Result → {PASS if deflect_ok else FAIL}")
        print()

        # -- Summary -------------------------------------------------------
        print(div)
        print("  SUMMARY OF CHECKS")
        print(f"  {'Check':<30} {'Demand':>12} {'Capacity':>12} {'Ratio':>8}  Status")
        print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*8}  {'─'*6}")
        print(f"  {'Bending stress (MPa)':<30} {sigma_max_MPa:>12.4f} {F_b_prime:>12.4f} "
              f"{sigma_max_MPa / F_b_prime:>8.4f}  {PASS if flex_ok else FAIL}")
        print(f"  {'Shear stress (MPa)':<30} {tau_max_MPa:>12.4f} {F_v_prime:>12.4f} "
              f"{tau_max_MPa / F_v_prime:>8.4f}  {PASS if shear_ok else FAIL}")
        print(f"  {'Tip deflection (mm)':<30} {delta_total_mm:>12.4f} {delta_allow_mm:>12.4f} "
              f"{delta_total_mm / delta_allow_mm:>8.4f}  {PASS if deflect_ok else FAIL}")
        print(div)
        overall_str = "✅ ALL CHECKS PASS" if all_pass else "❌ ONE OR MORE CHECKS FAILED"
        print(f"  OVERALL: {overall_str}")
        print(div)

    # ------------------------------------------------------------------
    # 11. Return structured results
    # ------------------------------------------------------------------
    demands  = [sigma_max_MPa, tau_max_MPa, delta_total_mm]
    passes   = [flex_ok, shear_ok, deflect_ok]
    return demands, passes, all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # CLI usage:
    #   python cantilever_beam.py                        # silent, default JSON
    #   python cantilever_beam.py --verbose              # verbose, default JSON
    #   python cantilever_beam.py my.json --verbose      # verbose, custom JSON
    args     = sys.argv[1:]
    verbose  = "--verbose" in args
    non_flag = [a for a in args if not a.startswith("--")]
    filepath = "./scripts/" + non_flag[0] if non_flag else "./scripts/initial_values.json"
    main(filepath, verbose=verbose)