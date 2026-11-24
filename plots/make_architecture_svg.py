#!/usr/bin/env python3
"""Generate docs/figures/architecture.svg for ship-nf.

Pure Python standard library; outputs a standalone vector SVG matching the
39th SHiP Collaboration Meeting slide grammar.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "architecture.svg"

INK = "#1b232c"
SLATE = "#2d3945"
MUTED = "#556270"
RULE = "#d2dae2"
BG = "#ffffff"
PANEL_BG = "#f7f9fb"

CYAN_FILL = "#d8f3ff"
CYAN_STROKE = "#22a7e0"
TEAL = "#0f6e7c"
TERRA = "#b04b0e"
DASH_GREY = "#94a0ac"

LADDER_COLS = [
    ("#0d3659", "#ffffff", "#d5e7f5"),
    ("#18507d", "#ffffff", "#d5e7f5"),
    ("#2a6ea3", "#ffffff", "#e5f2fc"),
    ("#468bc0", "#ffffff", "#eef6fc"),
    ("#6aa8d6", "#13293d", "#1e3a52"),
    ("#94c5e6", "#13293d", "#27455e"),
    ("#c7e3f5", "#13293d", "#2c4c66"),
]

FONT_MONO = "'DejaVu Sans Mono', 'Liberation Mono', Consolas, monospace"
FONT_SANS = "'DejaVu Sans', 'Liberation Sans', Helvetica, Arial, sans-serif"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def sharp_box(
    x, y, w, h, title, lines, fill="#ffffff", stroke=SLATE, title_color=INK,
    notch_color=None, dashed=False, title_size=12.4, body_size=10.4,
    mono_title=True, mono_lines=None, line_step=16.5,
):
    mono_lines = set(mono_lines or [])
    dash_attr = ' stroke-dasharray="5,4"' if dashed else ""
    t_font = FONT_MONO if mono_title else FONT_SANS
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.35"{dash_attr}/>',
    ]
    if notch_color:
        parts.append(f'<rect x="{x + w - 8}" y="{y}" width="8" height="8" fill="{notch_color}"/>')
    parts.append(
        f'<text x="{x + 12}" y="{y + 22}" font-family="{t_font}" font-size="{title_size}" '
        f'font-weight="700" fill="{title_color}">{esc(title)}</text>'
    )
    cy = y + 40
    for idx, line in enumerate(lines):
        is_mono = idx in mono_lines
        ff = FONT_MONO if is_mono else FONT_SANS
        fs = body_size - 0.5 if is_mono else body_size
        col = SLATE if is_mono else MUTED
        parts.append(
            f'<text x="{x + 12}" y="{cy}" font-family="{ff}" font-size="{fs}" fill="{col}">{esc(line)}</text>'
        )
        cy += line_step
    return "\n  ".join(parts)


def arrow(x1, y1, x2, y2, color=SLATE, width=1.6, dashed=False, marker="arr"):
    dash_attr = ' stroke-dasharray="5,4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}"{dash_attr} marker-end="url(#{marker})"/>'
    )


def path_arrow(d, color=SLATE, width=1.6, dashed=False, marker="arr"):
    dash_attr = ' stroke-dasharray="5,4"' if dashed else ""
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
        f'{dash_attr} marker-end="url(#{marker})"/>'
    )


def frame_header(W, H, tag_left, tag_right, section_tag, title):
    return [
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="{BG}"/>',
        f'<rect x="10" y="10" width="{W - 20}" height="{H - 20}" fill="{BG}" stroke="{INK}" stroke-width="1.4"/>',
        f'<rect x="26" y="22" width="7" height="7" fill="{SLATE}"/>',
        f'<text x="39" y="29" font-family="{FONT_MONO}" font-size="10.2" fill="{MUTED}">{esc(tag_left)}</text>',
        f'<text x="{W - 26}" y="29" text-anchor="end" font-family="{FONT_MONO}" font-size="10.2" fill="{MUTED}">{esc(tag_right)}</text>',
        f'<line x1="10" y1="39" x2="{W - 10}" y2="39" stroke="{RULE}" stroke-width="1"/>',
        f'<circle cx="31" cy="56" r="3" fill="{TEAL}"/>',
        f'<text x="40" y="60" font-family="{FONT_MONO}" font-size="10.5" font-weight="700" fill="{TEAL}">{esc(section_tag)}</text>',
        f'<text x="26" y="86" font-family="{FONT_MONO}" font-size="19" font-weight="700" fill="{INK}">{esc(title)}</text>',
    ]


def build_svg() -> str:
    W, H = 1220, 720
    elems = frame_header(
        W, H,
        "SHiP · muon DIS · Repo A (ship-nf)",
        "FairShip 26.10 (8a1cc53) · ship-exchange 0.3",
        "ARCHITECTURE  /  closed-loop generative proposal & utility surrogate",
        "The Normalizing Flow proposes P2 states; FairShip judges; U(s) guides",
    )

    elems.append(f'<rect x="26" y="102" width="848" height="392" fill="{PANEL_BG}" stroke="{RULE}" stroke-width="1.2"/>')
    elems.append(f'<text x="38" y="121" font-family="{FONT_MONO}" font-size="10.5" font-weight="700" fill="{TEAL}">REPO A · ship-nf  (PyTorch, zero ROOT/FairShip imports)</text>')

    elems.append(f'<rect x="926" y="102" width="268" height="392" fill="{PANEL_BG}" stroke="{RULE}" stroke-width="1.2"/>')
    elems.append(f'<text x="938" y="121" font-family="{FONT_MONO}" font-size="10.5" font-weight="700" fill="{SLATE}">EXTERNAL ORACLE · Repo B</text>')

    elems.append(sharp_box(
        40, 134, 246, 118,
        "1. Source States (P2)",
        [
            "TrainingProblem (states.parquet)",
            "s = (x, y, z, t, px, py, pz, pdg)",
            "φ(s) = (x, y, tx, ty, log |p|)",
            "exchange.py :: load_problem",
            "loop.py :: features",
        ],
        fill="#ffffff", stroke=SLATE, notch_color=SLATE, mono_lines={3, 4},
    ))

    elems.append(sharp_box(
        322, 134, 250, 118,
        "2. Utility Tilt & Resample",
        [
            "π_i ∝ h(U_i / max U)",
            "h(u) = [δ + (1 − δ) u]^α",
            "Jittered resample (0.05 σ)",
            "Optional untilted P0 copy (R4)",
            "loop.py :: h, propose",
        ],
        fill=CYAN_FILL, stroke=CYAN_STROKE, title_color=INK, notch_color=CYAN_STROKE, mono_lines={4},
    ))

    elems.append(sharp_box(
        608, 134, 250, 118,
        "3. Normalizing Flow q_θ",
        [
            "6-layer coupling flow in φ(s)",
            "main: RealNVP affine, N(0, I)",
            "exp/nsf-arm: Student-t₅ / RQS",
            "Exact log q_θ(x) + std. Jacobian",
            "flow.py :: Flow, fit",
        ],
        fill=CYAN_FILL, stroke=CYAN_STROKE, title_color=INK, notch_color=CYAN_STROKE, mono_lines={4},
    ))

    elems.append(sharp_box(
        40, 294, 200, 144,
        "Offline Studies",
        [
            "Tail extrapolation vs d(x)",
            "Self-consumption (MAD)",
            "Planted target (GOAL/DECOY)",
            "extrapolation.py",
            "selfconsume.py",
            "tail_demo.py, planted.py",
        ],
        fill="#ffffff", stroke=DASH_GREY, title_color=SLATE, dashed=True, mono_lines={3, 4, 5},
    ))

    elems.append(sharp_box(
        262, 294, 304, 144,
        "5. Quick Classifier U(s)",
        [
            "U(s) ≈ P(≥ 1 good track | s)",
            "Binomial logistic on quadratic φ(s)",
            "Newton–Raphson + L2 ridge (UBT excl.)",
            "Laplace cov. H⁻¹ → Thompson (planted)",
            "loop.py :: observed, fit_u, utility",
            "planted.py :: laplace_cov, trajectory",
        ],
        fill=CYAN_FILL, stroke=CYAN_STROKE, title_color=INK, notch_color=CYAN_STROKE, mono_lines={4, 5},
    ))

    elems.append(sharp_box(
        598, 294, 260, 144,
        "4. Filter & Slot Selection",
        [
            "Support: on P2 plane, p ≥ 2 GeV/c",
            "(R1 audit: p ≤ 400 GeV/c beam)",
            "Search slots: top-m by U(s)",
            "Estimate slots (ε): random draw",
            "  from untilted P0 flow + log q",
            "exchange.py :: write_model_delivery",
        ],
        fill="#ffffff", stroke=SLATE, notch_color=SLATE, mono_lines={5},
    ))

    elems.append(sharp_box(
        940, 134, 240, 128,
        "ship-fs-pipeline",
        [
            "Validates schema & SHA-256",
            "Injects MCTrack[0] at P2",
            "Context: P2_ONLY_DIAGNOSTIC",
            "No upstream / companion hits",
            "evaluate.py, states_root.py",
        ],
        fill="#ffffff", stroke=SLATE, notch_color=SLATE, mono_lines={4},
    ))

    elems.append(sharp_box(
        940, 294, 240, 144,
        "FairShip 26.10 (8a1cc53)",
        [
            "prepareEvents.py (Pythia 6 DIS)",
            "run_simScript --NewMuDIS (G4)",
            "ShipReco.py (GenFit tracks)",
            "Joins DIS on muDIS_muEntry",
            "Repeating DIS: y passes of n",
            "Endpoint: r_demo_v0 (≥1 goodTrk)",
        ],
        fill="#ffffff", stroke=SLATE, notch_color=SLATE, title_size=11.6, mono_lines={0, 1, 2},
    ))

    elems.append(arrow(286, 193, 320, 193, color=SLATE))
    elems.append(arrow(572, 193, 606, 193, color=SLATE))
    elems.append(arrow(733, 252, 733, 292, color=SLATE))

    elems.append(arrow(447, 294, 447, 254, color=CYAN_STROKE, width=1.9, marker="arr-cyan"))
    elems.append(f'<text x="455" y="278" font-family="{FONT_MONO}" font-size="10.2" font-weight="700" fill="{TEAL}">guides</text>')
    elems.append(arrow(566, 366, 596, 366, color=CYAN_STROKE, width=1.9, marker="arr-cyan"))
    elems.append(f'<text x="581" y="358" text-anchor="middle" font-family="{FONT_MONO}" font-size="9.5" font-weight="700" fill="{TEAL}">U(s)</text>')

    elems.append(path_arrow("M 858 342 L 900 342 L 900 198 L 938 198", color=SLATE, width=1.8))
    elems.append(
        f'<rect x="862" y="248" width="76" height="32" fill="{BG}" stroke="{SLATE}" stroke-width="1"/>'
        f'<text x="900" y="261" text-anchor="middle" font-family="{FONT_MONO}" font-size="9.2" font-weight="700" fill="{INK}">Proposal</text>'
        f'<text x="900" y="274" text-anchor="middle" font-family="{FONT_MONO}" font-size="9.2" font-weight="700" fill="{INK}">Delivery</text>'
    )

    elems.append(arrow(1060, 262, 1060, 292, color=SLATE))

    elems.append(path_arrow("M 1060 438 L 1060 470 L 414 470 L 414 440", color=SLATE, width=1.8))
    elems.append(
        f'<rect x="530" y="456" width="448" height="26" fill="{BG}" stroke="{SLATE}" stroke-width="1"/>'
        f'<text x="754" y="473" text-anchor="middle" font-family="{FONT_MONO}" font-size="10.0" font-weight="700" fill="{INK}">'
        f'EvaluationReport (0.3 Parquet): (proposal_id, category, n_dis, n_r)</text>'
    )

    elems.append(f'<rect x="26" y="510" width="528" height="184" fill="{PANEL_BG}" stroke="{RULE}" stroke-width="1.2"/>')
    elems.append(f'<text x="40" y="531" font-family="{FONT_MONO}" font-size="11.2" font-weight="700" fill="{INK}">STATISTICAL &amp; TAIL DIAGNOSTICS (diagnostics.py)</text>')
    diag_items = [
        ("Kish ESS &amp; PSIS k̂:", "weight degeneracy &amp; tail reliability (Vehtari et al. 2024)"),
        ("k-NN Coverage &amp; Mass:", "tail support vs held-out real muons (Naeem et al. 2020)"),
        ("Sliced W₁ &amp; 2-Sample:", "5D error vs d(x) &amp; real-vs-real floor (Bonneel et al. 2015)"),
        ("Defensive Mixture (R4):", "q = ε·p₀ + (1−ε)·q_tilted, w ≤ 1/ε (Hesterberg 1995)"),
        ("Self-Consumption (R3):", "real P₀ anchor mitigates collapse (Alemohammad et al. 2024)"),
    ]
    dy = 554
    for bold, rest in diag_items:
        elems.append(
            f'<text x="40" y="{dy}" font-family="{FONT_SANS}" font-size="10.4" fill="{INK}">'
            f'<tspan font-weight="700">{bold}</tspan> <tspan fill="{MUTED}">{rest}</tspan></text>'
        )
        dy += 23
    elems.append(
        f'<text x="40" y="678" font-family="{FONT_MONO}" font-size="9.5" fill="{TERRA}">'
        f'NOT YET MEASURED: useful candidates per CPU-hour vs unguided FairShip baseline.</text>'
    )

    elems.append(f'<rect x="570" y="510" width="624" height="184" fill="{PANEL_BG}" stroke="{RULE}" stroke-width="1.2"/>')
    elems.append(f'<text x="584" y="531" font-family="{FONT_MONO}" font-size="11.2" font-weight="700" fill="{INK}">THE 7-STAGE CUT LADDER (climb one rung at a time)</text>')
    elems.append(f'<text x="584" y="548" font-family="{FONT_SANS}" font-size="10.3" fill="{MUTED}">U_k fitted on rung k guides proposals toward rung k+1 (zero final-selection positives in current P2 pool)</text>')

    steps = [
        ("1", "Eligible", "DIS", ""),
        ("2", "≥1 good", "track", "NOW"),
        ("3", "2-track", "vertex", ""),
        ("4", "Exact 1", "cand.", ""),
        ("5", "Vertex", "geom/IP", ""),
        ("6", "Daught.", "qual/p", ""),
        ("7", "Veto", "UBT/SBT*", "OPEN"),
    ]
    sx0 = 584
    sw = 85
    base_y = 666
    for i, (num, l1, l2, badge) in enumerate(steps):
        bg_c, fg_c, sub_c = LADDER_COLS[i]
        bh = 36 + i * 11
        x = sx0 + i * sw
        y = base_y - bh
        stroke_c = TERRA if badge == "NOW" else "#ffffff"
        stroke_w = 2.0 if badge == "NOW" else 0.8
        elems.append(f'<rect x="{x}" y="{y}" width="{sw - 2}" height="{bh}" fill="{bg_c}" stroke="{stroke_c}" stroke-width="{stroke_w}"/>')
        elems.append(f'<text x="{x + 6}" y="{y + 13}" font-family="{FONT_MONO}" font-size="9.0" font-weight="700" fill="{fg_c}">{num}. {esc(l1)}</text>')
        elems.append(f'<text x="{x + 6}" y="{y + 25}" font-family="{FONT_MONO}" font-size="9.0" fill="{sub_c}">{esc(l2)}</text>')
        if badge == "NOW":
            elems.append(f'<rect x="{x + 2}" y="{y - 16}" width="{sw - 6}" height="14" fill="{TERRA}"/>')
            elems.append(f'<text x="{x + sw/2 - 1:.1f}" y="{y - 6}" text-anchor="middle" font-family="{FONT_MONO}" font-size="8.4" font-weight="700" fill="#ffffff">r_demo_v0</text>')
        elif badge == "OPEN":
            elems.append(f'<text x="{x + sw/2 - 1:.1f}" y="{y + 42}" text-anchor="middle" font-family="{FONT_MONO}" font-size="8.3" font-weight="700" fill="{TERRA}">NA (P2)</text>')

    elems.append(
        f'<text x="584" y="684" font-family="{FONT_MONO}" font-size="9.2" fill="{MUTED}">'
        f'* Model states carry no upstream/companion hits at P2 (P2_ONLY_DIAGNOSTIC; OPEN-INC-01).</text>'
    )

    elems.append(f'<line x1="10" y1="{H - 28}" x2="{W - 10}" y2="{H - 28}" stroke="{RULE}" stroke-width="1"/>')
    elems.append(f'<text x="26" y="{H - 15}" font-family="{FONT_MONO}" font-size="9.3" fill="{MUTED}">ship-nf · PhD thesis codebase (SHiP / ECN3 muon-DIS background)</text>')
    elems.append(f'<text x="{W - 26}" y="{H - 15}" text-anchor="end" font-family="{FONT_MONO}" font-size="9.3" fill="{MUTED}">contract/ship_exchange_v0.md</text>')

    body = "\n  ".join(elems)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="{SLATE}"/>
    </marker>
    <marker id="arr-cyan" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="{CYAN_STROKE}"/>
    </marker>
  </defs>
  {body}
</svg>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
