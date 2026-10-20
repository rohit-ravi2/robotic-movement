# Joint-strength parameter sources for the planar 5-actuator jumping model

Compiled 2026-10-04. Literature extraction only. Nothing here is tuned toward a jump outcome.

**Labels used throughout**
- **VERBATIM**: the number was read in the fetched full text (raw PDF or HTML text) and is copied exactly.
- **WEBFETCH**: read through a summarizing web fetcher, not from raw text. The number is probably right, but check it against the PDF before it becomes load-bearing.
- **DERIVED**: computed by me from VERBATIM numbers. The arithmetic is shown.
- **UNVERIFIED**: I could not confirm the number in any fetched text (search-engine summary only). Do not use it as sourced.

**Unilateral vs bilateral.** Every dynamometer number below is **unilateral** (one limb) unless marked otherwise. The model is bilaterally lumped, so leg actuators need ×2 (assuming no bilateral deficit; see section d). The one exception is King 2006 (S3) shoulder: those values are already "for the double arm".

---

## (a) Source table

| id | Full citation | PMID / DOI | Verified (title/author/year/journal seen on PubMed) | Accessed |
|---|---|---|---|---|
| S1 | Anderson DE, Madigan ML, Nussbaum MA. Maximum voluntary joint torque as a function of joint angle and angular velocity: model development and application to the lower limb. *J Biomech* 2007;40(14):3105-13 | PMID 17485097; doi:10.1016/j.jbiomech.2007.03.022; PMC6820133 (NIH author manuscript) | yes | **Full text** (PMC HTML), incl. Tables 1-3 |
| S2 | Harbo T, Brincks J, Andersen H. Maximal isokinetic and isometric muscle strength of major muscle groups related to age, body mass, height, and sex in 178 healthy subjects. *Eur J Appl Physiol* 2012;112:267-75 | PMID 21537927; doi:10.1007/s00421-011-1975-3 | yes | **Full text** (publisher PDF mirrored at paulogentil.com), incl. Table 5 and Methods |
| S3 | King MA, Wilson C, Yeadon MR. Evaluation of a torque-driven model of jumping for height. *J Appl Biomech* 2006;22(4):264-74 | PMID 17293623; doi:10.1123/jab.22.4.264 | yes | **Full text** (Loughborough repository accepted manuscript, figshare 9624392), incl. Table 1 |
| S4 | Yeadon MR, King MA, Wilson C. Modelling the maximum voluntary joint torque/angular velocity relationship in human movement. *J Biomech* 2006;39(3):476-82 | PMID 16389087; doi:10.1016/j.jbiomech.2004.12.012 | yes | **Full text** (Loughborough repository AM, figshare 9617573) |
| S5 | Forrester SE, Yeadon MR, King MA, Pain MTG. Comparing different approaches for determining joint torque parameters from isovelocity dynamometer measurements. *J Biomech* 2011;44(5):955-61 | PMID 21159340; doi:10.1016/j.jbiomech.2010.11.024 | yes | **Full text** (Loughborough repository AM, figshare 9561944) |
| S6 | Kentel BB, King MA, Mitchell SR. Evaluation of a subject-specific, torque-driven computer simulation model of one-handed tennis backhand ground strokes. *J Appl Biomech* 2011;27(4):345-54 | PMID 21896949; doi:10.1123/jab.27.4.345 | yes | **Full text** (Loughborough repository, figshare 9626582), incl. Table 2 |
| S7 | Frey-Law LA, Laake A, Avin KG, Heitsman J, Marler T, Abdel-Malek K. Knee and elbow 3D strength surfaces: peak torque-angle-velocity relationships. *J Appl Biomech* 2012;28(6):726-37 | PMID 22813601; doi:10.1123/jab.28.6.726; PMC7050840 | yes | **WEBFETCH only**: PMC page via summarizer. Direct download was blocked by a reCAPTCHA. |
| S8 | Chen H-C, Chen H-C, Cheng KB. How does arm motion enhance vertical jump performance - a simulation study. ISBS Conference Proceedings 2008 (Seoul), Modeling and Simulation | no PMID (conference proceedings; ojs.ub.uni-konstanz.de/cpa article 1850) | partial: title, authors and venue seen on the proceedings PDF itself, not on PubMed/Crossref | **Full text** (4-page proceedings PDF) |
| S9 | Stotz A, Maghames E, Mason J, Groll A, Zech A. Maximum isometric torque at individually-adjusted joint angles exceeds eccentric and concentric torque in lower extremity joint actions. *BMC Sports Sci Med Rehabil* 2022 | doi:10.1186/s13102-022-00401-9; PMC8783437 | partial: title/authors/DOI seen via WEBFETCH of the PMC page, PMID not retrieved | **WEBFETCH only**: Table 1 (angles). Torque values are only in a figure, not extracted. |
| S10 | Sale D, Quinlan J, Marsh E, McComas AJ, Belanger AY. Influence of joint position on ankle plantarflexion in humans. *J Appl Physiol* 1982;52(6):1636-42 | PMID 7107473; doi:10.1152/jappl.1982.52.6.1636 | yes | Abstract only |
| S11 | Domire ZJ, Challis JH. A critical examination of the maximum velocity of shortening used in simulation models of human movement. *Comput Methods Biomech Biomed Engin* 2010;13(6):693-9 | PMID 20162474; doi:10.1080/10255840903453082 | yes | Abstract only |
| S12 | Danneskiold-Samsøe B, et al. Isokinetic and isometric muscle strength in a healthy population with special reference to age and gender. *Acta Physiol (Oxf)* 2009;197 Suppl 673:1-68 | PMID 19744082; doi:10.1111/j.1748-1716.2009.02022.x | yes (title) | **Not accessed** (Wiley 403). Numbers below came from a search-engine summary only and are **UNVERIFIED**. |
| S13 | Selbie WS, Caldwell GE. A simulation study of vertical jumping from different starting postures. *J Biomech* 1996;29(9):1137-46 | PMID 8872270; doi:10.1016/0021-9290(96)00030-9 | yes | Abstract only. **No parameters extracted** (paywalled). |
| S14 | Cheng KB. The relationship between joint strength and standing vertical jump performance. *J Appl Biomech* 2008;24(3):224-33 | PMID 18843152; doi:10.1123/jab.24.3.224 | yes | Abstract only. **No parameters extracted** (paywalled). |
| S15 | Cheng KB, Wang CH, Chen HC, Wu CD, Chiu HT. The mechanisms that enable arm motion to enhance vertical jump performance - a simulation study. *J Biomech* 2008;41(9):1847-54 | PMID 18514208; doi:10.1016/j.jbiomech.2008.04.004 | yes | Abstract only. Paywalled; S8 is the proceedings version. |
| S16 | Pandy MG, Zajac FE, Sim E, Levine WS. An optimal control model for maximum-height human jumping. *J Biomech* 1990;23(12):1185-98 | PMID 2292598; doi:10.1016/0021-9290(90)90376-e | yes | Abstract only. Muscle-driven, not torque-driven; no parameters extracted. |
| S17 | Anderson FC, Pandy MG. A dynamic optimization solution for vertical jumping in three dimensions. *Comput Methods Biomech Biomed Engin* 1999;2(3):201-31 | PMID 11264828; doi:10.1080/10255849908907988 | yes | Abstract only. 54-muscle model; no torque parameters. |

---

## (b) Per-joint parameter tables

### Model forms (needed to interpret the parameters)

**S1 Anderson 2007** (VERBATIM, Eq. 9 and Table 1). Active torque, concentric (θ̇ ≥ 0):

`T = C1·cos(C2(θ − C3)) · [2·C4·C5 + θ̇(C5 − 3C4)] / [2·C4·C5 + θ̇(2C5 − 4C4)]`

Parameter meanings (S1 Table 1):
- C1 = "Maximum isometric joint torque (N-m)"
- C2 = "π divided by the (theoretical) range of joint angles in which active muscle force is present"
- C3 = "Joint angle at maximum isometric joint torque (rad)"
- C4 = "Angular velocity when torque is 75% of isometric torque (rad/s)"
- C5 = "Angular velocity when torque is 50% of isometric torque (rad/s)"

Eccentric torque multiplies by (1 − C6·θ̇). Passive torque is `B1·e^(k1θ) + B2·e^(k2θ)` (Eq. 1).

- **Angle convention** (S1 Results): "Joint angles were defined as positive for joint flexion (including dorsiflexion) in all joints… 0 rad represents the anatomic position in each joint."
- **Velocity sign:** positive θ̇ = concentric.
- **C1 is normalized:** S1 Table 3 footnote: "*Values are normalized by body weight×height." To get N·m, multiply by m·g·h.
- **Fixed C5/C4 ratio** (S1 Methods): "The value of this ratio was 2.020 for HE, 2.037 for HF, 2.606 for KE, 2.605 for KF, 3.602 for PF, and 2.777 for DF."

**S3/S4/S6 Loughborough 9-parameter generator** (VERBATIM, S4 Eqs. 2-4; S3 Table 1 note):
- Concentric tetanic torque: `T = C/(ωc + ω) − Tc`, with `Tc = T0·ωc/ωmax` and `C = Tc(ωmax + ωc)`.
- Multiply by the differential activation `a(ω)`, defined implicitly by `ω − ω1 = m(a − 0.5(amin + 1)) / ((1 − a)(a − amin))`.
- Multiply by the angle factor `(1 − k2(θce − θopt)²)` (S3; S6 uses an equivalent q parameter).

**T0 is the tetanic isometric torque, not the voluntary isometric torque.** Voluntary isometric = T0·a(0). I computed a(0) from each generator's (amin, m, ω1); those values are labeled DERIVED.

**S8 Cheng-group generator** (VERBATIM):
- `T = Tmax(θ)·h(ω)·A(t)`, with `h(ω) = (ω0 − ω)/(ω0 + Γω)`.
- Quote: "ω0=±20 rad/s is maximum extension angular velocity, and constant Γ=2.5 is a shape factor."
- Tmax is "effective torque for both extremities" (i.e., **bilateral**).

### Subject groups

- **S1 18-25 y group** (S1 Table 2, VERBATIM):
  - M: n = 7, 19.6 (1.1) y, 174.8 (5.2) cm, 72.8 (7.2) kg
  - F: n = 7, 19.6 (1.3) y, 160.6 (5.0) cm, 62.1 (6.3) kg
  - The n per sex is DERIVED from "14 subjects aged 18-25 … Each age group contained an equal number of males and females."
  - "All testing was done on the right leg with a Biodex System 3."
- **S2 <30 y group** (S2 Table 5 footnote, VERBATIM):
  - M: 24±5 y, 179±6 cm, 74±8 kg
  - F: 25±4 y, 168±9 cm, 59±10 kg
  - Per-group n was not extracted; the total was 93 M / 85 F aged 15-83.
  - "The dominant arm and non-dominant leg were tested."
- **S3**: n = 1, "international male high jumper of height 1.89 m and mass 82 kg". Takeoff leg (unilateral). Shoulder "for the double arm".
- **S4**: n = 2 athletes, knee extension only.
- **S5**: n = 3 male elite athletes, dominant leg.
  - S1: 25 y, 1.83 m, 86.0 kg
  - S2: 22 y, 1.82 m, 72.6 kg
  - S3: 30 y, 1.75 m, 89.3 kg
- **S6**: n = 1 elite tennis player, right arm.
- **S7** (WEBFETCH): n = 54.
  - 30 M: 24.2±5.1 y, 81.5±9.2 kg
  - 24 F: 22.8±4.2 y, 67.3±12.0 kg
  - Biodex System 3, seated, shoulder ≈25° flexion, hip 90°.
- **S9** (WEBFETCH): n = 18 M, 24.8±1.9 y, 81.2±8.5 kg, dominant leg, Isomed 2000, 60°/s.

---

### B1. Hip EXTENSION

| Quantity | Value | Units | Convention | Source | Location | Quote / derivation | Notes |
|---|---|---|---|---|---|---|---|
| C1 (normalized peak isometric), young M | 0.161 (0.049) | N·m/(N·m) = ×(BW·H) | — | S1 | Table 3, HE, 18-25 M | "0.161 (0.049)" | VERBATIM |
| C1, young F | 0.181 (0.047) | ×(BW·H) | — | S1 | Table 3 | "0.181 (0.047)" | VERBATIM. F > M when normalized; reported as-is |
| Peak isometric, young M (absolute) | ≈201 | N·m, unilateral (right leg) | — | S1 | DERIVED | 0.161 × 72.8 × 9.81 × 1.748 | Product of group means; not a reported mean |
| Peak isometric, young F (absolute) | ≈177 | N·m, unilateral | — | S1 | DERIVED | 0.181 × 62.1 × 9.81 × 1.606 | Same caveat |
| Angle of peak (C3), M | 0.932 (0.358) rad ≈ 53° | rad | flexion positive, 0 = anatomical | S1 | Table 3 | "0.932 (0.358)" | VERBATIM |
| Angle of peak (C3), F | 1.242 (0.418) rad ≈ 71° | rad | same | S1 | Table 3 | "1.242 (0.418)" | VERBATIM |
| Width C2, M / F | 0.958 (0.201) / 0.697 (0.130) | rad⁻¹ | cosine; zero at C3 ± π/(2C2) | S1 | Table 3 | — | Half-width to zero: M 1.64 rad (94°), F 2.25 rad (DERIVED) |
| Torque vs angle (% of peak), M | 20 / 59 / 87 / 100 / 94 % at θ = −0.5 / −0.05 / 0.40 / 0.85 / 1.30 rad | % | flexion + | S1 | DERIVED from C2, C3 | cos(C2(θ − C3)) | Over the S1 ROM (−0.5 to 1.3 rad) |
| ω at 75% (C4), M / F | 1.578 (0.286) / 1.567 (0.268) | rad/s | concentric + | S1 | Table 3 | VERBATIM | — |
| ω at 50% (C5), M / F | 3.190 (0.586) / 3.164 (0.542) | rad/s | — | S1 | Table 3 | VERBATIM | — |
| Zero-torque velocity implied by S1 fit, M / F | ≈6.5 / ≈6.4 | rad/s | — | S1 | DERIVED: ω0 = 2C4C5/(3C4 − C5) | — | S1 text: "hip torques decreasing to zero in an essentially linear manner". Test velocities were 60 and 120°/s plus a free "high speed" movement (Biodex 500°/s setting). Hip was tested standing. ω0 may be test-limited (limb inertia, apparatus). |
| Peak isokinetic concentric, <30 y M / F | 197 ± 58 / 151 ± 41 | N·m, unilateral (non-dominant leg) | 60°/s, ROM 60° | S2 | Table 5 "Hip extension Isokinetic" | "197 ± 58" / "151 ± 41" | VERBATIM. S2 did **not** measure isometric hip extension (isometric hip was flexion only) |
| Elite athlete generator (T0 tetanic) | 480 | N·m, unilateral (takeoff leg) | — | S3 | Table 1 "hip ext" | "T0 [Nm] … 480" | n = 1 elite high jumper |
| Elite: Tmax, ωmax, ωc, amin, m, ω1 | 719; 14.2; 3.2; 0.46; 0.29; 0.45 | N·m; rad/s; rad/s; —; —; rad/s | — | S3 | Table 1 | VERBATIM | ωc = 3.2 means a strongly curved hyperbola |
| Elite: θopt, k2, K | 1.6 rad; 0.27; 5092 N·m/rad | — | "external angle for the extensors" | S3 | Table 1 + text | VERBATIM | Mapping to anatomical flexion is not stated in S3 |
| Elite voluntary isometric = T0·a(0) | ≈303 | N·m | — | S3 | DERIVED | a(0) = 0.632 from amin, m, ω1 | — |
| Angle of peak concentric torque at 60°/s | 75° ± 11° | deg hip flexion | 0 = neutral upright standing | S9 | Table 1 | "75° ± 11° HF" | WEBFETCH. Concentric, not isometric. Supine. |
| Isometric HE, 20-29 y M / F | 326 (81.4) / 202 (48.8) | N·m | angle unknown | S12 | — | — | **UNVERIFIED** (search-engine summary only) |

### B2. Knee EXTENSION

| Quantity | Value | Units | Convention | Source | Location | Quote / derivation | Notes |
|---|---|---|---|---|---|---|---|
| C1, young M / F | 0.163 (0.040) / 0.159 (0.028) | ×(BW·H) | — | S1 | Table 3, KE | VERBATIM | — |
| Peak isometric, young M / F | ≈203 / ≈156 | N·m, unilateral (right leg) | — | S1 | DERIVED | C1 × m·g·h | Product of group means |
| Angle of peak C3, M / F | 1.133 (0.073) ≈ 65° / 1.274 (0.181) ≈ 73° | rad | flexion +, 0 = full extension | S1 | Table 3 | VERBATIM | — |
| Width C2, M / F | 1.258 (0.073) / 1.187 (0.084) | rad⁻¹ | cosine | S1 | Table 3 | VERBATIM | Half-width to zero: M 1.25 rad (72°) (DERIVED) |
| Torque vs angle (% of peak), M | 14 / 65 / 96 / 96 / 67 % at θ = 0 / 0.45 / 0.90 / 1.35 / 1.80 rad | % | flexion + | S1 | DERIVED | — | Hip was flexed ~70° during testing (rectus femoris shortened) |
| C4, M / F | 1.517 (0.593) / 1.393 (0.380) | rad/s | — | S1 | Table 3 | VERBATIM | — |
| C5, M / F | 3.952 (1.546) / 3.623 (0.989) | rad/s | — | S1 | Table 3 | VERBATIM | — |
| Implied zero-torque velocity, M / F | ≈20.0 / ≈18.2 | rad/s | — | S1 | DERIVED | 2C4C5/(3C4 − C5) | Extrapolated: tests ran at 75 and 150°/s plus a free high-speed movement |
| Torque at 2/4/8/12 rad/s (% of isometric), M | 69 / 50 / 27 / 14 % | % | — | S1 | DERIVED | — | — |
| Peak isometric, <30 y M / F | 265 ± 73 / 185 ± 47 | N·m, unilateral (non-dominant leg) | "neutral joint positions" (angle not specified) | S2 | Table 5 "Knee extension Isometric" | VERBATIM | — |
| Peak isokinetic, <30 y M / F | 215 ± 41 / 138 ± 17 | N·m | 90°/s, ROM 80° | S2 | Table 5 | VERBATIM | — |
| Isometric by angle (S7 table captioned "knee **flexion**"), M | 81.8 / 132.5 / 174.7 / 212.3 / 167.3 at 15 / 35 / 55 / 75 / 100° | N·m | 0° = full extension | S7 | Table 2 (caption as rendered on PMC) | WEBFETCH | **Possible caption swap.** These numbers peak at 75° flexion at 212 N·m, which is what knee extension does. The table captioned "knee extension" (103.5 → 71.5 N·m, monotonic decline from 15° to 100°) looks like knee flexion. Check the PDF. |
| S7 300°/s values | in table captioned "knee extension": M 39.9-49.5, F 14.2-35.4 N·m | N·m | — | S7 | Table 3 | WEBFETCH | Same caption caveat |
| Elite athletes, 9-param fits (T0 tetanic) | 310 / 325 (same athlete, 2 days), 335, 363 | N·m, unilateral (dominant leg) | — | S5 | Table 6 | VERBATIM | n = 3 elite |
| Elite ωmax (fit) | 1410 / 2000 / 2000 / 1530 °/s (24.6 / 34.9 / 34.9 / 26.7 rad/s) | °/s | — | S5 | Table 6 | VERBATIM | 2000°/s was the bound |
| Elite measured free knee-extension velocity | 1480 / 1030 / 1610 °/s (25.8 / 18.0 / 28.1 rad/s) | °/s | — | S5 | Table 1 | VERBATIM | Hip held at 140° |
| Elite ωc | 396 / 312 / 385 / 230 °/s | °/s | — | S5 | Table 6 | VERBATIM | — |
| Elite θopt and Gaussian SD r | θopt 116 / 103 / 112 / 135°; r 26.6 / 38.6 / 33.5 / 26.5° | deg | Gaussian exp(−(θopt − θ)²/2r²). Convention not stated; likely included angle with 180° = straight (inferred) | S5 | Table 6, Appendix A | VERBATIM | Inferred convention → 45-77° flexion |
| Elite amin | 0.796 / 0.721 / 0.848 / 0.784 | — | — | S5 | Table 6 | VERBATIM | — |
| Elite (2 subjects) T0, ωmax, ωc | Subject 1: 291 ± 9, 13.4, 51.4 ± 19.6; Subject 2: 303 ± 7, 26.8, 8.7 ± 5.6 | N·m; rad/s; rad/s | — | S4 | Table 1 | VERBATIM | **ωmax lower bound was fixed at 13.4 rad/s** ("set at the highest knee extension velocity in a recorded running jump"). Subject 1 sits on the bound, so it is not a measurement. |
| Elite high jumper generator | T0 328, Tmax 491, ωmax 13.4, ωc 21.2, amin 0.56, m 0.49, ω1 1.1, θopt 2.0 rad, k2 0.53, K 857 | N·m, rad/s, rad | "external angle for the extensors" | S3 | Table 1 "knee ext" | VERBATIM | ωmax = 13.4 is the same value as the S4 bound. Voluntary isometric ≈226 N·m (DERIVED, a(0) = 0.690) |
| Angle of peak concentric torque, 60°/s | 65° ± 5° | deg flexion | 0 = upright | S9 | Table 1 | WEBFETCH | Hip 90° |
| Isometric KE, 20-29 y M / F | 242 (55.8) / 160 (26.4) | N·m | — | S12 | — | — | **UNVERIFIED** |

### B3. Ankle PLANTARFLEXION

| Quantity | Value | Units | Convention | Source | Location | Quote / derivation | Notes |
|---|---|---|---|---|---|---|---|
| C1, young M / F | 0.095 (0.022) / 0.104 (0.034) | ×(BW·H) | — | S1 | Table 3, PF | VERBATIM | — |
| Peak isometric, young M / F | ≈119 / ≈102 | N·m, unilateral (right leg) | — | S1 | DERIVED | — | **Measured with knee flexed 50° and hip flexed 80°** ("Ankle testing was performed with the hip flexed 80 deg and the knee flexed 50 deg"). The gastrocnemius is slack, so this likely **underestimates** PF torque near takeoff (knee near extended). |
| Angle of peak C3, M / F | 0.408 (0.083) ≈ 23° / 0.424 (0.186) ≈ 24° | rad | **dorsiflexion positive**, 0 = anatomical | S1 | Table 3 | VERBATIM | Peak sits near the dorsiflexed end of the S1 ROM (0.5 rad). S1: PF torque "increased continuously over the range of motion". |
| Width C2, M / F | 1.391 (0.089) / 1.399 (0.190) | rad⁻¹ | cosine | S1 | Table 3 | VERBATIM | Half-width 1.13 rad (DERIVED) |
| Torque vs angle (% of peak), M | 17 / 52 / 80 / 97 / 99 % at θ = −0.60 / −0.32 / −0.05 / 0.23 / 0.50 rad | % | DF + | S1 | DERIVED | — | — |
| C4, M / F | 0.987 (0.595) / 0.862 (0.487) | rad/s | — | S1 | Table 3 | VERBATIM | — |
| C5, M / F | 3.558 (2.144) / 3.109 (1.760) | rad/s | — | S1 | Table 3 | VERBATIM | — |
| Implied ω0 | **none.** C5/C4 = 3.602 > 3, so the S1 curve never reaches zero; it asymptotes at ≈19% of isometric | — | — | S1 | DERIVED | — | Torque at 2/4/8/12 rad/s: 62 / 48 / 36 / 32 %. **Not physical as a max velocity. Do not use S1 for PF ω_max.** |
| Isokinetic "Ankle flexion", <30 y M / F | 128 ± 24 / 89 ± 28 | N·m, unilateral (non-dominant leg) | 60°/s, ROM 48° | S2 | Table 5 | VERBATIM | **Label ambiguity.** S2 calls these "ankle flexion"; magnitude implies plantarflexion. S2's isometric "ankle extension" (M 44 ± 11, F 29 ± 5) matches dorsiflexion magnitude. **No isometric plantarflexion in S2.** |
| Elite high jumper generator | T0 283, Tmax 424, ωmax 15.7, ωc 15.3, amin 0.57, m 0.46, ω1 0.86, θopt 1.5 rad, k2 0.43, K 767 | N·m, rad/s, rad | — | S3 | Table 1 "ank plant" | VERBATIM | Voluntary isometric ≈201 N·m (DERIVED, a(0) = 0.709) |
| Angle of optimum, qualitative | "With the knee bent, the optimum length for torque development corresponded to almost full dorsiflexion of the ankle. Similar results were obtained with the knee extended." | — | — | S10 | Abstract | VERBATIM | Abstract only |
| Angle of peak concentric torque, 60°/s | 7° ± 4° PF | deg | 0 = upright | S9 | Table 1 | WEBFETCH | Knee 90°, concentric. Conflicts with S1/S10 (see c). |

### B4. Shoulder FLEXION (arm swing)

| Quantity | Value | Units | Convention | Source | Location | Notes |
|---|---|---|---|---|---|---|
| Elite high jumper, **double arm** (bilateral) | T0 190, Tmax 286, ωmax 28.8, ωc 4.5, amin 0.24, m 0.00, ω1 0.00, θopt 0.5 rad, k2 0.09, K 3000 | S3 Table 1 "shld flex" | VERBATIM. "Values for shoulder torque are for the double arm in the model." With m = 0, a(0) is degenerate (0.62 at ω = ω1 = 0), so voluntary isometric ≈118 N·m bilateral is DERIVED with low confidence. |
| Elite tennis player, single arm | T0 56.29, Tmax 78.81, ωmax 20.74, ωc 6.70, amin 0.99, m 3.00, ω1 −3.00, q 0.28, θopt 4.96 rad, K 1386.9 | S6 Table 2 | VERBATIM. Voluntary isometric ≈56 N·m (a(0) ≈ 0.995, DERIVED). Angle convention for θopt not stated in the extracted text. |
| Normative young-adult isometric shoulder **flexion** | **not found** | — | S2 measured shoulder abduction/adduction only: isometric abduction <30 y M 60 ± 14, F 43 ± 13 N·m (Table 5, VERBATIM). |

### B5. Elbow flexion / extension

| Quantity | Value | Units | Convention | Source | Location | Notes |
|---|---|---|---|---|---|---|
| Isometric elbow flexion, <30 y M / F | 50 ± 18 / 30 ± 7 | N·m, unilateral (dominant arm) | "neutral joint positions" | S2 Table 5 ("Elow flexion Isometric", sic) | VERBATIM |
| Isokinetic elbow flexion, 60°/s, M / F | 52 ± 14 / 30 ± 6 | N·m | ROM 110° | S2 Table 5 | VERBATIM |
| Isokinetic elbow extension, 60°/s, M / F | 43 ± 11 / 28 ± 5 | N·m | — | S2 Table 5 | VERBATIM. No isometric extension in S2. |
| Isometric elbow flexion by angle, M | 56.4 ± 13.5 / 62.9 ± 14.8 / 63.0 ± 12.9 / 60.3 ± 12.6 / 56.4 ± 14.0 at 15 / 40 / 65 / 90 / 110° | N·m | 0° = full extension | S7 Table 4 | WEBFETCH. Broad plateau peaking at 40-65°. |
| Isometric elbow flexion by angle, F | 26.2 / 30.4 / 32.0 / 32.4 / 30.4 (same angles) | N·m | — | S7 Table 4 | WEBFETCH |
| Elbow flexion at 300°/s, M | † / † / 25.7 / 18.4 / 8.4 | N·m | — | S7 Table 4 | WEBFETCH. † = ">50% of participants unable to achieve isokinetic velocities" |
| Isometric elbow extension by angle, M | 40.2 / 50.5 / 52.8 / 50.7 / 45.4 | N·m | — | S7 Table 5 | WEBFETCH. Peak ≈65°. |
| Isometric elbow extension by angle, F | 22.1 / 27.6 / 29.2 / 27.7 / 25.1 | N·m | — | S7 Table 5 | WEBFETCH |
| Elbow extension at 300°/s, M | 5.5 / 17.1 / 24.6 / 26.7 / † | N·m | — | S7 Table 5 | WEBFETCH |
| Elite tennis player, elbow flexion | T0 75.82, Tmax 106.14, ωmax 18.10, ωc 5.43, amin 0.96, m 0.16, ω1 −1.53, q 0.03, θopt 0.33 rad, K 651.5 | — | — | S6 Table 2 | VERBATIM |
| Elite tennis player, elbow extension | T0 51.95, Tmax 72.73, ωmax 18.45, ωc 5.55, amin 0.93, m 0.03, ω1 −1.17, q 0.23, θopt 4.64 rad, K 280.3 | — | — | S6 Table 2 | VERBATIM |

### B6. Joint range of motion (from the same sources)

| Joint | ROM | Source | Notes |
|---|---|---|---|
| Hip | −0.5 rad (hyperextended) to 1.3 rad (flexed) | S1 Results | VERBATIM, "approximate" dynamometer ROM, not anatomical limits |
| Knee | 0 rad (full extension) to 1.8 rad | S1 | VERBATIM |
| Ankle | −0.6 rad (plantar flexed) to 0.5 rad (dorsiflexed) | S1 | VERBATIM. S1 notes the Biodex, not the subject, limited plantarflexion. |
| Isokinetic test ROMs | knee 80°, ankle 48°, hip 60°, elbow 110°, shoulder abd/add 70° | S2 Methods | VERBATIM. Test ranges, not anatomical ROM. |
| Elbow / knee test ranges | elbow 0-125°, knee 0-110° (0 = full extension) | S7 | WEBFETCH |

### B7. Velocity parameters used in jumping simulations

| Quantity | Value | Source | Notes |
|---|---|---|---|
| ω0 (all joints, assumed) | 20 rad/s | S8 Eq. 2 text | VERBATIM. Assumed, not measured. Same value for every joint. |
| Hyperbola shape Γ | 2.5 | S8 | VERBATIM. Gives 50% torque at ω = 0.222·ω0 = 4.4 rad/s (DERIVED). Compare S1 KE C5 = 3.95 rad/s. |
| Human vs animal Vmax | Simulations "performed with parameters derived from human muscle were much lower than previously measured jump heights" | S11 abstract | VERBATIM. Qualitative only: literature human Vmax under-predicts jumps. |

---

## (c) Disagreements, and which source is most complete

**Most complete internally consistent set: S1 (Anderson et al. 2007).** It is the only source that gives, from one protocol on one population (young adults, M and F, n = 7 each), the full set for all three jump joints:
- isometric magnitude
- angle of peak
- torque-angle width
- concentric velocity shape
- eccentric scaling
- passive torques

Every parameter is a published mean ± SD read verbatim. Its weaknesses:
1. n = 7 per sex.
2. C1 is reported normalized, so absolute N·m values are my product-of-means approximations.
3. The unweighted least-squares fit "sometimes" falls below measured torque. S5 explicitly warns this underestimates strength.
4. PF was tested with the knee at 50° flexion.
5. The PF velocity fit (C5/C4 > 3) has no zero crossing.
6. The hip ω0 (~6.5 rad/s) comes from a nearly linear fit under a standing-frame protocol.

S1 does not cover the shoulder or elbow.

The Loughborough generators (S3, S4, S5, S6) are internally consistent and include ωmax, but each is **n = 1-3 elite athletes**. Their T0 is **tetanic**, roughly 1.3-1.6× the voluntary isometric value.

**Specific disagreements**

1. **Isometric knee extension, young men** (unilateral):
   - S1: ≈203 N·m (derived)
   - S2: 265 ± 73 N·m (angle unspecified)
   - S7: ≈212 N·m peak (WEBFETCH, caption caveat)
   - S12: 242 N·m (UNVERIFIED)
   - Elite tetanic T0: 291-363 N·m (S3/S4/S5); elite voluntary ≈226 N·m (S3, derived)
   - The non-athlete range is roughly 200-265 N·m. S1 is at the low end, consistent with S5's warning about unweighted fits.
2. **Hip extension:**
   - S1 ≈201 N·m isometric (M) vs S2 197 ± 58 N·m isokinetic at 60°/s. Isokinetic torque should be below isometric, so S1 may be low or S2's posture favorable.
   - The elite high jumper's voluntary isometric is ≈303 N·m (derived).
   - S1 reports normalized female HE (0.181) above male (0.161). This is unusual and is reported as-is.
3. **Plantarflexion:**
   - S1 ≈119 N·m (knee 50° flexed) vs elite ≈201 N·m voluntary (S3). The gap is partly athlete status and partly knee posture.
   - No verified young-adult normative isometric PF value with the knee extended was found.
4. **Angle of peak plantarflexion torque:**
   - S1: ≈0.41 rad (23°) dorsiflexion, ascending toward dorsiflexion
   - S10 abstract: "almost full dorsiflexion", knee bent or extended
   - S9: 7° ± 4° plantarflexion, but that is the angle of peak **concentric** torque at 60°/s with the knee at 90°
   - The isometric sources (S1, S10) agree on a dorsiflexed optimum. S9 is a different quantity.
5. **Angle of peak knee-extension torque** agrees well across sources: S1 65-73° flexion, S9 65° ± 5°, S5 θopt 103-135° (≈45-77° flexion under the inferred convention), S3 θopt 2.0 rad.
6. **Angle of peak hip-extension torque:** S1 53° (M) / 71° (F) flexion vs S9 75° ± 11° (concentric). Roughly consistent; large SDs.
7. **ω_max (knee)**, spanning ~13-35 rad/s:
   - S1 implied zero crossing ≈20 rad/s (extrapolated)
   - S8 assumed 20 rad/s
   - S3/S4 13.4 rad/s, but this is a lower **bound** taken from a running jump, not a measurement
   - S5 measured free knee extension 18-28 rad/s, fitted 24.6-34.9 rad/s
   - The value is poorly constrained by dynamometry (max 400-500°/s ≈ 7-9 rad/s). S5's independent measurement protocol is the strongest evidence.
8. **ω_max (hip, ankle):**
   - S1 hip ≈6.5 rad/s vs elite S3 hip 14.2 rad/s
   - S1 ankle has no zero vs S3 15.7 rad/s
   - These disagree by more than 2× and are effectively unresolved for non-athletes.
9. **Curvature:** S1's hip curve is "essentially linear" (high ωc), while S3's elite hip has ωc = 3.2 rad/s (strongly curved). S5 knee ωc/ωmax ≈ 0.15-0.28.

**Recommended best-supported set** (unilateral, young adult male; double it for bilateral lumping)

| Joint | Peak isometric | θ_peak | Width | Velocity (C4 / C5 = ω at 75% / 50% torque) | Source | Caveats |
|---|---|---|---|---|---|---|
| Hip extension | ≈201 N·m | 0.93 rad flexion | C2 = 0.958 | 1.58 / 3.19 rad/s | S1 | — |
| Knee extension | ≈203 N·m | 1.13 rad flexion | C2 = 1.258 | 1.52 / 3.95 rad/s | S1 | Bracketed above by S2 (265) |
| Ankle plantarflexion | ≈119 N·m | 0.41 rad dorsiflexion | C2 = 1.391 | — | S1 | Knee-flexed underestimate. **Velocity curve unusable for ω_max**; no verified non-athlete ω_max exists. |
| Shoulder flexion | — | — | — | — | — | Only elite n = 1 values (S3 bilateral T0 190; S6 single-arm 56 N·m). No normative source. |
| Elbow flexion | 50 ± 18 (S2) or ≈63 (S7, WEBFETCH) N·m | 40-65° | — | — | S2 / S7 | — |
| Elbow extension | ≈53 N·m peak at 65° | 65° | — | — | S7 (WEBFETCH) | S2 isokinetic only (43 ± 11) |

For females, use the S1 F columns in B1-B3.

---

## (d) What I could NOT find or verify

1. **Normative young-adult isometric plantarflexion torque with the knee extended**, with torque-angle shape. S2 has no isometric PF; S1 tested at 50° knee flexion; S10 was abstract only.
2. **Normative young-adult isometric shoulder flexion torque** and its torque-angle and torque-velocity relations. Only elite n = 1 values (S3, S6).
3. **Non-athlete ω_max for hip extension and plantarflexion** from an independent measurement (an S5-style protocol). Values exist only as model fits or assumptions.
4. **Selbie & Caldwell 1996, Cheng 2008 (JAB and J Biomech), Pandy 1990, Anderson & Pandy 1999:** existence verified on PubMed, but full texts are paywalled. Their tabulated parameters were **not extracted**. Do not attribute any numeric value to them.
5. **Danneskiold-Samsøe 2009 (S12):** Wiley full text returned 403. All S12 numbers are **UNVERIFIED**.
6. **Frey-Law 2012 (S7):** raw tables not seen (reCAPTCHA). Values came via a summarizing fetcher, and the knee flexion/extension table captions appear swapped relative to physiology. Verify against the PDF before use.
7. **Stotz 2022 (S9):** torque magnitudes are only in Figure 1, not extracted. PMID not retrieved.
8. **Harbo 2012 (S2):** the isometric test angles are given only as "neutral joint positions" and were not further specified in the main text. Per-age-group n was not extracted. The "ankle flexion/extension" naming is ambiguous (see B3).
9. **Bilateral deficit:** no source here quantifies the bilateral-vs-2×unilateral correction for hip, knee or ankle. Kawakami et al. 1998 (PMID 9535581, title seen in search results only; not read) addresses plantarflexion.
10. **Anatomical ROM norms** (e.g. Soucie et al. 2011, PMID 21070485, title seen in search results only): not accessed. The ROM values above are dynamometer test ranges.
11. **Angle convention for θopt in S3/S6** ("external angle for the extensors"): stated in S3 but not mapped to anatomical flexion in the text I read. My mapping of S3 knee θopt = 2.0 rad to ~65° flexion is an inference.
