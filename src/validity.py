"""TrajectoryValidityReport: formal rollout validity, replacing the ad hoc 12 mm penetration
threshold of Phases 6-7. A trajectory is not scientifically valid merely because it completes.

Every GATING criterion is derived from a physical argument, not picked:
  * START_AT_REST     residual |qdot| at t=0 below 0.05 rad/s. The jump starts from rest; any
                      larger residual is free kinetic energy (the Phase 7 falling-spawn class).
  * ENERGY            (E_takeoff - E_0) - W_actuator <= tol. Mechanical energy cannot rise by
                      more than the actuators put in. tol = max(1 J, 2% of |W_actuator|), set
                      from the measured integration residual of rollouts on this plant (see
                      docs/VALIDITY_ENVELOPE.md), not tuned to admit any result.
  * ENERGY_CLOSURE    (Phase 10, closes AD3) two-sided: |dE - W_actuator - W_constraint - W_passive|
                      <= the same tol. W_constraint (contacts, joint limits, frictionloss) and
                      W_passive are integrated explicitly, so dissipation no longer hides an
                      energy-injecting defect inside the one-sided ENERGY slack. Measured closure on
                      24 stored G1 solutions: -0.35 to -5.4 J (<= 1.3% of W_actuator), all negative
                      (implicit-integration loss). None when a rollout predates the bookkeeping.
                      Tolerance RECALIBRATED after first production use (landability10: 9 clean
                      low-speed runs flagged at |closure| 3-4.5 J against a 2-3 J tol): integration
                      error scales with total energy throughput, so closure_tol =
                      max(5 J, 2% of (|W_actuator| + |W_constraint|)). A 20 J hidden injection still
                      fails (test S5b).
  * CONTACT_ENERGY    elastic energy stored in soft-contact penetration, 1/2 * F_peak * pen_max,
                      must be <= the same energy tolerance as ENERGY (the contact spring can
                      never hand back more than the accounting can resolve). Replaces the 1%-of-KE
                      draft, which was an unjustified number.
                      This replaces a fixed mm threshold with the quantity that actually
                      matters: how much energy the contact spring could hand back.
  * FOOT_ONLY         no floor contact on any non-foot geom before takeoff (a knee or hand
                      pushing off the floor is a contact exploit, not a jump).
  * TOOK_OFF          contact actually broken with upward COM velocity.
  * MOMENTUM          v_takeoff == (integral of ground normal force)/M - g*T_push, to 0.05 m/s.
                      Guards the takeoff-velocity MEASUREMENT itself (defect P8).
RECORDED, not gating (reported, never used to discard): peak GRF, GRF step spike, torque
saturation fraction, joint-limit excursion, base tilt at takeoff.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class TrajectoryValidityReport:
    start_qdot_max: float
    energy_residual: float
    energy_tol: float
    actuator_work: float
    contact_energy: float
    ke_takeoff: float
    nonfoot_contact: bool
    took_off: bool
    peak_grf_bw: float = 0.0
    grf_spike_bw: float = 0.0
    max_penetration: float = 0.0
    saturation_frac: float = 0.0
    joint_limit_excursion: float = 0.0
    upright_at_takeoff: float = 1.0
    momentum_error: float = 0.0
    energy_closure: float = None
    closure_tol: float = None
    reasons: list = field(default_factory=list)

    def evaluate(self):
        self.reasons = []
        if self.start_qdot_max > 0.05: self.reasons.append("START_NOT_AT_REST")
        if self.energy_residual > self.energy_tol: self.reasons.append("ENERGY_ACCOUNTING_VIOLATION")
        ctol = self.closure_tol if self.closure_tol is not None else self.energy_tol
        if self.energy_closure is not None and abs(self.energy_closure) > ctol:
            self.reasons.append("ENERGY_CLOSURE")
        if self.contact_energy > self.energy_tol: self.reasons.append("CONTACT_ENERGY")
        if self.nonfoot_contact: self.reasons.append("NONFOOT_CONTACT")
        if not self.took_off: self.reasons.append("NO_TAKEOFF")
        # vertical momentum: v_takeoff must equal (ground impulse)/M - g*T from rest. Tolerance
        # 0.05 m/s = one timestep of gravity plus discrete-integration slack (defect P8 guard).
        if abs(self.momentum_error) > 0.05: self.reasons.append("MOMENTUM_INCONSISTENT")
        return self

    @property
    def valid(self):
        return not self.reasons

    def to_dict(self):
        d = asdict(self); d["valid"] = self.valid; return d
