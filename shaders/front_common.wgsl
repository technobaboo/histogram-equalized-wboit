// Constants shared by mode 4's front-surface prepass and the accumulation pass that reads
// it back. Prepended to both.

/// Alpha a fragment needs before it counts as a surface rather than Gaussian haze.
const FRONT_CORE_ALPHA: f32 = 0.15;

/// How far behind the front surface, in normalized depth, a fragment must be before it is
/// treated as occluded by it. The depth window is the scene's own diameter, so these are
/// fractions of that: 1.5% of the scene's extent for the surface's own thickness, 4% for
/// the soft edge beyond it.
const FRONT_THICKNESS: f32 = 0.015;
const FRONT_SOFTNESS: f32 = 0.04;
