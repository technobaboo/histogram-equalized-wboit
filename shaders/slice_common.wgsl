// Shared fragment half of mode 4, prepended to both the mesh and the splat accumulation
// shaders.
//
// Mode 3 uses the tile CDF as a *weight*: it multiplies each fragment by the transmittance
// the CDF implies. That makes the weight only as good as the assumption that the tile's
// normalized depth profile matches the pixel's, and it fails toward under-occlusion
// wherever a tile straddles a silhouette (see CLAUDE.md).
//
// Mode 4 uses the same CDF as an *ordering key* instead. Fragments are scattered into four
// depth slabs by their position along the tile's cumulative optical depth, each slab
// accumulates order-independently, and the composite resolves the four in order. The
// approximation moves from "this tile's CDF predicts my transmittance" to the far weaker
// "this tile's CDF ranks my fragments correctly", which survives tile dilution: diluting
// the CDF rescales the quantile axis but leaves it monotone, so fragments keep their
// relative order and only slab *boundaries* drift.

const OD_SCALE: f32 = 4096.0;
const SLICE_COUNT: u32 = 4u;

// Snap confident fragments to a single slab instead of splitting them across two.
//
// The argument for it: for a nearly opaque fragment the tent basis manufactures two
// half-transparent copies of what should be one solid surface, at two different depths.
//
// Measured, it does not pay: on the splat scene it costs about 3% on foreground MSE
// (6.52e-4 on, 6.35e-4 off) *and* 6% on the high-frequency residual, so the grain it was
// supposed to suppress gets worse too. Left in place, and off, because the argument for
// it is about temporal popping across frames, which neither of those metrics can see --
// if a future scene shows that, this is the switch.
const SLICE_HARD_SNAP: bool = false;

struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

// Same group 2 layout as mode 3, minus the previous frame's optical depth: slabs carry
// their own optical depth, so mode 4 needs no transmittance feedback at all.
@group(2) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(2) @binding(1) var cdf_texture: texture_3d<f32>;
@group(2) @binding(2) var cdf_sampler: sampler;
@group(2) @binding(3) var<uniform> histo_params: HistoParams;
// Binding 4 is the previous frame's optical depth in mode 3. Mode 4 has no use for that
// and rebinds the slot to this frame's front-surface prepass, so the layout is shared.
@group(2) @binding(4) var front_surface_tex: texture_2d<f32>;

struct SliceOutput {
    @location(0) slice0: vec4<f32>,
    @location(1) slice1: vec4<f32>,
    @location(2) slice2: vec4<f32>,
    @location(3) slice3: vec4<f32>,
};

/// Record this fragment's optical depth into its tile's depth histogram and read back the
/// tile's cumulative-optical-depth quantile at this depth. Identical bookkeeping to mode
/// 3; only what the result is used for differs.
fn slice_quantile(frag_xy: vec2<f32>, normalized_z: f32, optical_depth: f32) -> f32 {
    let nb = histo_params.num_bins;
    let ts = histo_params.tile_size;
    let bin = min(u32(normalized_z * f32(nb)), nb - 1u);

    let tile_x = u32(frag_xy.x) / ts;
    let tile_y = u32(frag_xy.y) / ts;
    let tile_idx = tile_y * histo_params.tile_count_x + tile_x;
    let quantized_od = u32(clamp(optical_depth * OD_SCALE, 0.0, 65535.0));
    atomicAdd(&histogram[tile_idx * nb + bin], quantized_od);

    let u = frag_xy.x / f32(histo_params.tile_count_x * ts);
    let v = frag_xy.y / f32(histo_params.tile_count_y * ts);
    // Half-texel shift lines the texel centres up with bin edges; see histo_accum.wgsl.
    let w = normalized_z + 0.5 / f32(nb);
    return textureSampleLevel(cdf_texture, cdf_sampler, vec3f(u, v, w), 0.0).r;
}

/// Demote fragments that the per-pixel front surface says are occluded.
///
/// This is the correction the tile CDF cannot make on its own. A tile that straddles a
/// silhouette mixes pixels seeing the background *without* the foreground into one CDF,
/// which flattens its front-loading and under-occludes -- background bleeding through
/// solid geometry. The front-surface prepass is per-pixel and knows better: a fragment
/// clearly behind this pixel's nearest solid surface belongs in the last slab, whatever
/// quantile the tile assigned it.
///
/// The colour term is what keeps this from over-occluding. A fragment just behind the
/// front surface that *looks* like it -- same colour -- is almost certainly part of that
/// same surface, seen a fraction deeper, and demoting it would hollow the surface out.
/// Only a fragment that is both behind and visibly different is pushed back.
///
/// A pixel with no front surface has `front.w == 1`, so `behind` is 0 everywhere and the
/// whole mechanism is inert. That is also the state of the entire background.
fn front_occlusion(quantile: f32, frag_xy: vec2<f32>, normalized_z: f32, color: vec3<f32>) -> f32 {
    let front = textureLoad(front_surface_tex, vec2<i32>(frag_xy), 0);
    let depth_delta = max(normalized_z - front.w, 0.0);

    let behind = smoothstep(0.0, FRONT_THICKNESS, depth_delta);
    let depth_gate = exp(-pow(depth_delta / FRONT_SOFTNESS, 2.0));
    let color_delta = color - front.rgb;
    let color_gate = exp(-4.0 * dot(color_delta, color_delta));

    let disagreement = behind * (1.0 - depth_gate * color_gate);
    return mix(quantile, 1.0, disagreement);
}

/// Scatter one fragment across the slabs its quantile lands between.
///
/// The slab payload is `(color * tau, tau)`, not `(color * alpha, alpha)`: the composite
/// resolves a slab as `1 - exp(-tau)` with a single averaged colour, which is the exact
/// answer for a slab of uniform colour, and the average that assumption wants is weighted
/// by optical depth rather than by alpha.
fn slice_scatter(quantile: f32, color: vec3<f32>, alpha: f32, optical_depth: f32) -> SliceOutput {
    let last = f32(SLICE_COUNT - 1u);
    let position = clamp(quantile, 0.0, 1.0) * last;

    // Splitting a fragment between two slabs keeps the assignment continuous, so a
    // fragment drifting across a boundary fades rather than popping -- at the cost of
    // duplicating it at two depths. Snap back to one slab where that cost dominates: a
    // high-alpha fragment whose neighbours agree on which slab it belongs to.
    var assigned = position;
    if (SLICE_HARD_SNAP) {
        let stability = 1.0 - smoothstep(0.10, 0.60, fwidth(position));
        let solidity = smoothstep(0.10, 0.24, alpha);
        assigned = mix(position, round(position), stability * solidity);
    }

    let lower = u32(floor(assigned));
    let upper = min(lower + 1u, SLICE_COUNT - 1u);
    let upper_weight = fract(assigned);

    let contribution = vec4<f32>(color * optical_depth, optical_depth);

    var out: SliceOutput;
    out.slice0 = vec4<f32>(0.0);
    out.slice1 = vec4<f32>(0.0);
    out.slice2 = vec4<f32>(0.0);
    out.slice3 = vec4<f32>(0.0);
    switch lower {
        case 0u: { out.slice0 += contribution * (1.0 - upper_weight); }
        case 1u: { out.slice1 += contribution * (1.0 - upper_weight); }
        case 2u: { out.slice2 += contribution * (1.0 - upper_weight); }
        default: { out.slice3 += contribution * (1.0 - upper_weight); }
    }
    switch upper {
        case 0u: { out.slice0 += contribution * upper_weight; }
        case 1u: { out.slice1 += contribution * upper_weight; }
        case 2u: { out.slice2 += contribution * upper_weight; }
        default: { out.slice3 += contribution * upper_weight; }
    }
    return out;
}
