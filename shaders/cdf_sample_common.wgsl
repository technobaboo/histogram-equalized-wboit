// Mode 3's CDF lookup, prepended to both accumulation shaders (after their preludes).
//
// The CDF lives in four tile-resolution RGBA16F textures: channel c of texture t holds
// the normalized cumulative optical depth at bin edge (t * 4 + c + 1); edge 0 is
// identically zero. Depth interpolation is a manual lerp between the two edges straddling
// the fragment's depth (which is what a fragment f of the way through a bin picking up f
// of that bin's own optical depth means); spatial interpolation is hardware bilinear,
// optionally upgraded to a cubic B-spline so the weight field is C1 across tiles instead
// of merely C0 -- the eye Mach-bands on the C0 kinks, and exp(-tau * CDF) amplifies them
// exponentially exactly where the splat cloud is densest.

const HISTO_BINS: u32 = 16u;

// Cubic B-spline spatial gather (4 bilinear taps) instead of a single bilinear tap.
// Costs ~8 texture fetches per fragment instead of ~2; flip off to compare.
const SPATIAL_SMOOTH: bool = true;

struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

@group(2) @binding(0) var<uniform> histo_params: HistoParams;
@group(2) @binding(1) var cdf_tex0: texture_2d<f32>;
@group(2) @binding(2) var cdf_tex1: texture_2d<f32>;
@group(2) @binding(3) var cdf_tex2: texture_2d<f32>;
@group(2) @binding(4) var cdf_tex3: texture_2d<f32>;
@group(2) @binding(5) var cdf_sampler: sampler;
@group(2) @binding(6) var prev_optical_depth_tex: texture_2d<f32>;

/// Spatially filtered CDF value at bin edge `e` (0..=HISTO_BINS).
fn cdf_edge(e: u32, uv: vec2<f32>) -> f32 {
    if (e == 0u) {
        return 0.0;
    }
    let c = e - 1u;
    var v: vec4<f32>;
    switch (c >> 2u) {
        case 0u: { v = textureSampleLevel(cdf_tex0, cdf_sampler, uv, 0.0); }
        case 1u: { v = textureSampleLevel(cdf_tex1, cdf_sampler, uv, 0.0); }
        case 2u: { v = textureSampleLevel(cdf_tex2, cdf_sampler, uv, 0.0); }
        default: { v = textureSampleLevel(cdf_tex3, cdf_sampler, uv, 0.0); }
    }
    return v[c & 3u];
}

/// CDF at normalized depth `z` for one spatial tap: lerp between the straddling edges.
fn cdf_at(uv: vec2<f32>, normalized_z: f32) -> f32 {
    let e = clamp(normalized_z, 0.0, 1.0) * f32(HISTO_BINS);
    let lo = min(u32(e), HISTO_BINS - 1u);
    return mix(cdf_edge(lo, uv), cdf_edge(lo + 1u, uv), e - f32(lo));
}

/// The tile CDF sampled at this fragment: quantile of the tile's cumulative optical depth
/// at the fragment's depth, spatially smooth across the tile grid.
fn sample_cdf(frag_xy: vec2<f32>, normalized_z: f32) -> f32 {
    let tiles = vec2<f32>(f32(histo_params.tile_count_x), f32(histo_params.tile_count_y));
    let uv = frag_xy / (tiles * f32(histo_params.tile_size));
    if (!SPATIAL_SMOOTH) {
        return cdf_at(uv, normalized_z);
    }

    // Cubic B-spline reconstruction from 4 bilinear taps (per-axis pairs of weights
    // collapse into one adjusted tap each, the standard bicubic-via-bilinear trick).
    let x = uv * tiles - 0.5;
    let i = floor(x);
    let f = x - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let s0 = w0 + w1;
    let s1 = w2 + w3;
    // Tap positions in texel space, shifted to texel centers for the sampler.
    let p0 = (i - 0.5 + w1 / s0 + 0.5) / tiles;
    let p1 = (i + 1.5 + w3 / s1 + 0.5) / tiles;

    return s0.y * (s0.x * cdf_at(vec2<f32>(p0.x, p0.y), normalized_z)
                 + s1.x * cdf_at(vec2<f32>(p1.x, p0.y), normalized_z))
         + s1.y * (s0.x * cdf_at(vec2<f32>(p0.x, p1.y), normalized_z)
                 + s1.x * cdf_at(vec2<f32>(p1.x, p1.y), normalized_z));
}
