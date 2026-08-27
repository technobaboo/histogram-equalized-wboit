// Shared half of mode 3's binning pass, prepended to both the mesh and the splat binning
// shaders (after their respective preludes, which declare `camera`).
//
// The binning pass replaces the fragment-shader histogram atomics: geometry is rasterized
// a second time at ONE PIXEL PER TILE into four RGBA16F targets with additive blending, so
// each tile-pixel's 16 channels are the tile's depth histogram. Scatter becomes ordinary
// attachment blending -- the thing ROPs and tile memory are built for -- and the histogram
// clear becomes the render pass's LoadOp. No storage writes, no atomics, no contention.
//
// The 16-bin ceiling comes from WebGPU's default maxColorAttachmentBytesPerSample of 32
// bytes: exactly four Rgba16Float attachments, which is also the mobile-safe budget.

const HISTO_BINS: u32 = 16u;

struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

@group(2) @binding(0) var<uniform> histo_params: HistoParams;

struct BinningOutput {
    @location(0) h0: vec4<f32>,
    @location(1) h1: vec4<f32>,
    @location(2) h2: vec4<f32>,
    @location(3) h3: vec4<f32>,
};

/// Remap a clip position so full-resolution pixel (x, y) lands on binning texel
/// (x / tile_size, y / tile_size).
///
/// This is not the identity when the surface size is not a multiple of the tile size: the
/// tile grid then covers tile_count * tile_size pixels, slightly MORE than the surface,
/// while rasterizing unmodified clip coordinates into a tile_count-sized target squeezes
/// the surface into exactly tile_count texels. The mismatch reaches a large fraction of a
/// tile at the far edge, which would misalign the histogram against the CDF lookup.
fn binning_clip(clip: vec4<f32>) -> vec4<f32> {
    let grid = vec2<f32>(
        f32(histo_params.tile_count_x * histo_params.tile_size),
        f32(histo_params.tile_count_y * histo_params.tile_size),
    );
    let s = camera.viewport / grid;
    return vec4<f32>(
        s.x * clip.x + (s.x - 1.0) * clip.w,
        s.y * clip.y + (1.0 - s.y) * clip.w,
        clip.z,
        clip.w,
    );
}

/// Deposit one fragment's optical depth into the two bins its depth falls between,
/// linearly weighted -- the paper's smooth cross-layer transition, exact rather than
/// stochastic, because writing a second channel is free where a second atomic was not.
///
/// Bin k's center sits at normalized depth (k + 0.5) / HISTO_BINS, so the deposit is the
/// tent-filtered density whose piecewise-linear CDF the accum pass samples: scatter and
/// gather agree on the basis, which is what removes the bin-boundary banding.
fn tent_deposit(normalized_z: f32, od: f32) -> BinningOutput {
    var bins: array<vec4<f32>, 4>;

    let p = clamp(normalized_z, 0.0, 1.0) * f32(HISTO_BINS) - 0.5;
    let lo = i32(floor(p));
    let hi_w = p - f32(lo);
    // Clamping folds the half-tent hanging off each end back into the edge bin, so no
    // optical depth is lost at z = 0 or z = 1.
    let lo_c = clamp(lo, 0, i32(HISTO_BINS) - 1);
    let hi_c = clamp(lo + 1, 0, i32(HISTO_BINS) - 1);
    bins[lo_c >> 2][lo_c & 3] += od * (1.0 - hi_w);
    bins[hi_c >> 2][hi_c & 3] += od * hi_w;

    var out: BinningOutput;
    out.h0 = bins[0];
    out.h1 = bins[1];
    out.h2 = bins[2];
    out.h3 = bins[3];
    return out;
}

/// Per-fragment optical depth, clamped so a single near-opaque fragment cannot blow out
/// the f16 accumulation (the old atomic path's u16 quantization capped at 16 as well).
fn fragment_optical_depth(alpha: f32) -> f32 {
    return min(-log(max(1.0 - alpha, 1e-6)), 16.0);
}
