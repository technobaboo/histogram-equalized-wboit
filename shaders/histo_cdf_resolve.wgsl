// Mode 3's CDF build, as a fragment shader instead of a compute dispatch.
//
// Channel-packing the 16 bins into one tile-pixel made the per-tile prefix sum a
// PER-PIXEL operation: read this pixel's four histogram texels, scan 16 values in
// registers, write this pixel's four CDF texels. No workgroup memory, no barriers, no
// Hillis-Steele ladder -- and on hardware with subpass/tile-local reads this pass can
// merge with the binning pass so the histogram never leaves tile memory. The wgpu
// version keeps strictly same-pixel reads so that translation stays mechanical.

const HISTO_BINS: u32 = 16u;

@group(0) @binding(0) var hist0: texture_2d<f32>;
@group(0) @binding(1) var hist1: texture_2d<f32>;
@group(0) @binding(2) var hist2: texture_2d<f32>;
@group(0) @binding(3) var hist3: texture_2d<f32>;

struct CdfOutput {
    @location(0) e0: vec4<f32>,
    @location(1) e1: vec4<f32>,
    @location(2) e2: vec4<f32>,
    @location(3) e3: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle over the tile grid.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> CdfOutput {
    let px = vec2<i32>(pos.xy);
    var h: array<vec4<f32>, 4>;
    h[0] = textureLoad(hist0, px, 0);
    h[1] = textureLoad(hist1, px, 0);
    h[2] = textureLoad(hist2, px, 0);
    h[3] = textureLoad(hist3, px, 0);

    // Inclusive prefix sum: edges[i] holds the cumulative optical depth through bins
    // 0..=i, i.e. the CDF at bin edge i+1. Edge 0 is identically zero and stored nowhere;
    // the sampler reconstructs it. This is the same exclusive-prefix convention the old
    // 3D-texture path used, expressed as edge values instead of shifted texels.
    var edges: array<f32, HISTO_BINS>;
    var acc = 0.0;
    for (var i = 0u; i < HISTO_BINS; i++) {
        acc += h[i >> 2u][i & 3u];
        edges[i] = acc;
    }

    if (acc > 0.0) {
        let inv_total = 1.0 / acc;
        for (var i = 0u; i < HISTO_BINS; i++) {
            edges[i] *= inv_total;
        }
    } else {
        // Linear fallback when no fragments hit this tile.
        for (var i = 0u; i < HISTO_BINS; i++) {
            edges[i] = f32(i + 1u) / f32(HISTO_BINS);
        }
    }

    var out: CdfOutput;
    out.e0 = vec4<f32>(edges[0], edges[1], edges[2], edges[3]);
    out.e1 = vec4<f32>(edges[4], edges[5], edges[6], edges[7]);
    out.e2 = vec4<f32>(edges[8], edges[9], edges[10], edges[11]);
    out.e3 = vec4<f32>(edges[12], edges[13], edges[14], edges[15]);
    return out;
}
