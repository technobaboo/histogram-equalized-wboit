// splat_common.wgsl and binning_common.wgsl are prepended.
//
// Splat half of mode 3's binning pass. The EWA vertex stage is resolution-independent
// (the quad and its conic live in NDC-interpolated space), so the same projection
// rasterized at one pixel per tile evaluates the same Gaussians, just sampled at tile
// granularity -- a Monte Carlo estimate of each splat's optical depth over the tile,
// which is exactly what a histogram needs.

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> SplatVsOut {
    var out = splat_vertex(vertex_index, instance_index);
    out.clip_position = binning_clip(out.clip_position);
    return out;
}

@fragment
fn fs_main(in: SplatVsOut) -> BinningOutput {
    let alpha = splat_alpha(in);
    if (alpha < 0.0) {
        discard;
    }

    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );

    return tent_deposit(normalized_z, fragment_optical_depth(alpha));
}
