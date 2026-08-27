// splat_common.wgsl and slice_common.wgsl are prepended.
// Mode 4 for Gaussian splats -- the case the technique exists for.

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> SplatVsOut {
    return splat_vertex(vertex_index, instance_index);
}

@fragment
fn fs_main(in: SplatVsOut) -> SliceOutput {
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

    let optical_depth = -log(max(1.0 - alpha, 1e-6));
    var quantile = slice_quantile(in.clip_position.xy, normalized_z, optical_depth);
    quantile = front_occlusion(quantile, in.clip_position.xy, normalized_z, in.color);
    return slice_scatter(quantile, in.color, alpha, optical_depth);
}
