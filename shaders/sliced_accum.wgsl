// common.wgsl and slice_common.wgsl are prepended.
// Mode 4 for the built-in mesh scene.

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return basic_vertex(input);
}

@fragment
fn fs_main(in: VertexOutput) -> SliceOutput {
    let lit = simple_lighting(in.world_normal, in.color);
    let alpha = lit.a;

    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );

    let optical_depth = -log(max(1.0 - alpha, 1e-6));
    var quantile = slice_quantile(in.clip_position.xy, normalized_z, optical_depth);
    quantile = front_occlusion(quantile, in.clip_position.xy, normalized_z, lit.rgb);
    return slice_scatter(quantile, lit.rgb, alpha, optical_depth);
}
