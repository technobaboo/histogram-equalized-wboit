// common.wgsl and binning_common.wgsl are prepended.
//
// Mesh half of mode 3's binning pass: rasterize the scene at one pixel per tile and blend
// each fragment's optical depth into the tile's 16-channel histogram.

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out = basic_vertex(input);
    out.clip_position = binning_clip(out.clip_position);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> BinningOutput {
    let lit = simple_lighting(in.world_normal, in.color);
    let alpha = lit.a;

    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );

    return tent_deposit(normalized_z, fragment_optical_depth(alpha));
}
