// splat_common.wgsl is prepended.
// Mode 4 prepass, splats: the nearest *solid* splat at each pixel.
//
// Only the opaque core of each Gaussian is eligible. The faint outer support of a splat
// is not a surface -- treating it as one would place the front anchor in front of the
// thing it belongs to, and everything behind it would be wrongly demoted.

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> SplatVsOut {
    return splat_vertex(vertex_index, instance_index);
}

@fragment
fn fs_main(in: SplatVsOut) -> @location(0) vec4<f32> {
    let alpha = splat_alpha(in);
    if (alpha < FRONT_CORE_ALPHA) {
        discard;
    }
    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );
    return vec4<f32>(in.color, normalized_z);
}
