// common.wgsl is prepended.
// Mode 4 prepass, mesh scene: the nearest solid surface at each pixel.

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return basic_vertex(input);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let lit = simple_lighting(in.world_normal, in.color);
    if (lit.a < FRONT_CORE_ALPHA) {
        discard;
    }
    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );
    return vec4<f32>(lit.rgb, normalized_z);
}
