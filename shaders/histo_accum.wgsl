// common.wgsl and cdf_sample_common.wgsl are prepended.
//
// Mode 3 accumulation. Histogram recording moved out of this shader entirely -- the
// binning pass builds it by rasterization and additive blending (see binning_common.wgsl)
// -- so this pass is pure attachment blending plus read-only texture sampling again:
// no storage bindings, no atomics, nothing that disables early-Z or hidden-surface
// removal on tiled GPUs.

struct WboitOutput {
    @location(0) accum: vec4<f32>,
    // Optical depth tau = -ln(1 - alpha), accumulated ADDITIVELY. The product of (1-alpha)
    // and the sum of -ln(1-alpha) carry identical information, but the log form keeps its
    // precision where it matters: the product is recovered exactly as exp(-tau).
    @location(1) optical_depth: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return basic_vertex(input);
}

@fragment
fn fs_main(in: VertexOutput) -> WboitOutput {
    let lit = simple_lighting(in.world_normal, in.color);
    let alpha = lit.a;

    // Linearize depth
    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );

    // Quantile of the tile's cumulative optical depth at this fragment's depth, from the
    // previous frame's binning pass.
    let equalized_z = sample_cdf(in.clip_position.xy, normalized_z);

    // Transmittance in front of this fragment: exp(-tau * CDF(z)), with tau read back
    // per-pixel from the previous frame's optical-depth target.
    let prev_tau = textureLoad(prev_optical_depth_tex, vec2<i32>(in.clip_position.xy), 0).r;
    let wt = exp(-prev_tau * equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(lit.rgb * alpha * wt, alpha * wt);
    out.optical_depth = -log(max(1.0 - alpha, 1e-6));
    return out;
}
