// splat_common.wgsl and cdf_sample_common.wgsl are prepended.
// Mode 3: histogram-equalized WBOIT. Identical machinery to the mesh path -- weight this
// fragment by the transmittance implied by the previous frame's tile CDF. The histogram
// itself is built by the separate binning pass, so this shader only reads.

struct WboitOutput {
    @location(0) accum: vec4<f32>,
    // Optical depth tau = -ln(1 - alpha), accumulated ADDITIVELY. The product of (1-alpha)
    // and the sum of -ln(1-alpha) carry identical information, but the log form keeps its
    // precision where it matters: the product is recovered exactly as exp(-tau).
    @location(1) optical_depth: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> SplatVsOut {
    return splat_vertex(vertex_index, instance_index);
}

@fragment
fn fs_main(in: SplatVsOut) -> WboitOutput {
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

    let equalized_z = sample_cdf(in.clip_position.xy, normalized_z);

    let prev_tau = textureLoad(prev_optical_depth_tex, vec2<i32>(in.clip_position.xy), 0).r;
    let wt = exp(-prev_tau * equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(in.color * alpha * wt, alpha * wt);
    out.optical_depth = -log(max(1.0 - alpha, 1e-6));
    return out;
}
