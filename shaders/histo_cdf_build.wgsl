// Builds the per-tile depth CDF from last frame's optical-depth histogram, then clears it.
// One workgroup per tile, one thread per bin.
//
// The bin count is a runtime knob (cycled with `B`), so the scan is written as a loop over
// a compile-time maximum rather than an unrolled ladder. Threads beyond `num_bins` load
// zero and are masked out of the write, which leaves the prefix sum unaffected.

struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

const OD_SCALE: f32 = 4096.0;
// Must match BIN_COUNT_STEPS' largest entry in src/renderer.rs, and the workgroup size.
const MAX_BINS: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var cdf_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> histo_params: HistoParams;

var<workgroup> buf: array<f32, MAX_BINS>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tile_x = wg.x;
    let tile_y = wg.y;
    let bin = lid.x;
    let nb = histo_params.num_bins;
    let tile_idx = tile_y * histo_params.tile_count_x + tile_x;

    // Load and dequantize this tile's histogram bin.
    var val: f32 = 0.0;
    if (bin < nb) {
        val = f32(atomicLoad(&histogram[tile_idx * nb + bin])) / OD_SCALE;
    }
    buf[bin] = val;
    workgroupBarrier();

    // Hillis-Steele inclusive prefix sum. The loop bound is a constant, so every
    // invocation runs the same number of iterations and the barriers stay in uniform
    // control flow. Read-barrier-write-barrier lets one buffer serve as both source and
    // destination without a race.
    for (var stride = 1u; stride < MAX_BINS; stride = stride << 1u) {
        var acc = buf[bin];
        if (bin >= stride) {
            acc = acc + buf[bin - stride];
        }
        workgroupBarrier();
        buf[bin] = acc;
        workgroupBarrier();
    }

    if (bin < nb) {
        let total_od = buf[nb - 1u];
        var cdf_val: f32;
        if (total_od > 0.0) {
            // Exclusive prefix: optical depth strictly in FRONT of this bin. Subtracting
            // this bin's own contribution is what keeps a fragment from being occluded by
            // itself and by everything else that happens to share its bin -- which is
            // negligible for a handful of quads but dominant for a splat cloud.
            cdf_val = (buf[bin] - val) / total_od;
        } else {
            // Linear fallback when no fragments hit this tile
            cdf_val = f32(bin) / f32(nb);
        }

        textureStore(cdf_out, vec3i(i32(tile_x), i32(tile_y), i32(bin)), vec4f(cdf_val, 0.0, 0.0, 0.0));

        // Clear histogram for next frame
        atomicStore(&histogram[tile_idx * nb + bin], 0u);
    }
}
