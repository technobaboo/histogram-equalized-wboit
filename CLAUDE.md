# Histogram-Equalized WBOIT Demo

## Context

Comparing four transparency rendering techniques side-by-side in a wgpu demo. The demo
renders either a built-in quad/mesh scene or a **3D Gaussian Splatting** scene loaded from a
PLY file (see *3DGS Mode* below), through the same four techniques:
1. **Regular alpha blending** (baseline with ordering artifacts; with an exact per-view
   sort it is also the ground truth the quality harness scores everything else against)
2. **Naive WBOIT** (McGuire & Bavoil 2013 - static depth-based weight function)
3. **Histogram-equalized WBOIT** (uses a tiled depth histogram from the previous frame to
   build a CDF, which remaps depth so WBOIT weights spread across the full f16 exponential
   range)
4. **Quantile-sliced OIT** (uses the same CDF as an *ordering key* rather than a weight:
   fragments are scattered into four ordered optical-depth slabs, corrected by a per-pixel
   front-surface prepass, and the slabs are composited in order)

The goal is to show that adaptive weight redistribution via histogram equalization
significantly improves WBOIT quality when fragments cluster at similar depths -- and, in
mode 4, that the same histogram is worth considerably more as an ordering key than as a
weight.

Measured on `splats/rem_v3_clear.ply` (129k splats, 8px tiles, 64 bins). Loss is
foreground MSE against the sorted reference from `--quality 16 --size 960x540`; cost is
median frame time from `--headless --frames 200 --size 1280x720`:

| mode | fg MSE | PSNR | median ms |
|---|---|---|---|
| 2 naive WBOIT | 8.2e-3 | 20.9 dB | 1.5 |
| 3 histogram-equalized | 1.0e-3 | 30.0 dB | 6.0 |
| 4 quantile-sliced | 5.1e-4 | 32.9 dB | 8.6 |

Both numbers depend on the view set and the camera distance, so compare within a run, not
across runs -- the tables further down are all from `--quality 8 --size 640x360` and are
internally consistent with each other but not with this one.

**All mode 3 numbers in this file predate the rasterized-binning rework** (16
channel-packed bins, tent deposit, B-spline gather, no atomics -- see the mode 3 section)
and need re-measuring; the mode 2 and mode 4 numbers are unaffected.

## Project Setup

```
cargo new wboit-demo
```
Path: `/run/media/nova/MEDIA/2d/wboit-demo/`

### Cargo.toml dependencies
> Implemented in `Cargo.toml`

```toml
[package]
name = "wboit-demo"
version = "0.1.0"
edition = "2024"

[dependencies]
wgpu = "28"
winit = "0.30"
pollster = "0.4"
bytemuck = { version = "1", features = ["derive"] }
glam = "0.29"
env_logger = "0.11"
log = "0.4"
```

## File Structure

All files below are implemented and `cargo build` compiles clean (zero warnings).

```
wboit-demo/
├── .gitignore                  # DONE - Standard Rust gitignore
├── Cargo.toml                  # DONE - Project manifest with dependencies
├── Cargo.lock                  # DONE - Dependency lock file
├── CLAUDE.md                   # DONE - This file (project documentation)
├── splats/                     # Test 3DGS PLY files (not tracked)
├── src/
│   ├── main.rs                 # DONE - env_logger init, optional PLY argument, winit event loop
│   ├── bench.rs                # DONE - cost harness: pinned camera, GPU fence, headless
│   ├── quality.rs              # DONE - loss harness: random views vs. an exactly sorted reference
│   ├── app.rs                  # DONE - ApplicationHandler, keyboard/mouse/scroll input, redraw loop
│   ├── renderer.rs             # DONE - GPU state, 3 render paths, resize, buffer management
│   ├── camera.rs               # DONE - Orbit camera (spherical coords, mouse drag, scroll zoom)
│   ├── scene.rs                # DONE - 6 quads + cube + sphere, auto-rotation, mesh toggle
│   ├── mesh.rs                 # DONE - Procedural quad, cube, UV-sphere generators
│   ├── ply.rs                  # DONE - Binary PLY reader: INRIA + SuperSplat compressed
│   ├── splats.rs               # DONE - Splat GPU packing, importance order, async depth sorter
│   ├── vertex.rs               # DONE - Vertex, CameraUniform, ObjectUniform, HistogramParams, SplatParams
│   └── pipeline/
│       ├── mod.rs              # DONE - Re-exports
│       ├── alpha_blend.rs      # DONE - Mode 1 pipeline (SrcAlpha/OneMinusSrcAlpha blend)
│       ├── naive_wboit.rs      # DONE - Mode 2 accum pipeline (MRT) + composite pipeline
│       ├── histogram_wboit.rs  # DONE - Mode 3 accum + binning + CDF-resolve pipelines (+ mode 4's compute CDF)
│       ├── sliced_oit.rs       # DONE - Mode 4 front prepass + slab accum + ordered composite
│       └── splat.rs            # DONE - 3DGS variants of all four accumulation pipelines + binning
└── shaders/
    ├── common.wgsl             # DONE - Camera/Object/VertexInput/VertexOutput structs, basic_vertex, simple_lighting
    ├── alpha_blend.wgsl        # DONE - vs_main/fs_main calling basic_vertex + simple_lighting
    ├── wboit_accum.wgsl        # DONE - McGuire&Bavoil weight function, MRT output (accum + revealage)
    ├── wboit_composite.wgsl    # DONE - Fullscreen triangle, textureLoad accum/revealage, alpha composite
    ├── binning_common.wgsl     # DONE - Mode 3 binning pass: tile-grid clip remap, tent deposit into 16 channels
    ├── histo_binning.wgsl      # DONE - Mesh binning pass (scene rasterized at 1 px/tile, additive blend)
    ├── splat_binning.wgsl      # DONE - Splat binning pass (same EWA projection at 1 px/tile)
    ├── histo_cdf_resolve.wgsl  # DONE - Fragment-shader CDF build: per-pixel 16-bin scan + normalize
    ├── cdf_sample_common.wgsl  # DONE - Channel-packed CDF lookup: manual depth lerp, B-spline spatial gather
    ├── histo_accum.wgsl        # DONE - CDF lookup + transmittance weight (no histogram writes anymore)
    ├── histo_cdf_build.wgsl    # DONE - (mode 4 only) compute prefix sum over the atomic histogram buffer
    ├── histo_composite.wgsl    # DONE - WBOIT composite only (textures + revealage flag)
    ├── splat_common.wgsl       # DONE - EWA projection, conic evaluation, SH degree-3 eval
    ├── splat_alpha.wgsl        # DONE - Mode 1 fragment output for splats
    ├── splat_wboit.wgsl        # DONE - Mode 2 fragment output for splats
    └── splat_histo.wgsl        # DONE - Mode 3 fragment output for splats
```

## Data Types

### Vertex (40 bytes)
> Implemented in `src/vertex.rs:4-9` with `layout()` method at `:11-35`

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],  // location 0
    normal: [f32; 3],    // location 1
    color: [f32; 4],     // location 2
}
```

### CameraUniform (64 bytes)
> Implemented in `src/vertex.rs:38-41`

```rust
#[repr(C)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}
```

### ObjectUniform (80 bytes)
> Implemented in `src/vertex.rs:44-48`

```rust
#[repr(C)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    color: [f32; 4],  // RGBA override/tint
}
```

### HistogramParams (16 bytes)
> Implemented in `src/vertex.rs:51-56`

```rust
#[repr(C)]
struct HistogramParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    depth_range: f32,  // max linear depth for binning
}
```

## Render Architecture

### Mode 1: Alpha Blend
> Pipeline: `src/pipeline/alpha_blend.rs` (single `RenderPipeline`)
> Shader: `shaders/alpha_blend.wgsl` (prepended with `shaders/common.wgsl`)
> Render path: `src/renderer.rs` `render_alpha_blend()`

Single pass:
- Color target: swapchain (clear to dark gray)
- Depth: depth texture (Depth32Float, clear)
- Blend: `SrcAlpha / OneMinusSrcAlpha` (standard)
- Draw all transparent objects unsorted

### Mode 2: Naive WBOIT
> Pipelines: `src/pipeline/naive_wboit.rs` (`accum_pipeline` + `composite_pipeline` + `composite_bgl`)
> Shaders: `shaders/wboit_accum.wgsl` (prepended with common), `shaders/wboit_composite.wgsl` (standalone)
> Render path: `src/renderer.rs` `render_naive_wboit()`

**Pass 1 - Accumulation (MRT):**
- Color 0: `Rgba16Float` accum texture (clear to 0,0,0,0)
  - Blend: `One + One` (additive)
- Color 1: `R16Float` optical-depth texture (clear to 0)
  - Blend: `Zero + OneMinusSrc` (multiplicative)
- Depth: depth texture (write disabled, depth test only)
- Fragment outputs: `(premul_color * w, alpha * w)` to color0, `alpha` to color1
- Weight function: `w = alpha * clamp(exp2(13.0 - 26.0 * d), 1e-4, 8192.0)`, where `d` is
  depth normalized over the geometry's depth window (see *Depth binning window*). This steps
  evenly through the f16 exponent range rather than using McGuire & Bavoil's original curve.

**Pass 2 - Composite (fullscreen triangle):**
- Color target: swapchain (clear to background)
- Blend: `SrcAlpha / OneMinusSrcAlpha`
- Sample accum and optical-depth textures
- Output: `color = accum.rgb / max(accum.a, 1e-5)`, `alpha = 1 - exp(-tau)`

### Mode 3: Histogram-Equalized WBOIT (tiled, 4 passes, no atomics)
> Pipelines: `src/pipeline/histogram_wboit.rs` (`accum_pipeline` + `binning_pipeline` +
> `cdf_resolve_pipeline` + `composite_pipeline`), plus `SplatPipelines::binning_pipeline`
> Shaders: `shaders/histo_accum.wgsl` (prepended with common + cdf_sample_common),
> `shaders/{binning_common,histo_binning,splat_binning}.wgsl`,
> `shaders/histo_cdf_resolve.wgsl`, `shaders/histo_composite.wgsl` (standalone)
> Render path: `src/renderer.rs` `render_histogram_wboit()` + `draw_binning()`
> Texture setup: `create_tile_histo_textures()` in `src/renderer.rs`, rebuilt on resize/`T`

Reworked after the layered-WBOIT reading (Friederichs et al., GI 2021): the histogram is
now built by **rasterization + additive blending** instead of fragment-shader atomics, and
every discretized axis has a smooth basis on the *write* side, not just the read side.
Fixed at 16 depth bins, channel-packed into four `Rgba16Float` tile-resolution textures
(4 MRTs x 32 bytes/sample = WebGPU's default limit, the mobile-safe budget). Every
cross-pass read is same-pixel by construction ("subpass-shaped"), so a Vulkan backend can
fuse binning+resolve and accum+composite into subpasses / `dynamic_rendering_local_read`
with the histogram never leaving tile memory.

**Pass 1 - Accumulation:**
- Same MRT setup as Mode 2 (accum `Rgba16Float` + optical depth `R16Float`)
- Group 2: params, the four CDF textures resolved on the *previous* frame, sampler, and
  the previous frame's optical depth. Read-only -- no storage bindings, nothing that
  disables early-Z / hidden-surface removal on tiled GPUs.
- `sample_cdf()` (`cdf_sample_common.wgsl`): hardware bilinear across tiles, upgraded to a
  cubic **B-spline gather** (4 bilinear taps, `SPATIAL_SMOOTH` const to compare) so the
  weight field is C1 across the tile grid; depth is a manual lerp between the two CDF
  edges straddling the fragment (edge `e` lives in channel `(e-1)&3` of texture `(e-1)>>2`,
  edge 0 is implicit zero). Same exclusive-prefix self-occlusion semantics as before.
- Weight `exp(-prev_tau * CDF(z))` with `prev_tau` from the optical-depth feedback texture,
  exactly as before.

**Pass 2 - Composite:** unchanged (fullscreen triangle, accum + optical depth + flag).
Runs *before* the histogram passes: they only feed the next frame, so they sit at the tail
of the frame off the critical path, where a tiler overlaps them with presentation.

**Pass 3 - Binning:** the scene rasterized a second time at **one pixel per tile** into
the four histogram targets, additive `One + One`, no depth attachment. `LoadOp::Clear` IS
the histogram clear (free on tilers). Each fragment computes `od = -ln(1-alpha)` (clamped
to 16) and `tent_deposit()` splits it linearly between the two bins its depth falls
between -- the paper's smooth cross-layer weighting, exact rather than stochastic, because
a second channel write is free where a second atomic was not. Scatter and gather now agree
on the tent basis in depth, which is what removed the bin-boundary banding.
`binning_clip()` remaps clip x/y so full-res pixel (x,y) lands on tile texel (x/ts, y/ts)
even when the surface size is not a multiple of the tile size. The splat variant reuses
`splat_vertex()` unchanged (the quad + conic are resolution-independent), giving a Monte
Carlo estimate of each splat's optical depth over the tile.

**Pass 4 - CDF resolve (fragment, not compute):** channel-packing made the per-tile prefix
sum *per-pixel*: read this pixel's four histogram texels, scan 16 values in registers,
normalize (linear ramp fallback for empty tiles), write this pixel's four CDF texels. No
workgroup memory, no barriers, no Hillis-Steele. The CDF set is double-buffered: written
on frame N, read by frame N+1's accumulation (one-frame lag as before; frame 0 reads zeros
= uniform weights for one frame).

### Mode 4: Quantile-Sliced OIT (4 passes)
> Pipelines: `src/pipeline/sliced_oit.rs` (`front_pipeline` + `accum_pipeline` + `composite_pipeline`),
> plus `SplatPipelines::{front_pipeline, sliced_pipeline}` for the splat variants
> Shaders: `shaders/front_surface.wgsl` / `splat_front_surface.wgsl` (prepass),
> `shaders/slice_common.wgsl` + `sliced_accum.wgsl` / `splat_sliced.wgsl` (accum),
> `shaders/sliced_composite.wgsl`
> Render path: `src/renderer.rs` `render_sliced_oit()`

Mode 3 uses the tile CDF as a **weight**, which is only as good as the assumption that the
tile's normalized depth profile matches the pixel's -- the tile-dilution failure documented
below. Mode 4 uses the same CDF as an **ordering key**, which is a far weaker assumption:
diluting the CDF rescales the quantile axis but leaves it *monotone*, so fragments keep
their relative order and only slab boundaries drift. It still runs the ORIGINAL histogram
machinery -- the atomic storage buffer, the 3D CDF volume, and the Hillis-Steele compute
pass -- which mode 3 no longer touches after its rework; the rasterized-histogram
treatment has not been ported to mode 4 yet.

**Pass 0 - Front-surface prepass:**
- Color target: `Rgba16Float`, `(color.rgb, normalized_z)`, cleared to `(0,0,0,1)`
- Depth: the shared depth texture, **write enabled, compare Less** -- the only pipeline in
  the demo that does a real depth test, because it is resolving the *nearest* qualifying
  fragment per pixel
- Fragments below `FRONT_CORE_ALPHA` (0.15) are discarded: the faint outer support of a
  Gaussian is not a surface, and anchoring to it would demote everything genuinely behind it

**Pass 1 - Slab accumulation (4x MRT):**
- Four `Rgba16Float` targets, additive `One + One`
- Same histogram `atomicAdd` and CDF sample as mode 3, giving `quantile = CDF_tile(z)`
- `front_occlusion()` then corrects that quantile against the prepass (see below)
- `slice_scatter()` places the fragment at `quantile * 3` along the four slabs, split
  between the two it falls between, carrying `(color * tau, tau)`

**Pass 2 - CDF build:** the compute dispatch (`histo_cdf_build.wgsl`) that used to also be
mode 3's pass 2; now mode 4 is its only user.

**Pass 3 - Ordered composite:** each slab resolves to `1 - exp(-tau)` with a
tau-weighted average colour, then the four are alpha-composited front to back.

### Why mode 4 works, and which half of it does the work
> `shaders/slice_common.wgsl` `front_occlusion()`

The slicing on its own is worth almost nothing. Measured on the splat scene at 8px tiles,
slicing without the front-surface correction scores about what mode 3 scores; the entire
gain comes from `front_occlusion()`. The ablation, foreground MSE, lower is better:

| variant | fg MSE |
|---|---|
| slicing alone (no prepass) | ~1.7e-3 (i.e. mode 3) |
| + front anchor only | 2.8e-3 |
| + depth-gated demotion, no colour term | 1.6e-3 |
| + demotion with the colour term | 6.4e-4 |

What the correction does is narrow: a fragment clearly *behind* this pixel's nearest solid
surface, and visibly different in colour from it, has its quantile pushed to 1 -- into the
last slab, where everything in front occludes it. That is exactly the repair tile dilution
needs, because tile dilution's error is always toward *under*-occlusion, and the prepass is
per-pixel where the CDF is per-tile.

The colour term is what keeps it from over-correcting. A fragment just behind the front
surface that looks like it is almost certainly part of that same surface seen a fraction
deeper; demoting it would hollow the surface out. Dropping the colour term costs 2.5x.

A pixel with no front surface reads back `front.w == 1`, so `behind` is 0 and the whole
mechanism is inert -- which is the state of the entire background, for free.

**Ordering key beats weight at every tile size, and degrades far more gracefully.** Mode 4
at 32px tiles is better than mode 3 at 4px tiles while using 1/64th the CDF memory:

| tile | mode 3 fg MSE | mode 4 fg MSE |
|---|---|---|
| 32 px | 4.2e-3 | 8.5e-4 |
| 16 px | 2.9e-3 | 7.4e-4 |
| 8 px | 1.8e-3 | 6.4e-4 |
| 4 px | 1.0e-3 | 5.8e-4 |

Mode 3 improves 4.1x across that range, mode 4 only 1.5x -- the signature of the axis
having moved off the spatial resolution of the CDF.

**Cost.** The slab MRT is nearly free; the prepass is not. At 1280x720, 129k splats,
median ms: mode 3 = 6.0, mode 4 without the prepass = 6.3 (+5%), mode 4 with it = 8.6
(+43%). The extra geometry pass is the whole cost, and it buys 2.8x on loss.

### Histogram / CDF Storage

**Mode 3** (`create_tile_histo_textures()` in `src/renderer.rs`, rebuilt on resize/`T`):
- Histogram: four `Rgba16Float` textures at one texel per tile; channel `c` of texture `t`
  is bin `t*4 + c` (16 bins fixed). Accumulated by additive blending in the binning pass,
  cleared by `LoadOp::Clear`. Each bin holds **optical depth** (`-ln(1-alpha)`, f16,
  clamped to 16 per fragment), not fragment count.
- CDF: same four-texture layout, **double-buffered**; channel `c` of texture `t` holds the
  normalized cumulative optical depth at bin edge `t*4 + c + 1` (edge 0 is implicit zero).
  Written by the CDF-resolve fragment pass on frame N, read by frame N+1's accumulation.
- f16 is enough: the CDF only ever consumes ratios, and the old path quantized through
  `OD_SCALE = 4096` anyway.

**Mode 4** (unchanged): `array<atomic<u32>>` histogram buffer of `tiles * num_bins`
quantized-OD counters plus the `Rgba16Float` 3D CDF volume, built by the
`histo_cdf_build.wgsl` compute pass, `num_bins` cycled with `B`.

## Key Technical Notes

- **Depth binning window (`camera.depth_min` / `camera.depth_range`)**: Both WBOIT modes
  normalize eye-space depth to `[0,1]` before using it, mode 2 for its weight curve and
  mode 3 for its histogram bins. That normalization uses the span the geometry actually
  occupies -- `[distance - scene_radius, distance + scene_radius]` -- **not** `near`/`far`.
  Using near/far is a trap: a camera fitted to a splat scene sets `near = 0.01r` and
  `far = 50r`, so the geometry (which sits at ~1.8r..3.2r) lands in 4% of the range. Every
  fragment then collapses into ~1.7 of the 64 histogram bins, mode 2's weight ratio falls
  from ~1e7x to 1.5x, and both modes degenerate into a flat average along the ray -- the
  washed-out look. With the fitted window the same scene spans ~38 bins. `Camera::uniform()`
  derives the window each frame; `scene_radius` defaults to 6.0 for the built-in quad scene
  and is set by `fit_to()` for loaded splats.
- **CDF is stored at bin EDGES (exclusive prefix)**: the value at edge `k` is the optical
  depth strictly in *front* of `z = k/N`. Two reasons. First, transmittance in front of a
  fragment must not include the fragment's own optical depth, nor that of anything else
  sharing its bin -- negligible for six quads, but dominant for a splat cloud with hundreds
  of fragments per bin. Second, interpolating between edge values means a fragment a
  fraction `f` through bin k correctly picks up `f` of that bin's own optical depth. Mode
  3 stores edges 1..16 directly in the CDF channels (edge 0 is implicit zero) and lerps
  between the two straddling edges in `cdf_at()`; mode 4's 3D volume expresses the same
  convention as a half-texel shift, sampling at `normalized_z + 0.5/num_bins` because a 3D
  texture samples texel k at `(k+0.5)/N`. Sampling an inclusive CDF at `normalized_z` (the
  earliest form) read `tau(z + 0.5/N)` -- half a bin of over-occlusion on top of the
  self-occlusion.
- **Scatter and gather must share a smooth basis** (the Friederichs et al. GI 2021 lesson
  behind the mode 3 rework). The gather side was always interpolated -- bilinear across
  tiles, linear across bins -- but the scatter side was hard-binned in both axes: a
  fragment dumped 100% of its optical depth into exactly one (tile, bin) while a pixel
  near a tile edge *read* a 50/50 blend of tiles it wrote nothing/everything into. The
  interpolated CDF field was C0 but its content snapped as geometry crossed boundaries
  (grid-aligned popping), and C0 kinks Mach-band -- amplified by `exp(-tau * CDF)` exactly
  where the cloud is densest. Fixes now in place: `tent_deposit()` splits each fragment's
  optical depth linearly between its two straddling bins (free with channel-packed MRTs;
  it used to cost a second atomic), and the accum gather is a cubic B-spline across tiles
  (C1). The remaining asymmetry is spatial scatter (still box-per-tile, since a fragment
  can only blend into its own pixel) -- covered from the gather side.
- **Why the transmittance weight is exact**: `tau_total = -ln(R_prev)` and
  `CDF(z) = tau(z)/tau_total`, so `pow(R_prev, CDF(z)) = exp(-tau(z)) = T(z)`, the exact
  transmittance in front of the fragment. Since `sum(a_i * T_i)` telescopes to
  `1 - prod(1 - a_i)`, the composite `avg_color * (1 - exp(-tau))` reduces algebraically to
  the exact front-to-back integral. Mode 3's error is therefore entirely in how well the
  binned, tiled, one-frame-late CDF approximates the true `tau(z)`.
- **Tile size is the dominant quality knob in mode 3** (`TILE_SIZE_STEPS`, cycled with `T`,
  default 8px). The weight applied is `w_i = prev_R ^ CDF_tile(z_i)`, i.e.
  `exp(-tau_pixel * CDF_tile(z_i))`, but the correct transmittance is `exp(-tau_pixel(z_i))`.
  Those agree **only if** `CDF_tile(z) == tau_pixel(z) / tau_pixel` -- that is, only if the
  tile's *normalized* depth profile matches the pixel's. The magnitude of optical depth is
  per-pixel and correct (it comes from the optical-depth texture); only its **distribution in
  depth** is borrowed from the tile.

  The failure is directional: always toward **under-occlusion**. Any pixel in the tile that
  sees the background *without* the foreground adds optical depth to the far bins while
  adding none to the near bins, flattening the CDF's front-loading -- which is precisely the
  quantity that produces occlusion. Worked example: a pixel sees an arm (tau=5) at z1 then a
  sign (tau=5) at z2, so tau_pixel=10. Truth is `w_arm : w_sign = 1 : e^-5`, a ratio of 148.
  If the tile holds only pixels like this one, `CDF_tile(z2) = 0.5` and the model reproduces
  148 exactly. If half the tile's pixels see the sign but not the arm, `CDF_tile(z2)` falls
  to 0.333, the ratio collapses to 28, and the sign bleeds through the arm.

  This is why the technique looks near-perfect on the quad scene and washed out on splats:
  six big flat quads means every pixel in a tile sees the same quads at the same depths, so
  `CDF_tile ~= CDF_pixel`. A detailed 3DGS figure has depth complexity varying pixel to
  pixel, and at 32px granularity almost every tile straddles a silhouette, so the dilution is
  global rather than confined to edges. Measured: 32px ghosts badly, 8px is nearly clean, 4px
  matches sorted alpha blending.

  Note that **more depth bins does not help** -- the bins refine the depth axis, but this
  error lives on the spatial axis. Within mode 3, only smaller tiles (or a per-pixel CDF,
  which is far too much memory) address it. Mode 4 addresses it a different way, by
  demoting the CDF from a weight to an ordering key and repairing what is left with a
  per-pixel front-surface prepass -- which is why its tile-size curve is nearly flat where
  mode 3's is steep. Mode 3's channel-packed storage costs 12 tile-res RGBA16F textures
  (hist + 2x CDF) = 96 bytes/tile: 8px at 1080p is ~3.1 MB where the old 64-bin
  buffer+volume was ~25 MB. Mode 4's 3D CDF volume must stay `Rgba16Float` even though
  only `.r` is used, because `r32float` is not filterable in core WebGPU without
  `float32-filterable`, and the trilinear sample is load-bearing; mode 3's 2D textures are
  fully packed instead, with the depth lerp done in ALU.
- **Store optical depth, not revealage.** These are the same quantity --
  `prod(1 - a_i) = exp(sum(ln(1 - a_i))) = exp(-tau)` -- but they are not equally
  representable. The original design accumulated the *product* multiplicatively into an
  **R8Unorm** target, which bottoms out at `1/255`: any pixel with `tau > ln(255) = 5.54`
  stored 0, and its true optical depth was gone. Mode 3 then had to recover `tau` as
  `-ln(prev_R)` and floor the result against a magic constant, so that floor became the
  assumed `tau` for *every* saturated pixel.

  That is fatal for splats and invisible for quads. A splat scene builds opacity from many
  low-alpha Gaussians and routinely reaches `tau` of 20-30 per pixel, so essentially every
  solid pixel saturates. Six quads at `alpha ~ 0.4` give `tau ~ 3`, i.e. `R ~ 13/255` --
  comfortably inside the format. And because transmittance is *exponential* in `tau`, a
  shortfall of `d_tau` multiplies a background fragment's weight by `e^d_tau`; occlusion is
  exactly the front/back weight ratio, so under-counting `tau` by 10 makes the background
  22000x too bright. That was the see-through-figure artefact.

  The fix is to accumulate `tau` **additively** into an `R16Float` target instead. The
  information content is identical, but the log form is the well-conditioned one: `tau`
  grows linearly and fits f16 exactly, where `R` decays exponentially toward zero and
  destroys itself in 8 bits. Everything downstream simplifies:
  - the weight becomes `exp(-prev_tau * CDF(z))` -- no `log`, no `pow`, no floor, no guess
  - the composite's alpha becomes `1 - exp(-tau)`, which is now exact rather than
    quantized
  - the accum pass writes `tau` that mode 3 has already computed for its histogram

  General lesson worth keeping: **put the precision budget in the space where the quantity
  is linear.** Storing a transmittance is storing `e^-x` at fixed point; storing `x` costs
  the same bandwidth and loses nothing.
- **Three independent resolution limits govern mode 3.** They fail in different ways and are
  fixed by different knobs, so it is worth keeping them apart:
  - *spatial* -- tile size (`T`). Mixes pixels with different depth profiles into one CDF.
  - *depth* -- bin count (`B`). Two layers closer than one bin cannot be separated, so each
    is credited with part of the other's optical depth.
  - *magnitude* -- how much total optical depth a pixel can express. Was the tightest of
    the three while revealage lived in R8Unorm; resolved by storing `tau` in R16Float, per
    the note above.
- **Premultiplied alpha and sRGB do not commute** -- this is why partially transparent
  pixels composite too bright against the desktop. The surface negotiates
  `Bgra8UnormSrgb` + `CompositeAlphaMode::PreMultiplied` (logged at startup). With an sRGB
  format the GPU blends in linear space and applies the linear->sRGB encode *on write*, so
  the composite shader's `vec4(avg_color * alpha, alpha)` lands in memory as
  `srgb(color * alpha)`. But the near-universal convention for premultiplied 8-bit sRGB
  buffers -- Wayland's ARGB8888, Cairo, Skia -- is that premultiplication applies to the
  *encoded* value, `srgb(color) * alpha`, and compositors blend those encoded values
  directly.

  Since sRGB encoding is nonlinear, `srgb(c*a) != srgb(c)*a`. The two agree exactly at
  `alpha = 0` and `alpha = 1` and diverge worst in between, which is the signature of the
  artefact: opaque geometry looks right, translucent geometry looks washed out. For white
  at `alpha = 0.2` we write 124/255 where the compositor expects 51/255 -- **73/255 too
  bright**. Alpha itself is never sRGB-encoded, so only RGB is affected.

  `W` sidesteps it by clearing the swapchain opaque, forcing every pixel to `alpha = 1`
  where the two conventions coincide. A real fix would either encode manually into a
  non-sRGB surface (`vec4(linear_to_srgb(avg_color) * alpha, alpha)` into `Bgra8Unorm`,
  which also moves mode 1's blending into gamma space) or pre-compensate into the sRGB
  surface (`srgb_to_linear(linear_to_srgb(avg_color) * alpha)`, which keeps linear blending
  at the cost of a round trip).
- **Atomics in fragment shaders** (mode 4 only, since the mode 3 rework): `atomicAdd` on
  `var<storage, read_write>` IS allowed in fragment shaders per WGSL spec. The underlying
  Vulkan feature `fragmentStoresAndAtomics` is widely supported on desktop, but fragment
  storage writes disable early-Z/hidden-surface fast paths on mobile tilers and the
  same-address contention serializes on dense splat clouds -- which is exactly why mode 3
  moved to rasterized binning (histogram = scatter + accumulate = rasterizer + blend unit).
- **CDF build**: mode 3 resolves per-pixel in a fragment pass (16-bin scan in registers,
  see `histo_cdf_resolve.wgsl`); mode 4 keeps the one-workgroup-per-tile Hillis-Steele
  compute pass over the atomic buffer, which also clears the histogram.
- **First frame** (mode 3): the CDF textures start zero-initialized, so frame 0 runs with
  uniform weights and frame 1 onward is equalized; empty tiles fall back to a linear ramp
  in the resolve. The harnesses' priming frames absorb the warm-up.
- **Temporal lag**: 1-frame delay is acceptable. CDF adapts smoothly as camera moves.
- **R16Float for optical depth**: Additive blend `(One, One)` accumulates `tau = sum(-ln(1 - alpha))`. `r16float` is renderable and blendable in core WebGPU; `r32float` would need the `float32-blendable` feature, and f16 already resolves `tau` to better than 0.1% across the 0-30 range these scenes need.
- **Rgba16Float for accum**: Supports blending without the `float32-blendable` feature.
- **Shader loading**: `format!("{}\n{}", common_wgsl, specific_wgsl)` concatenation since WGSL has no `#include`. Used in all three pipeline files via `include_str!`.
- **CDF buffer as non-atomic storage**: The CDF buffer is written by compute shader (pass 2) and read by fragment shader (pass 1 of next frame). Since these are in different passes of different frames, there's no synchronization issue. Use `var<storage, read_write>` with plain `f32` (not atomic).
- **wgpu 28 API**: `PipelineLayoutDescriptor` uses `immediate_size` (not `push_constant_ranges`), `RenderPipelineDescriptor` uses `multiview_mask: None` (not `multiview: None`), `RenderPassColorAttachment` requires `depth_slice: None`.

## 3DGS Mode
> Implemented in `src/ply.rs`, `src/splats.rs`, `src/pipeline/splat.rs`, `shaders/splat_*.wgsl`

```
cargo run --release -- splats/rem_v3_clear.ply
```

With a PLY argument the splat scene replaces the quad/mesh scene entirely; all three
transparency modes render it. Without an argument, nothing changes from the mesh demo.

### PLY parsing
> `src/ply.rs`

Two on-disk variants are handled, both `binary_little_endian`:

- **INRIA / original 3DGS**: `x,y,z`, `scale_0..2` (log space, exponentiated on load),
  `rot_0..3` (`w,x,y,z`, normalized), `opacity` (logit, sigmoid applied on load),
  `f_dc_0..2`, and optionally `f_rest_0..44` (channel-major: 15 coeffs of R, then G, then B).
- **PlayCanvas / SuperSplat compressed**: an `element chunk` of per-256-splat bounds plus
  four packed `u32` per vertex. Bit layouts match the PlayCanvas engine decoder:
  - `packed_position` / `packed_scale`: 11/10/11 unorm, lerped between the chunk's
    min/max bounds. Scale is still log space.
  - `packed_rotation`: 2-10-10-10 "largest three" — the top 2 bits name the omitted
    (largest) component, the rest are unorm mapped to `[-1/sqrt(2), 1/sqrt(2)]`.
  - `packed_color`: 8888. RGB lerps between the chunk's colour bounds and is *already* the
    evaluated DC band (not a raw `f_dc` coefficient); A is *already* sigmoid-applied opacity.
  - An optional `element sh` carries quantized higher bands (`byte * 8/255 - 4`).

The loader reports a clear error on a truncated file rather than producing garbage.

### Coordinate frame
3DGS reconstructions come out of COLMAP with +Y down and +Z into the screen. Positions and
covariances are flipped by `diag(1, -1, -1)` at load time so the existing orbit camera's
+Y-up convention is correct. The shader undoes that flip on the view direction before
evaluating SH, whose coefficients live in the original frame.

### GPU layout
- `SplatGpu`, 64 bytes: `vec4(pos.xyz, opacity)`, `vec4(cov_xx, cov_xy, cov_xz, _)`,
  `vec4(cov_yy, cov_yz, cov_zz, _)`, `vec4(color.rgb, _)`. The 3D covariance is precomputed
  on the CPU as `M M^T` where `M = flip * R(quat) * diag(scale)`.
- SH buffer: `array<f32>`, 45 floats per splat, channel-major. A one-float dummy is bound
  when the file has no higher bands, so the bind group layout is the same either way.
- Order buffer: `array<u32>`, instance index -> splat index.

Splats are stored **sorted by importance** (`opacity * cbrt(sx*sy*sz)`) so the render cap
can simply draw a prefix and get the best subset of that size.

### Rendering
> `shaders/splat_common.wgsl`

One instanced 4-vertex triangle strip per splat. The vertex shader does EWA splatting
(Zwicker et al., as used by Kerbl et al. 2023):

1. Transform the centre to view space; cull anything at or behind the near plane.
2. Build the projection Jacobian `J` (with the sample point clamped to 1.3x the frustum
   extent, so the affine approximation stays sane off-screen) and the view rotation `W`.
3. `cov2d = J W cov3d W^T J^T`, plus a `+0.3` low-pass on the diagonal so every splat
   covers about a pixel.
4. Eigen-decompose the 2x2 result to get an oriented 3-sigma quad — far fewer wasted
   fragments than an axis-aligned bound on elongated splats.
5. Evaluate SH for the view direction and emit the conic (inverse `cov2d`).

The fragment shader evaluates `alpha = opacity * exp(-0.5 * d^T conic d)`, discards below
`1/255`, and hands the result to whichever mode's output is compiled in. All three splat
pipelines disable depth testing (`depth_write: false`, `compare: Always`) but still declare
a depth-stencil state, because they draw into the same render passes as the mesh path.

### Sorting (mode 1 only)
> `src/splats.rs`

Modes 2, 3 and 4 are order-independent by construction and draw in whatever order the
buffer holds. Mode 1 needs a real back-to-front sort every frame, which is done as a **16-bit
counting sort on a background thread**: view-space depth is quantized to `u16`, so the sort
is a single O(n) counting pass with no comparisons. Measured at ~6 ms for 921k splats.

The render thread never blocks on it — it keeps drawing the previous frame's order until a
new one arrives, and only one request is ever in flight (the worker drops stale requests and
sorts the newest camera). At orbit speeds the one-frame lag is not visible.

## Benchmarking

Two harnesses, measuring different things. Neither substitutes for the other: a technique
can be cheap and wrong, or accurate and unaffordable, and the whole point of mode 4 is that
it trades one for the other.

```
cargo run --release -- splats/rem_v3_clear.ply --headless --frames 200 \
    --screenshot out/shot.png                     # cost
cargo run --release -- splats/rem_v3_clear.ply --quality 16   # loss
scripts/sweep.sh splats/rem_v3_clear.ply out/     # both, over modes + tiles + bins
```

### Cost: `--bench` / `--headless`
> Implemented in `src/bench.rs`; CLI in `src/main.rs`; capture in `Renderer::capture_png`

Interactive frame times cannot be compared: vsync clamps them to the refresh rate, and any
camera movement changes splat overdraw by an order of magnitude. `--bench` pins the camera,
ignores all input and disables vsync; `--headless` goes further and skips winit entirely,
rendering into an offscreen texture and timing a GPU fence (`Renderer::wait_for_gpu`) so it
measures execution rather than submission.

**Prefer `--headless` for anything you intend to compare.** Measured on the same machine
and scene, the windowed path varied by ~2x between back-to-back runs (GPU clock ramp plus
compositor scheduling) while headless reproduces to under 1%.

Flags: `--mode N`, `--frames N`, `--warmup N`, `--dist F` (camera distance as a multiple of
scene radius -- the main overdraw control), `--tile N`, `--bins N`, `--size WxH`,
`--screenshot PATH`. `--screenshot` implies `--headless`, because a swapchain image is not
guaranteed to be copyable. With more than one mode, `.modeN` is inserted before the
extension so a single invocation yields one PNG per mode.

Captured PNGs are un-premultiplied on the way out (the render target holds premultiplied
alpha) and tagged sRGB, so they can be eyeballed against each other. For a *number* rather
than an eyeball, use the quality harness below.

### Loss: `--quality N`
> Implemented in `src/quality.rs`; readback in `Renderer::capture_linear_rgba`;
> reference ordering in `splats::exact_back_to_front_order`

Scores every mode against a ground truth over `N` camera poses drawn from a seeded
SplitMix64 stream, so a run reproduces exactly from its seed and two builds see identical
poses. The reference is **mode 1 driven by a full-precision per-view back-to-front sort**
-- not the interactive 16-bit counting sort, so its quantization stays out of the measured
error.

Flags: `--quality N`, `--seed N`, plus `--mode`, `--tile`, `--bins`, `--size`, `--dist`.
Always headless. Default distance is 2.8x scene radius rather than the cost harness's
1.15x, so the whole scene stays inside the frustum at every orientation and no view is
scored on clipped geometry.

Error is measured in **linear premultiplied RGBA**: premultiplied because that is what the
target holds and un-premultiplying divides by a near-zero alpha over most of a splat frame;
linear because sRGB-space error over-weights dark pixels at these magnitudes. Three
numbers come out:

- **foreground MSE** over pixels either image gives alpha > 1/255, and its PSNR. The
  headline: a large empty background cannot dilute it.
- **full-frame MSE**, where silhouette spill outside the subject shows up.
- **high-frequency residual**, the mean squared *gradient* of the luma error field. Plain
  MSE cannot tell a uniform tint from the same energy scattered as per-pixel grain, and
  grain is the failure mode of stochastic and quantized techniques. This is what caught
  mode 4's hard-snap variant being worse in two ways at once.

Every candidate is rendered `PRIMING_FRAMES` extra times at its pose before the frame that
is scored, because modes 2, 3 and 4 all consume one-frame-old state; without that the score
would partly reflect the *previous* view's camera.

This harness came from `MalekiRe/tiled_gpu_gaussian_splatting`, rewritten onto the headless
path. Its numbers agree with the original's to five significant figures on the same scene,
which is a useful check that neither reimplemented the metric wrong.

## Scene
> Implemented in `src/scene.rs` (scene setup + auto-rotation) and `src/mesh.rs` (geometry generators)

- **6 overlapping semi-transparent quads** at various depths and angles, each a different color with alpha 0.35-0.55 (`src/scene.rs` `new()`)
- **Toggleable meshes**: A transparent cube (`mesh::cube`) and UV-sphere (`mesh::uv_sphere`), slowly auto-rotating (`src/scene.rs` `update()`)
- **Camera**: Orbit camera using spherical coordinates (`src/camera.rs`)
  - `glam::Mat4::perspective_rh()` for projection
  - `glam::Mat4::look_at_rh()` for view
  - Spherical coords: `eye = target + distance * vec3(cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw))`
- **Background**: Dark gray (0.1, 0.1, 0.1) opaque clear, or transparent with `W` (`Renderer::clear_color`)

## Controls
> Implemented in `src/app.rs` `window_event()` handler

- `1` / `2` / `3` / `4`: Switch rendering mode (sets `renderer.mode`)
- `M`: Toggle mesh visibility (toggles `scene.show_meshes`)
- `W`: Toggle the window background between opaque dark grey and transparent. Opaque is
  the default and shows true colours; see the premultiplied-sRGB note above for why.
- `A`: Toggle exact alpha (`1 - exp(-tau)`) vs. the `1 - exp(-accum.a)` approximation
- `C`: (3DGS only) Cycle the render cap: 100% / 50% / 25% / 10% of the scene
- `[` / `]`: (3DGS only) Shrink / grow splats, for dialling overdraw up and down
- `T`: Cycle the histogram tile size (32/16/8/4 px). The single biggest quality knob for
  mode 3 -- see *Tile size is the dominant quality knob* above. Sizes that would exceed the
  device's storage-binding limit are skipped.
- `B`: Cycle the histogram bin count (32/64/128/256) -- **mode 4 only** since the mode 3
  rework fixed its bin count at 16 channel-packed bins. Sets mode 4's CDF *depth*
  resolution, a different axis from what `T` controls. Sizes exceeding the device's
  storage-binding limit are skipped.
- `R`: Reset the camera to its framing of the loaded scene
- `Escape`: Exit
- Mouse drag (left button): Orbit camera (`camera.on_mouse_move`)
- Scroll wheel: Zoom in/out (`camera.on_scroll`)
- Print current mode name to console on switch

## Implementation Order

1. **Scaffolding** - DONE: `Cargo.toml`, `src/main.rs` (event loop + env_logger), `src/app.rs` (ApplicationHandler)
2. **Core types** - DONE: `src/vertex.rs` (all 4 types + vertex buffer layout), `src/camera.rs` (orbit cam with drag/scroll)
3. **Renderer init** - DONE: `src/renderer.rs` (device/surface/queue, depth texture, bind group layouts, all buffer creation)
4. **Mesh generation** - DONE: `src/mesh.rs` (quad, cube, uv_sphere returning `Mesh { vertices, indices }`)
5. **Scene** - DONE: `src/scene.rs` (6 quads + cube + sphere, auto-rotation in `update()`, visibility filtering in `render()`)
6. **Mode 1** - DONE: `src/pipeline/alpha_blend.rs` + `shaders/alpha_blend.wgsl` + `shaders/common.wgsl`
7. **Mode 2** - DONE: `src/pipeline/naive_wboit.rs` + `shaders/wboit_accum.wgsl` + `shaders/wboit_composite.wgsl`
8. **Mode 3** - DONE: `src/pipeline/histogram_wboit.rs` + `shaders/histo_accum.wgsl` + `shaders/histo_composite.wgsl`
9. **Polish** - DONE: Input handling in `app.rs`, resize in `renderer.rs` `resize()`, mode switching, console output
10. **Harnesses** - DONE: `src/bench.rs` (cost), `src/quality.rs` (loss), `scripts/sweep.sh`
11. **Mode 4** - DONE: `src/pipeline/sliced_oit.rs` + `shaders/{front_common,front_surface,splat_front_surface,slice_common,sliced_accum,splat_sliced,sliced_composite}.wgsl`

## Verification

1. `cargo build` compiles clean - VERIFIED (zero warnings)
2. `cargo run` opens a window with transparent geometry
3. Press `1` - alpha blending with visible ordering artifacts
4. Press `2` - WBOIT (order-independent but static weights)
5. Press `3` - histogram-equalized WBOIT (global, better layer separation at clustered depths)
6. Press `4` - quantile-sliced OIT (visibly the closest to mode 1 on a splat scene)
7. Press `M` - toggle meshes on/off
8. Mouse drag to orbit, verify all modes render correctly from different angles
9. Resize window - no crashes, all textures/buffers recreated
10. `--quality 16` on a splat scene ranks the modes 4 < 3 < 2 by foreground MSE
