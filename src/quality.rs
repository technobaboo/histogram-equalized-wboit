//! Reproducible *quality* benchmark: how far each mode's image is from ground truth.
//!
//! `bench.rs` answers "what does this technique cost"; this answers "what does it lose".
//! The reference is mode 1 driven by a full-precision per-view back-to-front sort, which
//! for splats is the exact front-to-back integral up to rasterization order within a
//! single splat -- so any difference from it is the technique's own error, not the
//! scene's.
//!
//! Views are drawn from a fixed SplitMix64 stream, so a run is reproducible from its seed
//! and two runs of different builds see exactly the same poses.
//!
//! Error is measured in **linear premultiplied RGBA**. Premultiplied because that is what
//! the render target holds and un-premultiplying divides by a near-zero alpha over most
//! of a splat frame; linear because sRGB-space error weights dark pixels far more heavily
//! than the eye does at these magnitudes.

use crate::camera::Camera;
use crate::renderer::{RenderMode, Renderer};
use crate::scene::Scene;
use crate::splats::{SplatScene, exact_back_to_front_order};

/// Frames rendered at a pose before the one that gets scored, to let the one-frame
/// temporal feedback in modes 2 and 3 settle on this camera rather than the last one.
const PRIMING_FRAMES: u32 = 2;

#[derive(Clone)]
pub struct QualityConfig {
    pub views: u32,
    pub seed: u64,
    /// Modes to score. Mode 1 is always the reference and is skipped if listed.
    pub modes: Vec<RenderMode>,
    pub tile_size: Option<u32>,
    /// Draw the built-in mesh scene alongside a loaded splat scene.
    pub mesh_overlay: bool,
    pub bins: Option<u32>,
    /// Camera distance as a multiple of scene radius; each view jitters up from here.
    pub distance_scale: f32,
    pub width: u32,
    pub height: u32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            views: 16,
            seed: 1,
            modes: RenderMode::ALL.to_vec(),
            tile_size: None,
            mesh_overlay: false,
            bins: None,
            // Far enough out that the whole scene stays inside the frustum at every
            // orientation, so no view is scored on clipped geometry.
            distance_scale: 2.8,
            width: 1280,
            height: 720,
        }
    }
}

/// Per-view error for one mode.
#[derive(Default)]
struct Samples {
    foreground: Vec<f64>,
    full_frame: Vec<f64>,
    sparkle: Vec<f64>,
}

/// The three losses for one image pair.
struct Losses {
    /// Mean squared error over pixels either image considers covered. The headline
    /// number: it does not let a large empty background dilute the score.
    foreground: f64,
    /// Mean squared error over every pixel. Lower than `foreground` by construction;
    /// worth watching because it is where silhouette spill outside the subject shows up.
    full_frame: f64,
    /// Mean squared *gradient* of the luma error field. Pure MSE cannot tell a uniform
    /// tint from the same energy scattered as per-pixel grain, and grain is what
    /// stochastic and quantized techniques actually fail with.
    sparkle: f64,
}

pub fn run(config: QualityConfig, splats: Option<SplatScene>) -> Result<(), String> {
    let mut renderer = Renderer::new(None, (config.width, config.height), false);
    let scene = Scene::new();
    renderer.upload_scene(&scene);
    renderer.mesh_overlay = config.mesh_overlay;
    // Alpha has to survive to the readback for the foreground mask to mean anything, so
    // the frame is composited over a transparent clear rather than the opaque grey.
    renderer.opaque_background = false;

    let mut camera = Camera::new(config.width as f32 / config.height as f32);
    camera.viewport = (config.width as f32, config.height as f32);
    if let Some(splats) = &splats {
        camera.fit_to(splats.center, splats.radius);
        renderer.upload_splats(splats);
    }
    let radius = splats.as_ref().map_or(6.0, |s| s.radius);

    if let Some(t) = config.tile_size {
        let (applied, _) = renderer.set_tile_size(t);
        if applied != t {
            eprintln!("tile size {t} unavailable, using {applied}");
        }
    }
    if let Some(b) = config.bins {
        let (applied, _) = renderer.set_bin_count(b);
        if applied != b {
            eprintln!("bin count {b} unavailable, using {applied}");
        }
    }

    let modes: Vec<RenderMode> = config
        .modes
        .iter()
        .copied()
        .filter(|m| *m != RenderMode::AlphaBlend)
        .collect();
    if modes.is_empty() {
        return Err("no candidate modes: mode 1 is the reference, not a candidate".into());
    }

    let views = config.views.max(1);
    let mut results: Vec<(RenderMode, Samples)> =
        modes.iter().map(|m| (*m, Samples::default())).collect();
    let mut rng = SplitMix64::new(config.seed);

    println!(
        "quality: {views} views, seed {}, {}x{}, camera at {:.2}-{:.2}x scene radius",
        config.seed,
        config.width,
        config.height,
        config.distance_scale,
        config.distance_scale * 1.25,
    );
    println!("reference: mode 1 with a full-precision per-view back-to-front sort");

    for view in 0..views {
        // Uniform on the sphere: yaw uniform, pitch as asin of a uniform, so poses are
        // not bunched at the poles.
        camera.yaw = rng.unit() * std::f32::consts::TAU;
        camera.pitch = (2.0 * rng.unit() - 1.0).asin();
        camera.distance = radius * config.distance_scale * (1.0 + 0.25 * rng.unit());

        if let Some(splats) = &splats {
            let order = exact_back_to_front_order(
                &splats.positions,
                camera.forward(),
                renderer.splat_draw_count(),
            );
            renderer.upload_splat_order(&order);
        }
        renderer.mode = RenderMode::AlphaBlend;
        renderer.render(&camera, &scene);
        renderer.wait_for_gpu();
        let reference = renderer.capture_linear_rgba()?;

        for (mode, samples) in &mut results {
            renderer.mode = *mode;
            // Modes 2 and 3 read one-frame-old optical depth, and mode 3 a one-frame-old
            // CDF. Prime both at this exact pose so the score reflects the technique and
            // not the previous view's camera.
            for _ in 0..PRIMING_FRAMES {
                renderer.render(&camera, &scene);
            }
            renderer.render(&camera, &scene);
            renderer.wait_for_gpu();
            let candidate = renderer.capture_linear_rgba()?;

            let losses = image_losses(
                &reference,
                &candidate,
                config.width as usize,
                config.height as usize,
            );
            samples.foreground.push(losses.foreground);
            samples.full_frame.push(losses.full_frame);
            samples.sparkle.push(losses.sparkle);
        }
        if views > 1 {
            // Progress goes to stderr so a piped stdout stays a clean table.
            eprint!("\r  view {}/{views}   ", view + 1);
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }
    }
    eprintln!("\r{:32}\r", "");

    report(&results);
    Ok(())
}

fn report(results: &[(RenderMode, Samples)]) {
    println!("--- quality ---");
    println!("linear premultiplied-RGBA MSE against the sorted reference, lower is better\n");
    println!(
        "{:<34} {:>11} {:>11} {:>11} {:>9}",
        "mode", "fg MSE", "worst", "std dev", "fg PSNR"
    );
    for (mode, s) in results {
        let mean = mean(&s.foreground);
        let worst = s.foreground.iter().copied().fold(0.0, f64::max);
        let psnr = if mean > 0.0 {
            -10.0 * mean.log10()
        } else {
            f64::INFINITY
        };
        println!(
            "{:<34} {:>11.4e} {:>11.4e} {:>11.4e} {:>6.2} dB",
            mode.name(),
            mean,
            worst,
            variance(&s.foreground, mean).sqrt(),
            psnr,
        );
    }

    println!("\nfg MSE covers pixels either image gives alpha > 1/255; PSNR is derived from it.");
    println!("full-frame MSE (catches spill outside the subject's silhouette):");
    for (mode, s) in results {
        println!("  {:<32} {:.4e}", mode.name(), mean(&s.full_frame));
    }
    println!("high-frequency residual (sparkle and grain, which flat MSE hides):");
    for (mode, s) in results {
        println!("  {:<32} {:.4e}", mode.name(), mean(&s.sparkle));
    }

    // The comparison anyone actually wants: how much of naive WBOIT's error each
    // technique removes.
    if let Some((_, base)) = results.iter().find(|(m, _)| *m == RenderMode::NaiveWboit) {
        let base_mean = mean(&base.foreground);
        if base_mean > 0.0 {
            println!("\nfg MSE relative to naive WBOIT:");
            for (mode, s) in results {
                if *mode == RenderMode::NaiveWboit {
                    continue;
                }
                let m = mean(&s.foreground);
                println!(
                    "  {:<32} {:.2}x lower ({:+.2} dB)",
                    mode.name(),
                    base_mean / m.max(f64::MIN_POSITIVE),
                    10.0 * (base_mean / m.max(f64::MIN_POSITIVE)).log10(),
                );
            }
        }
    }
    println!();
}

fn image_losses(
    reference: &[[f32; 4]],
    candidate: &[[f32; 4]],
    width: usize,
    height: usize,
) -> Losses {
    assert_eq!(reference.len(), candidate.len());
    assert_eq!(reference.len(), width * height);

    let mut foreground_sum = 0.0f64;
    let mut foreground_channels = 0usize;
    let mut full_sum = 0.0f64;
    for (r, c) in reference.iter().zip(candidate) {
        let mut pixel = 0.0f64;
        for ch in 0..4 {
            let d = (r[ch] - c[ch]) as f64;
            pixel += d * d;
        }
        full_sum += pixel;
        if r[3] > 1.0 / 255.0 || c[3] > 1.0 / 255.0 {
            foreground_sum += pixel;
            foreground_channels += 4;
        }
    }

    // Collapse the error to one luma-weighted scalar per pixel, then measure how fast it
    // varies: a smooth error field is a tint, a noisy one is grain.
    let error = |i: usize| {
        let (r, c) = (reference[i], candidate[i]);
        0.2126 * (r[0] - c[0]) as f64
            + 0.7152 * (r[1] - c[1]) as f64
            + 0.0722 * (r[2] - c[2]) as f64
            + 0.5 * (r[3] - c[3]) as f64
    };
    let mut sparkle_sum = 0.0f64;
    let mut edges = 0usize;
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let e = error(i);
            if x + 1 < width {
                let d = e - error(i + 1);
                sparkle_sum += d * d;
                edges += 1;
            }
            if y + 1 < height {
                let d = e - error(i + width);
                sparkle_sum += d * d;
                edges += 1;
            }
        }
    }

    Losses {
        foreground: foreground_sum / foreground_channels.max(1) as f64,
        full_frame: full_sum / (reference.len().max(1) * 4) as f64,
        sparkle: sparkle_sum / edges.max(1) as f64,
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn variance(values: &[f64], mean: f64) -> f64 {
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len().max(1) as f64
}

/// SplitMix64. Small, seedable and stateless enough to reproduce a pose set exactly from
/// its seed, which is the only property the harness needs from an RNG.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next value in [0, 1), from the top 24 bits so every result is exactly
    /// representable in f32.
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut v = self.0;
        v = (v ^ (v >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        v ^= v >> 31;
        ((v >> 40) as f32) / ((1u32 << 24) as f32)
    }
}
