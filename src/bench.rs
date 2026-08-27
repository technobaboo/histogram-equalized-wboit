//! Reproducible benchmark mode.
//!
//! Interactive frame times are useless for comparing techniques: vsync clamps them to the
//! refresh rate, and any camera movement changes overdraw by an order of magnitude. This
//! runs each mode over a fixed frame count from a pinned camera with input ignored and
//! vsync off, then reports a summary and exits.
//!
//! What it measures is wall-clock time between redraws, i.e. end-to-end throughput. With
//! vsync off and a GPU-bound scene that tracks GPU time closely, but it is not a
//! per-pass GPU timing -- attributing cost to individual passes still needs either
//! timestamp queries or an ablation build.

use crate::renderer::RenderMode;

#[derive(Clone)]
pub struct BenchConfig {
    pub warmup: u32,
    pub frames: u32,
    pub modes: Vec<RenderMode>,
    /// Camera distance as a multiple of scene radius. Lower means the subject fills more
    /// of the viewport, which is the dominant control on splat overdraw.
    pub distance_scale: f32,
    pub tile_size: Option<u32>,
    /// Draw the built-in mesh scene alongside a loaded splat scene.
    pub mesh_overlay: bool,
    pub bins: Option<u32>,
    /// Render without a window and exit; required for `screenshot`.
    pub headless: bool,
    /// Where to write a PNG of the final frame of each mode. With more than one mode,
    /// `.modeN` is inserted before the extension.
    pub screenshot: Option<std::path::PathBuf>,
    pub width: u32,
    pub height: u32,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup: 60,
            frames: 300,
            modes: RenderMode::ALL.to_vec(),
            distance_scale: 1.15,
            tile_size: None,
            mesh_overlay: false,
            bins: None,
            headless: false,
            screenshot: None,
            width: 1280,
            height: 720,
        }
    }
}

impl BenchConfig {
    /// Output path for one mode, disambiguated when several modes are being run.
    pub fn screenshot_path(&self, mode: RenderMode) -> Option<std::path::PathBuf> {
        let base = self.screenshot.as_ref()?;
        if self.modes.len() <= 1 {
            return Some(base.clone());
        }
        let stem = base.file_stem().unwrap_or_default().to_string_lossy();
        let ext = base.extension().map(|e| e.to_string_lossy().to_string());
        let name = match ext {
            Some(ext) => format!("{stem}.mode{}.{ext}", mode as u32),
            None => format!("{stem}.mode{}", mode as u32),
        };
        Some(base.with_file_name(name))
    }
}

struct ModeResult {
    mode: RenderMode,
    mean_ms: f32,
    median_ms: f32,
    p95_ms: f32,
}

pub struct Bench {
    pub config: BenchConfig,
    mode_index: usize,
    seen: u32,
    pub(crate) samples: Vec<f32>,
    results: Vec<ModeResult>,
}

impl Bench {
    pub fn new(config: BenchConfig) -> Self {
        Self {
            config,
            mode_index: 0,
            seen: 0,
            samples: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn current_mode(&self) -> Option<RenderMode> {
        self.config.modes.get(self.mode_index).copied()
    }

    /// Record one frame. Returns true once every mode has been measured and the summary
    /// has been printed, at which point the caller should exit.
    pub fn frame(&mut self, dt: f32) -> bool {
        self.seen += 1;
        // Discard the warm-up window: shader compilation, buffer allocation and GPU clock
        // ramp all land in the first frames and would otherwise dominate the mean.
        if self.seen > self.config.warmup {
            self.samples.push(dt * 1000.0);
        }
        if self.seen < self.config.warmup + self.config.frames {
            return false;
        }

        let mode = self.config.modes[self.mode_index];
        self.finish_mode(mode);
        self.seen = 0;
        self.mode_index += 1;

        if self.mode_index < self.config.modes.len() {
            return false;
        }
        self.report();
        true
    }

    /// Fold the collected samples for one mode into a result row.
    pub(crate) fn finish_mode(&mut self, mode: RenderMode) {
        if self.samples.is_empty() {
            return;
        }
        self.samples
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = self.samples.len();
        self.results.push(ModeResult {
            mode,
            mean_ms: self.samples.iter().sum::<f32>() / n as f32,
            median_ms: self.samples[n / 2],
            p95_ms: self.samples[(n * 95 / 100).min(n - 1)],
        });
        self.samples.clear();
    }

    pub(crate) fn report(&self) {
        println!("\n--- benchmark ---");
        println!(
            "{} warm-up + {} measured frames per mode, camera at {:.2}x scene radius",
            self.config.warmup, self.config.frames, self.config.distance_scale
        );
        println!(
            "\n{:<34} {:>9} {:>9} {:>9} {:>8}",
            "mode", "mean ms", "med ms", "p95 ms", "fps"
        );
        for r in &self.results {
            println!(
                "{:<34} {:>9.2} {:>9.2} {:>9.2} {:>8.0}",
                r.mode.name(),
                r.mean_ms,
                r.median_ms,
                r.p95_ms,
                1000.0 / r.median_ms.max(1e-6),
            );
        }

        // Deltas are the actually interesting number: what each technique costs over the
        // same rasterization work.
        if let Some(base) = self
            .results
            .iter()
            .find(|r| r.mode == RenderMode::NaiveWboit)
        {
            if let Some(histo) = self
                .results
                .iter()
                .find(|r| r.mode == RenderMode::HistogramWboit)
            {
                let delta = histo.median_ms - base.median_ms;
                println!(
                    "\nhistogram-equalized costs {:+.2} ms/frame over naive WBOIT ({:+.0}%)",
                    delta,
                    100.0 * delta / base.median_ms.max(1e-6),
                );
            }
        }
        println!();
    }
}

/// Run the whole benchmark with no window: create a headless renderer, render each mode
/// for warm-up + measured frames, optionally capture a PNG, and print the summary.
///
/// Timing here brackets a GPU fence, so it measures execution rather than submission, and
/// there is no compositor or vsync in the loop at all.
pub fn run_headless(
    config: BenchConfig,
    splats: Option<crate::splats::SplatScene>,
    distance_scale: Option<f32>,
) -> Result<(), String> {
    use crate::camera::Camera;
    use crate::renderer::Renderer;
    use crate::scene::Scene;

    let mut renderer = Renderer::new(None, (config.width, config.height), false);
    let scene = Scene::new();
    renderer.upload_scene(&scene);
    renderer.mesh_overlay = config.mesh_overlay;

    let mut camera = Camera::new(config.width as f32 / config.height as f32);
    camera.viewport = (config.width as f32, config.height as f32);
    if let Some(splats) = &splats {
        camera.fit_to(splats.center, splats.radius);
        renderer.upload_splats(splats);
    }
    let radius = splats.as_ref().map_or(6.0, |s| s.radius);
    camera.distance = radius * distance_scale.unwrap_or(config.distance_scale);

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

    let mut bench = Bench::new(config);
    let total = bench.config.warmup + bench.config.frames;

    for mode in bench.config.modes.clone() {
        renderer.mode = mode;
        for frame in 0..total {
            let started = std::time::Instant::now();
            renderer.render(&camera, &scene);
            renderer.wait_for_gpu();
            let dt = started.elapsed().as_secs_f32();
            // Warm-up frames also let the one-frame temporal feedback in modes 2 and 3
            // converge before anything is measured or captured.
            if frame >= bench.config.warmup {
                bench.samples.push(dt * 1000.0);
            }
        }
        bench.finish_mode(mode);

        if let Some(path) = bench.config.screenshot_path(mode) {
            renderer.capture_png(&path)?;
            println!("wrote {}", path.display());
        }
    }

    bench.report();
    Ok(())
}
