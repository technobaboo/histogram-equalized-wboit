mod app;
mod bench;
mod camera;
mod mesh;
mod pipeline;
mod ply;
mod quality;
mod renderer;
mod scene;
mod splats;
mod vertex;

use std::path::PathBuf;

const USAGE: &str = "\
usage: wboit-demo [PLY] [options]

  PLY                 Gaussian splat scene to load; omit for the built-in quad scene

  --quality N         run the image-quality benchmark over N random views and exit:
                      scores every mode against mode 1 with an exact per-view sort.
                      Always headless.
  --seed N            RNG seed for --quality view selection (default 1)

  --bench             run the benchmark and exit (pins the camera, ignores input,
                      disables vsync) instead of opening an interactive window
  --headless          render with no window at all and exit. Implied by --screenshot.
                      Times a GPU fence rather than presentation, so results are far
                      more stable than the windowed path.
  --screenshot PATH   write a PNG of the last frame. With more than one mode, `.modeN`
                      is inserted before the extension.
  --size WxH          headless render size (default 1280x720)
  --mode N            benchmark only mode N (1 alpha blend, 2 naive, 3 histogram,
                      4 quantile-sliced)
  --frames N          measured frames per mode (default 300)
  --warmup N          discarded frames per mode (default 60)
  --dist F            camera distance as a multiple of scene radius (default 1.15 in
                      bench, 2.8 in quality, 2.5 interactive). Lower = more overdraw.
  --meshes            draw the built-in quad/mesh scene together with the loaded PLY
                      (same as pressing G interactively; no effect without a PLY)
  --tile N            histogram tile size in pixels (32/16/8/4)
  --bins N            histogram depth bins (32/64/128/256)
  -h, --help          show this
";

struct Args {
    path: Option<PathBuf>,
    bench: Option<bench::BenchConfig>,
    quality: Option<quality::QualityConfig>,
    distance_scale: Option<f32>,
    tile_size: Option<u32>,
    bins: Option<u32>,
    mesh_overlay: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut path = None;
    let mut bench_on = false;
    let mut cfg = bench::BenchConfig::default();
    let mut quality_views = None;
    let mut seed = 1u64;
    let mut mode = None;
    let mut distance_scale = None;
    let mut tile_size = None;
    let mut bins = None;
    let mut headless = false;
    let mut mesh_overlay = false;

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        // Every flag below that takes a value consumes the next argument.
        let value = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_else(|| {
                eprintln!("{arg} needs a value\n\n{USAGE}");
                std::process::exit(2);
            })
        };
        match arg {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--bench" => bench_on = true,
            "--quality" => quality_views = value(&mut i).parse::<u32>().ok().or(Some(16)),
            "--seed" => seed = value(&mut i).parse().unwrap_or(seed),
            "--headless" => headless = true,
            "--meshes" => mesh_overlay = true,
            "--screenshot" => cfg.screenshot = Some(PathBuf::from(value(&mut i))),
            "--size" => {
                let v = value(&mut i);
                let (w, h) = v
                    .split_once(['x', 'X'])
                    .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
                    .unwrap_or_else(|| {
                        eprintln!("--size wants WxH, e.g. 1920x1080");
                        std::process::exit(2);
                    });
                cfg.width = w;
                cfg.height = h;
            }
            "--mode" => mode = value(&mut i).parse::<u32>().ok(),
            "--frames" => cfg.frames = value(&mut i).parse().unwrap_or(cfg.frames),
            "--warmup" => cfg.warmup = value(&mut i).parse().unwrap_or(cfg.warmup),
            "--dist" => distance_scale = value(&mut i).parse::<f32>().ok(),
            "--tile" => tile_size = value(&mut i).parse::<u32>().ok(),
            "--bins" => bins = value(&mut i).parse::<u32>().ok(),
            other if other.starts_with('-') => {
                eprintln!("unknown option {other}\n\n{USAGE}");
                std::process::exit(2);
            }
            other => path = Some(PathBuf::from(other)),
        }
        i += 1;
    }

    if let Some(m) = mode {
        cfg.modes.retain(|r| *r as u32 == m);
        if cfg.modes.is_empty() {
            eprintln!("--mode must be one of 1..{}", renderer::RenderMode::ALL.len());
            std::process::exit(2);
        }
    }
    if let Some(d) = distance_scale {
        cfg.distance_scale = d;
    }
    cfg.tile_size = tile_size;
    cfg.bins = bins;
    cfg.mesh_overlay = mesh_overlay;
    // A screenshot needs a copyable target, which only the offscreen path guarantees.
    cfg.headless = headless || cfg.screenshot.is_some();

    let quality = quality_views.map(|views| {
        let mut q = quality::QualityConfig {
            views,
            seed,
            modes: cfg.modes.clone(),
            tile_size,
            bins,
            width: cfg.width,
            height: cfg.height,
            mesh_overlay,
            ..Default::default()
        };
        if let Some(d) = distance_scale {
            q.distance_scale = d;
        }
        q
    });

    let screenshot_requested = cfg.screenshot.is_some();
    Args {
        path,
        bench: (bench_on || headless || screenshot_requested).then_some(cfg),
        quality,
        distance_scale,
        tile_size,
        bins,
        mesh_overlay,
    }
}

fn main() {
    env_logger::init();
    let args = parse_args();

    // With a PLY argument the demo renders that Gaussian splat scene through all three
    // transparency modes; without one it falls back to the built-in quad/mesh scene.
    let splats = match &args.path {
        Some(path) => {
            let started = std::time::Instant::now();
            let data = match ply::load(path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Failed to load {}: {e}", path.display());
                    std::process::exit(1);
                }
            };
            println!(
                "Loaded {} splats from {} ({} SH bands) in {:.2}s",
                data.len(),
                path.display(),
                data.sh_degree,
                started.elapsed().as_secs_f32(),
            );
            let scene = splats::SplatScene::from_ply(data);
            println!(
                "Scene centre {:?}, radius {:.2}",
                scene.center.to_array(),
                scene.radius
            );
            Some(scene)
        }
        None => None,
    };

    // Quality scoring is always offscreen: it needs a copyable target to read back.
    if let Some(cfg) = args.quality {
        if let Err(e) = quality::run(cfg, splats) {
            eprintln!("quality run failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    // Headless never touches winit: no window, no compositor, no vsync.
    if let Some(cfg) = args.bench.as_ref().filter(|c| c.headless) {
        if let Err(e) = bench::run_headless(cfg.clone(), splats, args.distance_scale) {
            eprintln!("headless run failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = app::App::new(
        splats,
        args.bench,
        args.distance_scale,
        args.tile_size,
        args.bins,
        args.mesh_overlay,
    );
    event_loop.run_app(&mut app).unwrap();
}
