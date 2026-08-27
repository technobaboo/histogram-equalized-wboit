use crate::bench::{Bench, BenchConfig};
use crate::camera::Camera;
use crate::renderer::{RenderMode, Renderer};
use crate::scene::Scene;
use crate::splats::{Sorter, SplatScene};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

pub struct App {
    window: Option<std::sync::Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    scene: Scene,
    last_frame: std::time::Instant,
    /// Rolling frame-time accumulator, reported once a second.
    frame_count: u32,
    frame_accum: f32,
    scene_uploaded: bool,

    /// Loaded from the command line; `None` means the built-in quad scene.
    splats: Option<SplatScene>,
    sorter: Option<Sorter>,
    splats_uploaded: bool,
    /// Index into `CAP_FRACTIONS`.
    cap: usize,

    /// Present only with `--bench`: pins the camera, ignores input, and exits when done.
    bench: Option<Bench>,
    /// CLI overrides applied once the renderer exists.
    distance_scale: Option<f32>,
    tile_size: Option<u32>,
    bins: Option<u32>,
}

/// Render-cap steps cycled with `C`, as a fraction of the loaded scene.
const CAP_FRACTIONS: [f32; 4] = [1.0, 0.5, 0.25, 0.1];

impl App {
    pub fn new(
        splats: Option<SplatScene>,
        bench: Option<BenchConfig>,
        distance_scale: Option<f32>,
        tile_size: Option<u32>,
        bins: Option<u32>,
    ) -> Self {
        let sorter = splats
            .as_ref()
            .map(|s| Sorter::new(std::sync::Arc::clone(&s.positions)));
        Self {
            window: None,
            renderer: None,
            camera: Camera::new(16.0 / 9.0),
            scene: Scene::new(),
            last_frame: std::time::Instant::now(),
            frame_count: 0,
            frame_accum: 0.0,
            scene_uploaded: false,
            splats,
            sorter,
            splats_uploaded: false,
            cap: 0,
            bench: bench.map(Bench::new),
            distance_scale,
            tile_size,
            bins,
        }
    }

    fn benching(&self) -> bool {
        self.bench.is_some()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let title = if self.splats.is_some() {
            "3DGS WBOIT Demo - 1/2/3 modes, A exact-alpha, W window, T tile, B bins, C cap, R reset"
        } else {
            "WBOIT Demo - 1/2/3 modes, A exact-alpha, W window, T tile, B bins, M meshes"
        };
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_transparent(true);

        let window = std::sync::Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();
        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.viewport = (size.width as f32, size.height as f32);
        if let Some(splats) = &self.splats {
            self.camera.fit_to(splats.center, splats.radius);
        }
        // A fixed distance makes overdraw -- and therefore the whole measurement --
        // reproducible across runs.
        let distance_scale = self
            .distance_scale
            .or_else(|| self.bench.as_ref().map(|b| b.config.distance_scale));
        if let Some(scale) = distance_scale {
            let radius = self.splats.as_ref().map_or(6.0, |s| s.radius);
            self.camera.distance = radius * scale;
        }

        let renderer = Renderer::new(
            Some(window.clone()),
            (size.width, size.height),
            !self.benching(),
        );
        self.renderer = Some(renderer);
        self.window = Some(window);

        if let Some(renderer) = &mut self.renderer {
            if let Some(t) = self
                .tile_size
                .or_else(|| self.bench.as_ref().and_then(|b| b.config.tile_size))
            {
                let (applied, _) = renderer.set_tile_size(t);
                if applied != t {
                    eprintln!("tile size {t} unavailable, using {applied}");
                }
            }
            if let Some(b) = self
                .bins
                .or_else(|| self.bench.as_ref().and_then(|b| b.config.bins))
            {
                let (applied, _) = renderer.set_bin_count(b);
                if applied != b {
                    eprintln!("bin count {b} unavailable, using {applied}");
                }
            }
            if let Some(bench) = &self.bench {
                if let Some(mode) = bench.current_mode() {
                    renderer.mode = mode;
                }
            }
        }

        if let Some(renderer) = &self.renderer {
            println!("Mode: {}", renderer.mode.name());
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // A benchmark that responds to a stray keypress or mouse drag is not a benchmark.
        if self.benching()
            && !matches!(
                event,
                WindowEvent::CloseRequested
                    | WindowEvent::Resized(_)
                    | WindowEvent::RedrawRequested
            )
        {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    self.camera.aspect = new_size.width as f32 / new_size.height.max(1) as f32;
                    self.camera.viewport = (new_size.width as f32, new_size.height.max(1) as f32);
                    renderer.resize(new_size.width, new_size.height);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(renderer) = &mut self.renderer {
                    match &logical_key {
                        Key::Character(c) => match c.as_str() {
                            "1" => {
                                renderer.mode = RenderMode::AlphaBlend;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "2" => {
                                renderer.mode = RenderMode::NaiveWboit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "3" => {
                                renderer.mode = RenderMode::HistogramWboit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "a" | "A" => {
                                renderer.use_revealage = !renderer.use_revealage;
                                println!(
                                    "Exact alpha: {} (computed via {})",
                                    if renderer.use_revealage { "ON" } else { "OFF" },
                                    if renderer.use_revealage {
                                        "1 - exp(-tau) from the optical-depth buffer"
                                    } else {
                                        "1 - exp(-accum.a) approximation"
                                    }
                                );
                            }
                            "r" | "R" => {
                                self.camera.reset();
                                println!("Camera reset");
                            }
                            "m" | "M" => {
                                self.scene.show_meshes = !self.scene.show_meshes;
                                println!(
                                    "Meshes: {}",
                                    if self.scene.show_meshes { "ON" } else { "OFF" }
                                );
                            }
                            "w" | "W" => {
                                renderer.opaque_background = !renderer.opaque_background;
                                println!(
                                    "Background: {}",
                                    if renderer.opaque_background {
                                        "opaque dark grey (true colours; sidesteps the \
premultiplied-sRGB mismatch with the compositor)"
                                    } else {
                                        "transparent (composited by the window manager)"
                                    }
                                );
                            }
                            "b" | "B" => {
                                let (bins, mb) = renderer.cycle_bin_count();
                                println!(
                                    "Histogram bins: {bins} ({mb:.1} MB) \
- more bins separate layers that sit close together in depth"
                                );
                            }
                            "t" | "T" => {
                                let (tile, mb) = renderer.cycle_tile_size();
                                println!(
                                    "Histogram tile: {tile}x{tile} px ({mb:.1} MB) \
- smaller tiles = less background bleed-through in mode 3"
                                );
                            }
                            "c" | "C" if renderer.has_splats() => {
                                self.cap = (self.cap + 1) % CAP_FRACTIONS.len();
                                if let Some((drawn, total)) =
                                    renderer.set_splat_fraction(CAP_FRACTIONS[self.cap])
                                {
                                    println!(
                                        "Render cap: {:.0}% ({drawn} / {total} splats)",
                                        CAP_FRACTIONS[self.cap] * 100.0
                                    );
                                }
                            }
                            "[" | "]" => {
                                let factor = if c.as_str() == "[" { 1.0 / 1.15 } else { 1.15 };
                                if let Some(scale) = renderer.adjust_splat_scale(factor) {
                                    println!("Splat size: {scale:.2}x");
                                }
                            }
                            "o" | "O" => {
                                self.scene.force_opaque = !self.scene.force_opaque;
                                println!(
                                    "Opaque: {}",
                                    if self.scene.force_opaque { "ON" } else { "OFF" }
                                );
                            }
                            _ => {}
                        },
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    self.camera.dragging = state == ElementState::Pressed;
                    if !self.camera.dragging {
                        self.camera.last_mouse = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.camera.on_mouse_move(position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                self.camera.on_scroll(scroll);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                self.frame_count += 1;
                self.frame_accum += dt;
                if self.frame_accum >= 1.0 && !self.benching() {
                    let ms = self.frame_accum * 1000.0 / self.frame_count as f32;
                    println!(
                        "{:.2} ms/frame ({:.0} fps) [{}]",
                        ms,
                        1000.0 / ms,
                        self.renderer.as_ref().map_or("", |r| r.mode.name()),
                    );
                    self.frame_count = 0;
                    self.frame_accum = 0.0;
                }

                self.scene.update(dt);

                if let Some(renderer) = &mut self.renderer {
                    if !self.scene_uploaded {
                        renderer.upload_scene(&self.scene);
                        self.scene_uploaded = true;
                    }
                    if let Some(splats) = &self.splats
                        && !self.splats_uploaded
                    {
                        renderer.upload_splats(splats);
                        self.splats_uploaded = true;
                    }

                    // Keep the depth order fresh for mode 1. The sort runs off-thread, so
                    // this never blocks; we just pick up whatever it has finished.
                    if let Some(sorter) = &mut self.sorter {
                        if let Some(order) = sorter.poll() {
                            renderer.upload_splat_order(&order);
                        }
                        if renderer.mode == RenderMode::AlphaBlend {
                            sorter.request(self.camera.forward(), renderer.splat_draw_count());
                        }
                    }

                    renderer.render(&self.camera, &self.scene);

                    if let Some(bench) = &mut self.bench {
                        if bench.frame(dt) {
                            event_loop.exit();
                        } else if let Some(mode) = bench.current_mode() {
                            renderer.mode = mode;
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
