//! Mode 4: quantile-sliced OIT.
//!
//! Reuses mode 3's histogram, CDF volume and compute pass wholesale -- including its bind
//! group layout, which declares one binding (the previous frame's optical depth) that
//! these shaders do not read. What changes is only what the CDF is used *for*: mode 3
//! turns it into a per-fragment weight, mode 4 turns it into a slab index. See
//! `shaders/slice_common.wgsl`.

use crate::vertex::Vertex;

/// Ordered slabs a fragment can land in. Four is the point where the extra bandwidth stops
/// paying for itself on the scenes here; it is also what fits one MRT pass comfortably.
/// Changing it means changing `SLICE_COUNT` and the switch ladders in `slice_common.wgsl`
/// and the loop in `sliced_composite.wgsl`.
pub const SLICE_COUNT: usize = 4;

/// Slab format. `Rgba16Float` for the same reason the accum target is: blendable without
/// the `float32-blendable` feature, and `tau` stays linear so f16 resolves it finely.
pub const SLICE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Front-surface prepass target: `(color.rgb, normalized_z)`. Cleared to depth 1, which
/// reads back as "nothing in front of anything" and makes the correction inert.
pub const FRONT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct SlicedOitPipeline {
    pub front_pipeline: wgpu::RenderPipeline,
    pub accum_pipeline: wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,
    pub composite_bgl: wgpu::BindGroupLayout,
}

/// The prepass is the one place in the whole demo that wants a real depth test: it is
/// looking for the *nearest* qualifying fragment per pixel, which is exactly what a
/// depth buffer resolves for free.
pub fn front_depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: Default::default(),
        bias: Default::default(),
    }
}

pub fn front_target() -> [Option<wgpu::ColorTargetState>; 1] {
    [Some(wgpu::ColorTargetState {
        format: FRONT_FORMAT,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })]
}

/// Additive blend into every slab: within a slab, accumulation is order-independent.
pub fn slice_targets() -> [Option<wgpu::ColorTargetState>; SLICE_COUNT] {
    std::array::from_fn(|_| {
        Some(wgpu::ColorTargetState {
            format: SLICE_FORMAT,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            write_mask: wgpu::ColorWrites::ALL,
        })
    })
}

impl SlicedOitPipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        object_bgl: &wgpu::BindGroupLayout,
        histo_accum_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sliced composite bgl"),
            entries: &std::array::from_fn::<_, SLICE_COUNT, _>(|i| {
                wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }
            }),
        });

        let accum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sliced accum layout"),
            bind_group_layouts: &[camera_bgl, object_bgl, histo_accum_bgl],
            immediate_size: 0,
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sliced composite layout"),
            bind_group_layouts: &[&composite_bgl],
            immediate_size: 0,
        });

        let accum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sliced accum shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}\n{}",
                    include_str!("../../shaders/common.wgsl"),
                    include_str!("../../shaders/front_common.wgsl"),
                    include_str!("../../shaders/slice_common.wgsl"),
                    include_str!("../../shaders/sliced_accum.wgsl"),
                )
                .into(),
            ),
        });

        let front_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("front surface shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    include_str!("../../shaders/common.wgsl"),
                    include_str!("../../shaders/front_common.wgsl"),
                    include_str!("../../shaders/front_surface.wgsl"),
                )
                .into(),
            ),
        });
        let front_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("front surface layout"),
            bind_group_layouts: &[camera_bgl, object_bgl],
            immediate_size: 0,
        });
        let front_targets = front_target();
        let front_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("front surface pipeline"),
            layout: Some(&front_layout),
            vertex: wgpu::VertexState {
                module: &front_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &front_shader,
                entry_point: Some("fs_main"),
                targets: &front_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(front_depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let targets = slice_targets();
        let accum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sliced accum pipeline"),
            layout: Some(&accum_layout),
            vertex: wgpu::VertexState {
                module: &accum_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &accum_shader,
                entry_point: Some("fs_main"),
                targets: &targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sliced composite shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sliced_composite.wgsl").into(),
            ),
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sliced composite pipeline"),
            layout: Some(&composite_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            front_pipeline,
            accum_pipeline,
            composite_pipeline,
            composite_bgl,
        }
    }
}
