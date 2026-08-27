use crate::vertex::Vertex;

/// Number of tile-resolution RGBA16F textures holding the histogram and the CDF. Four
/// textures x four channels = 16 depth bins, which is WebGPU's default 32-byte
/// maxColorAttachmentBytesPerSample exactly -- the mobile-safe MRT budget.
pub const HISTO_TEXTURES: usize = 4;
pub const HISTO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn additive_target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format,
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
}

/// The four additive histogram targets of the binning pass.
pub fn binning_targets() -> [Option<wgpu::ColorTargetState>; HISTO_TEXTURES] {
    std::array::from_fn(|_| additive_target(HISTO_FORMAT))
}

pub struct HistogramWboitPipeline {
    pub accum_pipeline: wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,
    /// Mesh half of the binning pass: the scene rasterized at one pixel per tile,
    /// blending optical depth into the 16-channel tile histogram.
    pub binning_pipeline: wgpu::RenderPipeline,
    /// Fullscreen-at-tile-resolution prefix sum: this pixel's four histogram texels in,
    /// this pixel's four CDF texels out. Replaces the compute dispatch for mode 3.
    pub cdf_resolve_pipeline: wgpu::RenderPipeline,
    /// Mode 4's CDF build (compute over the atomic histogram buffer into the 3D CDF
    /// volume). Mode 3 no longer uses it.
    pub cdf_build_pipeline: wgpu::ComputePipeline,
    /// Group 2 of mode 3's accumulation: params, four CDF textures, sampler, prev
    /// optical depth.
    pub histo_accum_bgl: wgpu::BindGroupLayout,
    /// Group 2 of mode 4's pipelines: the atomic histogram buffer + 3D CDF volume layout
    /// that mode 3 used before the binning pass replaced them.
    pub slice_accum_bgl: wgpu::BindGroupLayout,
    /// Group 2 of the binning pass: just the histogram params uniform.
    pub binning_params_bgl: wgpu::BindGroupLayout,
    /// Group 0 of the CDF resolve: the four histogram textures.
    pub hist_read_bgl: wgpu::BindGroupLayout,
    pub histo_composite_tex_bgl: wgpu::BindGroupLayout,
    pub cdf_build_bgl: wgpu::BindGroupLayout,
    pub flag_bgl: wgpu::BindGroupLayout,
}

impl HistogramWboitPipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        object_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        // Mode 3 accumulation, group 2: b0 params, b1..b4 CDF textures, b5 sampler,
        // b6 previous frame's optical depth.
        let histo_accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histo accum bgl"),
            entries: &[
                uniform_entry(0),
                texture_entry(1, true),
                texture_entry(2, true),
                texture_entry(3, true),
                texture_entry(4, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_entry(6, false),
            ],
        });

        // Binning pass, group 2: params only. The pass writes through attachments, so it
        // binds nothing else. Vertex visibility because binning_clip() aligns the clip
        // position to the tile grid in the vertex stage.
        let binning_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("binning params bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // CDF resolve, group 0: the four histogram textures, read at own pixel only.
        let hist_read_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hist read bgl"),
            entries: &[
                texture_entry(0, false),
                texture_entry(1, false),
                texture_entry(2, false),
                texture_entry(3, false),
            ],
        });

        // Mode 4's accumulation layout (formerly mode 3's as well): b0 histogram storage
        // (rw atomic), b1 CDF 3D texture, b2 sampler, b3 params, b4 front surface.
        let slice_accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slice accum bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform_entry(3),
                texture_entry(4, false),
            ],
        });

        let histo_composite_tex_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("histo composite tex bgl"),
                entries: &[texture_entry(0, false), texture_entry(1, false)],
            });

        // CDF build compute (mode 4): histogram rw, CDF 3D storage texture write, params
        let cdf_build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cdf build bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let flag_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histo flag bgl"),
            entries: &[uniform_entry(0)],
        });

        let accum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histo_accum pipeline layout"),
            bind_group_layouts: &[camera_bgl, object_bgl, &histo_accum_bgl],
            immediate_size: 0,
        });

        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histo composite pipeline layout"),
            bind_group_layouts: &[&histo_composite_tex_bgl, &flag_bgl],
            immediate_size: 0,
        });

        let binning_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histo binning pipeline layout"),
            bind_group_layouts: &[camera_bgl, object_bgl, &binning_params_bgl],
            immediate_size: 0,
        });

        let cdf_resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cdf resolve pipeline layout"),
            bind_group_layouts: &[&hist_read_bgl],
            immediate_size: 0,
        });

        let cdf_build_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cdf build pipeline layout"),
            bind_group_layouts: &[&cdf_build_bgl],
            immediate_size: 0,
        });

        // Shader sources
        let common_wgsl = include_str!("../../shaders/common.wgsl");
        let cdf_sample_wgsl = include_str!("../../shaders/cdf_sample_common.wgsl");
        let accum_wgsl = include_str!("../../shaders/histo_accum.wgsl");
        let composite_wgsl = include_str!("../../shaders/histo_composite.wgsl");
        let binning_common_wgsl = include_str!("../../shaders/binning_common.wgsl");
        let binning_wgsl = include_str!("../../shaders/histo_binning.wgsl");
        let cdf_resolve_wgsl = include_str!("../../shaders/histo_cdf_resolve.wgsl");
        let cdf_build_wgsl = include_str!("../../shaders/histo_cdf_build.wgsl");

        let (accum_pipeline, composite_pipeline) = create_pipeline_pair(
            device,
            surface_format,
            &accum_layout,
            &composite_layout,
            &format!("{}\n{}\n{}", common_wgsl, cdf_sample_wgsl, accum_wgsl),
            composite_wgsl,
            "histo",
        );

        let binning_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("histo binning shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}\n{}", common_wgsl, binning_common_wgsl, binning_wgsl).into(),
            ),
        });

        // The binning pass rasterizes into the tile grid: no depth attachment, purely
        // additive targets, so ordering never matters.
        let binning_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("histo binning pipeline"),
            layout: Some(&binning_layout),
            vertex: wgpu::VertexState {
                module: &binning_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &binning_shader,
                entry_point: Some("fs_main"),
                targets: &binning_targets(),
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

        let cdf_resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cdf resolve shader"),
            source: wgpu::ShaderSource::Wgsl(cdf_resolve_wgsl.into()),
        });

        let cdf_targets: [Option<wgpu::ColorTargetState>; HISTO_TEXTURES] =
            std::array::from_fn(|_| {
                Some(wgpu::ColorTargetState {
                    format: HISTO_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            });

        let cdf_resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cdf resolve pipeline"),
            layout: Some(&cdf_resolve_layout),
            vertex: wgpu::VertexState {
                module: &cdf_resolve_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cdf_resolve_shader,
                entry_point: Some("fs_main"),
                targets: &cdf_targets,
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

        let cdf_build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cdf_build shader"),
            source: wgpu::ShaderSource::Wgsl(cdf_build_wgsl.into()),
        });

        let cdf_build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cdf_build pipeline"),
            layout: Some(&cdf_build_layout),
            module: &cdf_build_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            accum_pipeline,
            composite_pipeline,
            binning_pipeline,
            cdf_resolve_pipeline,
            cdf_build_pipeline,
            histo_accum_bgl,
            slice_accum_bgl,
            binning_params_bgl,
            hist_read_bgl,
            histo_composite_tex_bgl,
            cdf_build_bgl,
            flag_bgl,
        }
    }
}

fn create_pipeline_pair(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    accum_layout: &wgpu::PipelineLayout,
    composite_layout: &wgpu::PipelineLayout,
    accum_source: &str,
    composite_source: &str,
    label_prefix: &str,
) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
    let accum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label_prefix}_accum shader")),
        source: wgpu::ShaderSource::Wgsl(accum_source.into()),
    });

    let accum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label_prefix}_accum pipeline")),
        layout: Some(accum_layout),
        vertex: wgpu::VertexState {
            module: &accum_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &accum_shader,
            entry_point: Some("fs_main"),
            targets: &[
                additive_target(wgpu::TextureFormat::Rgba16Float),
                additive_target(wgpu::TextureFormat::R16Float),
            ],
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
        label: Some(&format!("{label_prefix}_composite shader")),
        source: wgpu::ShaderSource::Wgsl(composite_source.into()),
    });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label_prefix}_composite pipeline")),
        layout: Some(composite_layout),
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

    (accum_pipeline, composite_pipeline)
}
