//! Render pipelines for the Gaussian splat front-end. One per transparency mode; they all
//! share the same vertex stage and differ only in their fragment outputs and blend state.
//!
//! Splats carry their own ordering (or deliberately ignore it), so depth testing is off in
//! every mode -- but the pipelines still declare a depth-stencil state, because they draw
//! into the same render passes as the mesh path.

pub struct SplatPipelines {
    pub alpha_pipeline: wgpu::RenderPipeline,
    pub front_pipeline: wgpu::RenderPipeline,
    pub wboit_pipeline: wgpu::RenderPipeline,
    pub histo_pipeline: wgpu::RenderPipeline,
    /// Mode 3's binning pass: the same EWA projection rasterized at one pixel per tile,
    /// blending optical depth into the 16-channel tile histogram.
    pub binning_pipeline: wgpu::RenderPipeline,
    pub sliced_pipeline: wgpu::RenderPipeline,
    pub splat_bgl: wgpu::BindGroupLayout,
}

fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: Default::default(),
        bias: Default::default(),
    }
}

fn shader(device: &wgpu::Device, label: &str, body: &str) -> wgpu::ShaderModule {
    let common = include_str!("../../shaders/splat_common.wgsl");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{}\n{}", common, body).into()),
    })
}

/// Mode 4 needs the slice helpers between the splat prelude and its own body.
fn sliced_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("splat sliced shader"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "{}\n{}\n{}\n{}",
                include_str!("../../shaders/splat_common.wgsl"),
                include_str!("../../shaders/front_common.wgsl"),
                include_str!("../../shaders/slice_common.wgsl"),
                include_str!("../../shaders/splat_sliced.wgsl"),
            )
            .into(),
        ),
    })
}

fn splat_front_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("splat front surface shader"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "{}\n{}\n{}",
                include_str!("../../shaders/splat_common.wgsl"),
                include_str!("../../shaders/front_common.wgsl"),
                include_str!("../../shaders/splat_front_surface.wgsl"),
            )
            .into(),
        ),
    })
}

impl SplatPipelines {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        histo_accum_bgl: &wgpu::BindGroupLayout,
        slice_accum_bgl: &wgpu::BindGroupLayout,
        binning_params_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let splat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat bgl"),
            entries: &[
                storage(0), // splats
                storage(1), // spherical harmonics
                storage(2), // depth-sorted draw order
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            cull_mode: None,
            ..Default::default()
        };

        // Mode 1: sorted back-to-front alpha blending.
        let alpha_shader = shader(
            device,
            "splat alpha shader",
            include_str!("../../shaders/splat_alpha.wgsl"),
        );
        let alpha_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat alpha layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl],
            immediate_size: 0,
        });
        let alpha_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat alpha pipeline"),
            layout: Some(&alpha_layout),
            vertex: wgpu::VertexState {
                module: &alpha_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &alpha_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Modes 2 and 3 share the MRT accumulation targets: additive accum, multiplicative
        // revealage.
        let mrt_targets = [
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
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
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R16Float,
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
                write_mask: wgpu::ColorWrites::RED,
            }),
        ];

        let wboit_shader = shader(
            device,
            "splat wboit shader",
            include_str!("../../shaders/splat_wboit.wgsl"),
        );
        let wboit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat wboit layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl],
            immediate_size: 0,
        });
        let wboit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat wboit pipeline"),
            layout: Some(&wboit_layout),
            vertex: wgpu::VertexState {
                module: &wboit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &wboit_shader,
                entry_point: Some("fs_main"),
                targets: &mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let histo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat histo shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    include_str!("../../shaders/splat_common.wgsl"),
                    include_str!("../../shaders/cdf_sample_common.wgsl"),
                    include_str!("../../shaders/splat_histo.wgsl"),
                )
                .into(),
            ),
        });
        let histo_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat histo layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl, histo_accum_bgl],
            immediate_size: 0,
        });
        let histo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat histo pipeline"),
            layout: Some(&histo_layout),
            vertex: wgpu::VertexState {
                module: &histo_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &histo_shader,
                entry_point: Some("fs_main"),
                targets: &mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Mode 3's binning pass: no depth attachment, four additive tile-histogram
        // targets, nothing bound beyond the params uniform.
        let binning_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat binning shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    include_str!("../../shaders/splat_common.wgsl"),
                    include_str!("../../shaders/binning_common.wgsl"),
                    include_str!("../../shaders/splat_binning.wgsl"),
                )
                .into(),
            ),
        });
        let binning_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat binning layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl, binning_params_bgl],
            immediate_size: 0,
        });
        let binning_targets = crate::pipeline::histogram_wboit::binning_targets();
        let binning_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat binning pipeline"),
            layout: Some(&binning_layout),
            vertex: wgpu::VertexState {
                module: &binning_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &binning_shader,
                entry_point: Some("fs_main"),
                targets: &binning_targets,
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Mode 4: the old histogram-buffer accumulation bind groups, four slabs instead
        // of an accum/optical-depth pair.
        let sliced_shader = sliced_shader(device);
        let sliced_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat sliced layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl, slice_accum_bgl],
            immediate_size: 0,
        });
        let slice_targets = crate::pipeline::sliced_oit::slice_targets();
        let sliced_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat sliced pipeline"),
            layout: Some(&sliced_layout),
            vertex: wgpu::VertexState {
                module: &sliced_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sliced_shader,
                entry_point: Some("fs_main"),
                targets: &slice_targets,
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Mode 4 prepass. The only splat pipeline with depth writes on: it is resolving
        // the nearest solid splat per pixel, which is what a depth test is for.
        let front_shader = splat_front_shader(device);
        let front_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat front surface layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl],
            immediate_size: 0,
        });
        let front_targets = crate::pipeline::sliced_oit::front_target();
        let front_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat front surface pipeline"),
            layout: Some(&front_layout),
            vertex: wgpu::VertexState {
                module: &front_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &front_shader,
                entry_point: Some("fs_main"),
                targets: &front_targets,
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(crate::pipeline::sliced_oit::front_depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            alpha_pipeline,
            front_pipeline,
            wboit_pipeline,
            histo_pipeline,
            binning_pipeline,
            sliced_pipeline,
            splat_bgl,
        }
    }
}
