#cene‑specific configuration for Boat_Pose – 100 views × 66 frames
# ------------------------------------------------------------

ModelHiddenParams = dict(
            kplanes_config = {
                        # 3 spatial planes + 1 time axis  (x, y, z, t)
                                'grid_dimensions':           2,        # leave at 2 for K‑Planes
                                        'input_coordinate_dim':      4,        # (x, y, z, t)
                                                'output_coordinate_dim':     16,       # channel dim of each plane
                                                        'resolution':                [64, 64, 64, 76]   # <── adjust last value if
                                                                                                               #     you render ≠ 66 frames
                                                                                                                   },

                # multi‑resolution Fourier features used by K‑Planes
                    multires                   = [1, 2],

                        # deformation network depth (0 ⇒ no explicit deformation MLP)
                            defor_depth                = 0,

                                # hidden width of tiny MLPs that sit behind the planes
                                    net_width                  = 128,

                                        # regularisers
                                            plane_tv_weight            = 2e-4,
                                                time_smoothness_weight     = 1e-3,
                                                    l1_time_planes             = 1e-4,

                                                        # runtime flags – keep them False for a fully dynamic model
                                                            no_do                      = False,   # disable offsets
                                                                no_dshs                    = False,   # disable SH features
                                                                    no_ds                      = False,   # disable scaling
                                                                        empty_voxel                = False,   # keep all voxels until pruning
                                                                            render_process             = False,
                                                                                static_mlp                 = False
                                                                                )

OptimizationParams = dict(
            dataloader                 = True,    # use the repo’s Dataset/Dataloader
                iterations                 = 15_000,  # total Adam steps
                    batch_size                 = 1,       # fits on a single 24 GB GPU
                        coarse_iterations          = 3_000,   # only K‑Planes before fine net kicks in
                            densify_until_iter         = 10_000,  # prune & densify early voxels

                                # opacity pruning thresholds
                                    opacity_threshold_coarse   = 0.005,
                                        opacity_threshold_fine_init= 0.005,
                                            opacity_threshold_fine_after=0.005
                                            )

