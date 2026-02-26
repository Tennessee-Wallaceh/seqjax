"""Template plan for `seqjax.cli generate-slurm-jobs`.

Copy this file and adjust values.
"""

PLAN = {
    "experiment_name": "svar-ablation",
    "output_root": "experiments/jobs",
    "shared": {
        "model": "aicher_stochastic_vol",
        "inference": "buffer-vi",
        "sequence_length": 1000,
        "test_samples": 1000,
        "fit_seed_repeats": 3,
        "data_seed_repeats": 1,
        "base_fit_seed": 0,
        "base_data_seed": 0,
        "wall_time": "02:30:00",
        "gpus": 1,
        "source_venv": ".venv/bin/activate",
        "install_cmd": "uv pip install -e .[dev]",
        "fixed_codes": [
            "OPT.ADAM.LR-1e-3",
            "OPT.ADAM.MAXT-60m",
            "MC-50",
            "BS-20",
            "M-10",
            "B-14",
        ],
        "sub_grids": {
            "conv_defaults": [
                ["EMB.C1D.P-10", "EMB.C1D.PK-avg"],
                ["EMB.C1D.P-15", "EMB.C1D.PK-max"],
            ]
        },
    },
    "studies": [
        {
            "name": "conv1d_capacity",
            "axes": {
                "embedder": [
                    {
                        "codes": ["EMB.C1D", "EMB.C1D.H-2", "EMB.C1D.K-3", "EMB.C1D.D-4"],
                        "sub_grid": "conv_defaults",
                    },
                    {
                        "codes": ["EMB.C1D", "EMB.C1D.H-4", "EMB.C1D.K-5", "EMB.C1D.D-4"],
                        "sub_grid": "conv_defaults",
                    },
                ],
                "latent": [
                    ["LAX.MAF", "LAX.MAF.W-20", "LAX.MAF.D-2", "LAX.MAF.FL-1"],
                    ["LAX.MAF", "LAX.MAF.W-32", "LAX.MAF.D-2", "LAX.MAF.FL-2"],
                ],
            },
        },
        {
            "name": "pooling_ablation",
            "axes": {
                "pool": [
                    ["EMB.C1D", "EMB.C1D.PK-avg", "EMB.C1D.P-15"],
                    ["EMB.C1D", "EMB.C1D.PK-max", "EMB.C1D.P-15"],
                ]
            },
            "fixed_codes": ["LAX.MAF", "LAX.MAF.W-32", "LAX.MAF.D-2", "LAX.MAF.FL-2"],
        },
    ],
}
