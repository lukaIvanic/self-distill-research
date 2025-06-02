class WandbConfig:
    class ProfilerConfig:
        def __init__(self):
            self.profiler_wait_steps = 200
            self.profiler_warmup_steps = 5
            self.profiler_active_steps = 5
            self.profiler_repeat_cycles = 5  # Total active steps = profiler_active_steps * profiler_repeat_cycles
            self.profiler_record_shapes = True
            self.profiler_profile_memory = True
            self.profiler_with_ops = True
            self.profiler_with_stack = False  # Set to False by default, as it can be costly

    def __init__(self):
        # --- Overall Logging Control ---
        self.is_wandb_enabled = True  # Master switch for all W&B interactions
        self.is_profiler_enabled = True  # Master switch for torch.profiler
        self.is_trace_handler_custom = True  # Choose dir where pytorch.profiler will save measured metrics

        # --- W&B Configuration ---
        self.wandb_project_name = "self-distill-research"
        self.wandb_entity = "luka_newbie"
        self.wandb_watch_level = "all"  # Options: "all", "gradients", "parameters", "none"
        self.wandb_log_freq_model_watch = 100  # Frequency for wandb.watch
        self.wandb_log_freq_metrics = 10  # Frequency for wandb.log() for loss, lr, etc.
        self.does_wandb_log_graph = False  # Enable to get 'model' tab in wandb

        self.profilerConfig = self.ProfilerConfig()


class TrainingConfig:
    """
           Testing for 3k steps:
           lr | batch_size
           1e-3 | 256
           1e-3 | 512
           1e-3 | 1024
           3e-4 | 256
           3e-4 | 512
           3e-4 | 1024
           1e-4 | 256
           1e-4 | 512
           1e-4 | 1024
    """

    class HyperparameterConfig:

        def __init__(self):
            self.vocab_size = 5000  # Dummy vocab size
            self.d_model = 1  # Embedding dimension / model dimension
            self.num_heads = 1  # Number of attention heads
            self.num_layers = 1  # Number of Transformer blocks
            self.ctx_len = 32  # Max sequence length for dummy data and positional embeddings
            self.dropout_rate = 0.1



    def __init__(self):
        self.warmup_steps = int(1e2)
        self.train_steps = int(1e3)
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.scheduler_type = "cosine"  # Options: "cosine", "inverse_sqrt", "linear"
        self.min_learning_rate = 1e-5
        self.seed = 42

        self.hyperParamConfig = self.HyperparameterConfig()

class Config:

    def __init__(self):
        self.wandbConfig = WandbConfig()
        self.trainingConfig = TrainingConfig()
