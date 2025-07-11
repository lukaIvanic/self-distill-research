import os

import torch
from torch.amp import GradScaler

class WandbConfig:

    class LoggingConfig:
        def __init__(self):
            self.strict_error = True

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
        self.is_profiler_enabled = False  # Master switch for torch.profiler
        self.is_trace_handler_custom = False  # Choose dir where pytorch.profiler will save measured metrics

        # --- W&B Configuration ---
        self.wandb_project_name = "self-distill-research"
        self.wandb_entity = "luka_newbie"
        self.wandb_watch_level = "all"  # Options: "all", "gradients", "parameters", "none"
        self.wandb_log_freq_model_watch = 1000  # Frequency for wandb.watch
        self.wandb_log_freq_metrics = 200  # Frequency for wandb.log() for loss, lr, etc.
        self.does_wandb_log_graph = True  # Enable to get 'model' tab in wandb

        self.profilerConfig = self.ProfilerConfig()
        self.loggingConfig = self.LoggingConfig()


class DatasetConfig:
    def __init__(self):

        self.tokenizer_path  = os.path.join(os.getcwd(), "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")
        self.cache_dir = os.path.join(os.getcwd(), "/dataset_creation/temp_files/cache_hf_datasets")
        self.dataset_name = "wikitext"
        self.dataset_config = "wikitext-103-v1"
        self.num_workers_dataloader = 0
        self.pin_memory_dataloader = True  # TODO: investigate this
        self.vocab_size_from_tokenizer = True


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
            self.d_model = 32  # Embedding dimension / model dimension
            self.num_heads = 1  # Number of attention heads
            self.num_layers = 6  # Number of Transformer blocks
            self.ctx_len = 64  # Max sequence length for dummy data and positional embeddings
            self.dropout_rate = 0.0

    class DistillConfig:

        def __init__(self):

            self.distill_mode = 'logits_outputs'


    def __init__(self):
        self.warmup_steps = int(1e3)
        self.train_steps = int(1e4)
        self.peak_lr = 1e-3
        self.batch_size = 64
        self.scheduler_type = "cosine"  # Options: "cosine", "inverse_sqrt", "linear"
        self.min_lr = 1e-4
        self.training_precision = "bfloat16"  # Options: "bfloat16", "float16" (uses GradScaler), "float32"
        # TODO: add elsewhere check for gradient norm setting
        self.gradient_clip_norm = 1.0
        self.seed = 42


        self.experimental_steps = int(1e3)
        self.experimental_stop = False


        self.precision_dtype = None
        self.scaler = None
        self.initialize_precision()
        self.doesClipGradients = True
        self.distill_enabled = True

        self.hyperParamConfig = self.HyperparameterConfig()
        self.distillConfig = self.DistillConfig()


    def initialize_precision(self):
        if self.training_precision == "float16":
            self.precision_dtype = torch.float16
            self.scaler = GradScaler(device='cuda', enabled=True)
            print(f"Using Automatic Mixed Precision with dtype: {self.precision_dtype} and GradScaler.")
        elif self.training_precision == "bfloat16":
            self.precision_dtype = torch.bfloat16
            print(f"Using Automatic Mixed Precision with dtype: {self.precision_dtype}.")
        elif self.training_precision == "float32":
            self.precision_dtype = torch.float32
            print("Using float32 precision.")


class CheckpointConfig:
    def __init__(self):
        # TODO: set option to save or not to save, be careful it doesn't
        #       contradict with other wandb logging options,
        #       in the validate_config function

        self.artifact_base_names = ["sub_1M_distill_dummies",
                                    "distilled_model_dummies",
                                    "sub_1M_distill_dummies_smaller",
                                    "ctx_len_1M_exp",
                                    "100M_teacher"]
        self.artifact_base_name = "distilled_model_dummies"  # TODO: fix for consistency
        self.alias_to_load = None
        if self.artifact_base_name not in self.artifact_base_names:
            raise ValueError("CheckpointConfig.__init__() error, chose invalid artifact base.")

        self.checkpoint_frequency = 2000

        self.attempt_load_checkpoint_if_exists= False
        self.strict_state_dict_loading = True



class Config:

    def __init__(self):
        self.wandbConfig = WandbConfig()
        self.datasetConfig = DatasetConfig()
        self.trainingConfig = TrainingConfig()
        self.checkpointConfig = CheckpointConfig()
