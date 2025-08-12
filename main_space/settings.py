import os

import torch
from torch.amp import GradScaler

class WandbConfig:

    class LoggingConfig:
        def __init__(self):
            self.strict_error = True

    class ProfilerConfig:
        def __init__(self):
            self.profiler_wait_steps = 100
            self.profiler_warmup_steps = 10
            self.profiler_active_steps = 2
            self.profiler_repeat_cycles = 1  # Total active steps = profiler_active_steps * profiler_repeat_cycles
            self.profiler_record_shapes = True
            self.profiler_profile_memory = True
            self.profiler_with_ops = False
            self.profiler_with_stack = False  # Set to False by default, as it can be costly

    def __init__(self):
        # --- Overall Logging Control ---
        self.is_wandb_enabled = False  # Master switch for all W&B interactions
        self.is_profiler_enabled = False  # Master switch for torch.profiler
        self.is_trace_handler_custom = False  # Choose dir where pytorch.profiler will save measured metrics

        # --- W&B Configuration ---
        self.wandb_project_name = "self-distill-research"
        self.wandb_entity = "luka_newbie"
        self.wandb_watch_level = "all"  # Options: "all", "gradients", "parameters", "none"
        self.wandb_log_freq_model_watch = 1000  # Frequency for wandb.watch
        self.wandb_log_freq_metrics = 10  # Frequency for wandb.log() for loss, lr, etc.
        self.does_wandb_log_graph = True  # Enable to get 'model' tab in wandb
        self.log_validation = False

        self.profilerConfig = self.ProfilerConfig()
        self.loggingConfig = self.LoggingConfig()


class DatasetConfig:
    def __init__(self):

        self.tokenizer_path  = os.path.join(os.getcwd(), "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")
        self.cache_dir = os.path.join(os.getcwd(), "/dataset_creation/temp_files/cache_hf_datasets")
        self.dataset_name = "wikitext"
        self.dataset_config = "wikitext-103-v1"
        self.num_workers_dataloader = 8
        print(f"NUMBER OF WORKERS FOR DATALOADER IS SET TO: {self.num_workers_dataloader}!!!!!!!!!")
        self.pin_memory_dataloader = True
        print(f"PIN MEMORY DATALOADER IS SET TO: {self.pin_memory_dataloader}!!!!!!!!!")
        self.vocab_size_from_tokenizer = True


class TrainingConfig:


    class HyperparameterConfig:

        def __init__(self):
            self.run_name = "Micro_tests"
            self.vocab_size = 5000  # Dummy vocab size
            self.d_model = 4  # Embedding dimension / model dimension
            self.num_heads = 1  # Number of attention heads
            self.num_layers = 2  # Number of Transformer blocks
            self.ctx_len = 16  # Max sequence length for dummy data and positional embeddings
            self.dropout_rate = 0.0
            self.attach_aux_heads = False

    class DistillConfig:

        def __init__(self):

            self.distill_mode = 'hidd_distill' # Can be "logits_outputs", "hidd_distill"


    def __init__(self):

        self.warmup_steps = int(200)
        self.train_steps = int(12000)
        self.peak_lr = 1e-2
        self.batch_size = 32
        self.scheduler_type = "cosine"  # Options: "cosine", "inverse_sqrt", "linear"
        self.min_lr = 1e-4
        self.training_precision = "float32"  # Options: "bfloat16", "float16" (uses GradScaler), "float32"
        self.gradient_clip_norm = 1.0
        self.seed = 42


        self.experimental_steps = int(150)
        self.experimental_stop = False


        self.precision_dtype = None
        self.scaler = None
        self.initialize_precision()
        self.doesClipGradients = True
        self.distill_enabled = False

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

        self.artifact_base_names = ["50_mil_classic",
                                    "sub_1M_distill_dummies",
                                    "distilled_model_dummies",
                                    "sub_1M_distill_dummies_smaller",
                                    "ctx_len_1M_exp",
                                    "100M_teacher",
                                    "1M_classic",
                                    "2M_classic",
                                    "5M_classic",
                                    "10M_classic",
                                    "30M_classic",
                                    "70M_classic",
                                    "150M_classic",
                                    "5M_classic_GPU",
                                    "500M_classic",
                                    "VECI_M_classic_GPU",
                                    "30M_distill_from_30M",
                                    "30M_distil_from_30M_soft_focused",
                                    "30M_distill_from_30M_hidd",
                                    "5M_distill_from_30_logits",
                                    "5M_distill_from_30_hidd",
                                    "5M_distill_from_10M_logits",
                                    "800M_classic",
                                    "1B_classic",
                                    "4M_distilled_5M",
                                    "1M_classic_2x_train",
                                    "1M_classic_2x_train_incr_lr",
                                    "1M_classic_2x_train_incr_lr_1e-3",
                                    "1M_distill_2M_outs_alpha_0.9",
                                    "1M_distill_2M_outs_alpha_0.1",
                                    "1M_distill_2M_outs_alpha_0.5",
                                    "1M_distill_2M_outs_norm_alpha_0.5",
                                    "1M_distill_2M_hidd_norm_alpha_0.5",
                                    "30M_self_distill_first_checkpoint",
                                    "10k_classic",
                                    "30M_testing_script",
                                    "30M_classic_w_heads",
                                    "30M_distill_30M_hidd_val",
                                    "12M_distill_30M_hidd_val",
                                    "12M_classic",
                                    "12M_distill_30M_hidd",
                                    "12M_distill_12M_hidd",
                                    "12M_distill_30M_hidd_2_5_7",
                                    "25M_classic",
                                    "25M_distill_30M_hidd_2_5_7",
                                    "25M_distill_25M_hidd_2_3_4",
                                    "25M_classic_aux_test"]

        self.artifact_base_name = "25M_classic_aux_test"  # TODO: fix for consistency
        self.alias_to_load = None
        if self.artifact_base_name not in self.artifact_base_names:
            raise ValueError("CheckpointConfig.__init__() error, chose invalid artifact base.")

        self.checkpoint_frequency = 3000

        self.attempt_load_checkpoint_if_exists= False
        self.strict_state_dict_loading = True



class Config:

    def __init__(self):
        self.wandbConfig = WandbConfig()
        self.datasetConfig = DatasetConfig()
        self.trainingConfig = TrainingConfig()
        self.checkpointConfig = CheckpointConfig()
