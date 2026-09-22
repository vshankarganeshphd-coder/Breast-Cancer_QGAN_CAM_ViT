from dataclasses import dataclass

@dataclass
class Config:
    image_size: int = 256
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    dropout: float = 0.25
    qfa_scale: float = 1.75
    cam_reduction: int = 16
    transformer_depth: int = 6
    attention_heads: int = 8
    patch_size: int = 16
    embed_dim: int = 256
    segmentation_loss_weight: float = 0.6
    classification_loss_weight: float = 0.4
    sfo_population: int = 30
    sfo_iterations: int = 50
    num_classes: int = 3
    seed: int = 42
    num_workers: int = 4
    early_stopping_patience: int = 20
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

CFG = Config()
