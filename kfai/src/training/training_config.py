from pydantic import Field, PositiveFloat, PositiveInt
from kfai.src.config.base_config import BaseConfig


class TrainingConfig(BaseConfig):
    """Training configuration settings."""

    batch_size: PositiveInt = Field(
        default=1,
        description="Number of training examples used per batch.",
    )

    train_split: float = Field(
        default = 0.9,
        description ="Percent of data in training split."
    )

    epochs: PositiveInt = Field(
        default=10,
        description="Number of full passes over the training dataset.",
    )

    learning_rate: PositiveFloat = Field(
        default=0.0001,
        description="Learning rate for training.",
    )

    pos_weight: PositiveInt = Field(
        default=2,
        description="Weight for positive class in criterion (due to class imbalance)."
    )

    root_dir: str = Field(
        default='a360_data/Images',
        description="Directory containing image data."
    ) 
    
    mask_dir: str = Field(
        default='a360_data/Masks',
        description="Directory containing masks data."
    )