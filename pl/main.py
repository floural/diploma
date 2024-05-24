from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from dataset import DataLoaderFactory
from module import TweetLM
from config import Config

tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")


train_loader = DataLoaderFactory.get_merged_dataloader(
        sentiment_data_path=Config.sentiment_train_data_path,
        stance_data_path=Config.stance_train_data_path,
        is_train=True,
    )
val_loader = DataLoaderFactory.get_merged_dataloader(
    sentiment_data_path=Config.sentiment_val_data_path,
    stance_data_path=Config.stance_val_data_path,
    is_train=False,
)

model = TweetLM(pad_token_id=DataLoaderFactory._tokenizer.eos_token_id)
trainer = Trainer(
    precision="16-mixed",
    max_epochs=Config.num_epochs,
    logger=tb_logger,
)
trainer.fit(model, [train_loader], [val_loader])
