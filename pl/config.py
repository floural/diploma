class Config:
    name_or_path = "stabilityai/stablelm-zephyr-3b"

    num_cls_sent = 5
    num_cls_stan = 2

    batch_size = 32
    num_epochs = 30
    lr = 5e-5

    sentiment_train_data_path = "data/sentiment/new_train.jsonl"
    sentiment_val_data_path = "data/sentiment/validation.jsonl"

    stance_train_data_path = "data/stance/train.csv"
    stance_val_data_path = "data/stance/test.csv"
