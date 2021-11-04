configure = {
    "batch_size": 64,
    "hidden_size": 768,
    "num_workers": 0,
    "lr": 0.00005,
    "dropout": 0.1,
    "num_epochs": 60,
    "seed": 212,

    "train_data_path": "model_data/duconv.train.data",
    "dev_data_path": "model_data/duconv.dev.data",

    "label_vocab_path": "data/label.txt",
    "model_base_dir": "exp",

    "bert_version": "hfl/chinese-roberta-wwm-ext"
    # "hfl/chinese-roberta-wwm-ext", "bert-base-chinese" "hfl/chinese-bert-wwm", "hfl/chinese-bert-wwm-ext"
}
