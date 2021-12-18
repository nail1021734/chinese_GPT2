from utils.config import build_configs
from utils.train import train
from transformers import GPT2Config

if __name__ == '__main__':
    # Create training config.
    config, model_config = build_configs(
        # Training config.
        max_length=512,
        tokenizer_name='chinese_tokenizer_big',
        save_ckpt_step=80000,
        log_step=500,
        exp_name='LM_exp1',
        dataset_name='Taiwan_news_dataset',
        # from_pretrained_model='checkpoint/MLM_exp12/checkpoint-3120000.pt',

        # Hyperparameters.
        seed=22,
        lr=2e-5,
        epoch_num=30,
        batch_size=16,
        accumulate_step=40,
        warm_up_step_rate=0.02,
        weight_decay=0.01,

        # Model config. (Use to create `GPT2Config` object.)
        model_config=GPT2Config.from_pretrained(
            'ckiplab/gpt2-base-chinese').__dict__)

    train(config=config, model_config=model_config)
