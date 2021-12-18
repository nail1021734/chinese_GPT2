import pickle

from utils.create_data import create_MN_data

if __name__ == '__main__':
    # for i in range(5):
    MN_dataset = create_MN_data(
        ckpt_path='checkpoint/MLM_exp12/checkpoint-3120000.pt',
        dataset_name='MLM_dataset_v4',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        k=40,
        data_num=80000,
        mask_strategy='Sentence_sent_rate',
        use_test_data=False,
        batch_size=32,
        # mask_rate={'max': 0.35, 'min': 0.25},
        mask_rate=0.30,
    )
    pickle.dump(MN_dataset, open('test.pkl', 'wb'))
