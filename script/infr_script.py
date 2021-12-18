from utils.inference import format_article, inference

if __name__ == '__main__':
    sen = inference(
        ckpt_path='checkpoint/MLM_exp12/checkpoint-3120000.pt',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        prompts=['[ARTICLE]新北市永和區8日上午,[MASK_S],警方鎖定54歲符姓男子涉嫌重大,晚間持拘票上門將其逮捕。符男向警方供稱,因不滿84歲[MASK_N]長期施暴77歲母親,才會懷恨殺害父親後,將遺體拖到頂樓棄置,數天後忍受不了臭味才將父親肢解棄屍。附近住戶表示,[MASK_S],而符男在建設公寓時還在頂樓預留空地增設違建,[MASK_S],認為符男性格比較自私。符男落網後向警方供稱,[MASK_S],11月28日趁父親熟睡時,使用[MASK_N]父親,並把遺體拖到頂樓用[MASK_N]裝著,直到12月6日屍體腐爛傳出味道,才使用美工刀跟鋸子肢解遺體,7日凌晨用登機箱載運到保平路大排水溝傾倒棄屍。附近住戶表示,符男所住的社區是他們家自地自建的,死者的女兒為建商,但房子正在建築時,符男擔任建地監工,卻違規在頂樓處預留空間增蓋違建,房子蓋好後符男還偷接社區公電,覺得符男的個性比較自私。原本還擔任社區管理委員他,因為和其他委員不合就辭職離開管委會。[SEP]'],
        k=40,
    )
    print(format_article(sen[0]))
