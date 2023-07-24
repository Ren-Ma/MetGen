import re
import numpy as np
import jieba
import jieba.posseg as pseg
import pandas as pd
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BertTokenizer, Text2TextGenerationPipeline

model_dir = './tmp/CCL'
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir)
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

CFN_word2frame = pd.read_csv('./CFN_LEX_CN.csv')
def get_CFN(word):
    """输出word的frame"""
    if word in CFN_word2frame['word'].values:  # 如果word在CFN中存在
        word_frames = CFN_word2frame[CFN_word2frame['word'] == word]['FN']
        word_frame = word_frames.iloc[0]  # 现在只选第一个
    else:  # 如果这个词汇找不到对应的frame的话，就输出'UNK'
        word_frame = 'No_Frame'
    return word_frame
def get_json_format(sent, src_word, tgt_word=None):
    src_word_ft = '<V>' + src_word + ':' + get_CFN(src_word) + '<V>'
    if tgt_word:
        tgt_word_frame = get_CFN(tgt_word)
    else:
        tgt_word_frame = 'No_Frame'
    sent_src = tgt_word_frame + '<EOT>' + re.sub(src_word, src_word_ft, sent)
    return sent_src

# examples = {'他收购了他的竞争对手的公司。' : '收购',  # 吃掉
#             '导演新创作了系列的科幻电影。' : '创作',  # 出笼
#             '酒精麻痹你每个神经' : '麻痹',  # 浸透
#            '军训一天结束后大家成了死尸，带着疲惫的身子离开了。' : '带',  # 拖
#             '这辆汽车很耗费油' : '耗费',  # 吃
#             }

def get_verb_fr_sent(sentence):
    """从句子中抽取动词"""
    sent_words = pseg.lcut(sentence, use_paddle=True)
    verbs = [word for word, pos in sent_words if pos == 'v']
    return verbs

def meta_gen_row(row):
    """利用训练好的隐喻生成模型对df中每一行的句子进行隐喻生成"""
    sent_src = get_json_format(row[sent_col], row['ltr_words'])
    gen_sent = text2text_generator(sent_src, max_length=50)[0]['generated_text']
    meta_sent = re.sub(' ', '', gen_sent)
    return meta_sent

# for sent, src_word in examples.items():
#     sent_src = get_json_format(sent, src_word)
#     print(re.sub(' ', '', text2text_generator(sent_src, max_length=50)[0]['generated_text']))
filename = '粮食'
df = pd.read_excel('./text_fr_caihua/' + filename + '.xlsx')
sent_col = 'sentence'
df['ltr_words'] = df[sent_col].apply(get_verb_fr_sent)
df_explode = df.explode('ltr_words', ignore_index=True)
df_explode = df_explode[df_explode['ltr_words'].notna()]
df_explode_copy = df_explode.copy()
for idx, row in tqdm(df_explode.iterrows()):
    df_explode_copy.at[idx, 'meta_sent'] = meta_gen_row(row)
df_explode_copy.to_excel('./text_fr_caihua/' + filename + '_Meta_Gen.xlsx', index=False)
df_explode_meta_gen = df_explode.apply(lambda x: meta_gen_row(x), axis=1)
# 把预测结果中的空格键去掉-----------------------------------------------------------
with open('./tmp/CCL_no_frame/generated_predictions.txt') as f:
    file = f.readlines()
    predictions = open('./tmp/CCL_no_frame/generated_predictions_nospace.txt', 'w', encoding='utf-8')
    for line in file:
        predictions.write(str(re.sub(' ', '', line)))
        # train_data.write(line)
    predictions.close()

# 把预测结果添加到test.xlsx中
test = pd.read_excel('./data/test.xlsx')
predictions = open('./tmp/CCL_no_frame/generated_predictions_nospace.txt').readlines()
test['predicted_no_frame'] = [line.strip() for line in predictions]
test['sentence_eq_predicted'] = (test['sentence'] == test['predicted'])
test['sentence_eq_predicted_no_frame'] = (test['sentence'] == test['predicted_no_frame'])
test['predicted_eq_predicted_no_frame'] = (test['predicted'] == test['predicted_no_frame'])
sum(test['sentence_eq_predicted'])  # 438
sum(test['sentence_eq_predicted_no_frame'])  # 442
sum(test['predicted_eq_predicted_no_frame'])  # 617
# test['sent_no_meta'] = test.apply(lambda row: re.sub(row['meta_word'], row['meta_sub'], row['sentence']), axis=1)
test.to_excel('./data/test.xlsx', index=False)

