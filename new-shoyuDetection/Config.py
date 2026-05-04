#coding:utf-8

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'models/best.pt'

# 对应data.yaml中的类别：key是类别索引（从0开始），value是英文名称
names = {
    0: 'time',
    1: 'you/your/this',
    2: 'morning',
    3: '9',
    4: '0',
    5: 'happy',
    6: 'new',
    7: 'wish',
    8: 'please',
    9: 'road',
    10: 'birthday',
    11: 'flat',
    12: 'safe',
    13: 'friend',
    14: '8',
    15: 'know',
    16: 'business card',
    17: 'marry/wife',
    18: 'tea',
    19: 'have',
    20: 'flavor',
    21: 'today',
    22: 'door',
    23: 'stop',
    24: 'thank you',
    25: 'slow',
    26: 'walk',
    27: 'late/night',
    28: 'I/me',
    29: 'love',
    30: 'good',
    31: 'peason',
    32: 'what',
    33: 'name',
    34: 'introduce'
}

# 对应names的中文名称（按索引顺序一一对应）
CH_names = [
    '时间', '你/你的/这个', '早上', '9', '0', '开心', '新的',
    '祝愿', '请', '路', '生日', '平的', '安全', '朋友', '8',
    '知道', '名片', '结婚/妻子', '茶', '有', '味道', '今天', '门',
    '停', '谢谢', '慢', '走', '晚/晚上', '我', '爱', '好的',
    '人', '什么', '名字', '介绍'
]