#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import json
import os

# 读取数据集
data1 = pd.read_csv('data/winemag-data_first150k.csv')
data2 = pd.read_csv('data/winemag-data-130k-v2.csv')
# print(data1.head())
# print("-----------------------")
# print(data2.head())
write_data_path1 = "data/"
# 判断两个文件中的标称属性和数值属性，分别存放成列表
# 150k的标称属性
nominal_array_150k = list(data1.select_dtypes(include=['object']))
# 150k的数值属性
numerical_array_150k = list(data1.select_dtypes(include=['int', 'float64']))
# print(nominal_array_150k)
# print(numerical_array_150k)
# 130k的标称属性
nominal_array_130k = list(data2.select_dtypes(include=['object']))
# 130k的数值属性
numerical_array_130k = list(data2.select_dtypes(include=['int', 'float64']))


data150k = pd.read_csv('data/winemag-data_first150k.csv', usecols=nominal_array_150k)
data150k.to_csv('result/150k.csv', index=False)

data130k = pd.read_csv('data/winemag-data-130k-v2.csv', usecols=nominal_array_130k)
data130k.to_csv('result/130k.csv', index=False)


# In[29]:


# 最小支持度
min_support = 0.25
# 最小置信度
min_confidence = 0.5
# 构建单元素候选项集合
def generate_data(dataset):
    list1 = []
    for data in dataset:
        for item in data:
            if [item] not in list1:
                list1.append([item])
    list1.sort()
    return [frozenset([item]) for item in list1]


# In[30]:


# 扫描项集集合，过滤掉小于最小支持度的项集
def scan_data(dataset, ck):
    # 根据待选项集ck的情况，判断数据集D中ck元素的出现频率
    ck_count = dict()
    for data in dataset:
        for cand in ck:
            if cand.issubset(data):
                if cand not in ck_count:
                    ck_count[cand] = 1
                else:
                    ck_count[cand] += 1
    num_items = float(len(dataset))
    return_list = []
    support_data = dict()
    # 过滤非频繁项集
    for key in ck_count:
        support = ck_count[key] / num_items
        if support >= min_support:
            return_list.insert(0, key)
        support_data[key] = support
    return return_list, support_data


# In[31]:


# 非重复的合并两个项集
# 当待选项集不是单个元素时，如k>=2的情况下， 合并元素时容易出现重复
# 因此，针对包括k个元素的频繁项集，对比每个频繁项集的第k-2位是否一致
def apriori_gen(lk, k):
    return_list = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i+1, len_lk):
            # 第k-2个项相同时，将两个集合合并
            l1 = list(lk[i])[:k-2]
            l2 = list(lk[j])[:k-2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                return_list.append(lk[i] | lk[j])
    return return_list


# In[32]:


# apriori主函数
def apriori_data(dataset):
    gen_data = generate_data(dataset)
    dataset = [set(data) for data in dataset]
    L1, support_data = scan_data(dataset, gen_data)
    l = [L1]
    k = 2
    while len(l[k-2]) > 0:
        ck = apriori_gen(l[k-2], k)
        lk, support_k = scan_data(dataset, ck)
        support_data.update(support_k)
        l.append(lk)
        k += 1
    return l, support_data


# In[33]:


def get_data_set(dataset):
    data = pd.read_csv(dataset)
    data


# In[35]:


data_set = get_data_set('result/150k.csv')
freq_set, support_data = apriori_data(data_set)
support_data_out = sorted(support_data.items(), key=lambda d: d[1], reverse=True)
write_data_path = 'result/150k/'
# 频繁项集输出到结果文件
freq_set_file = open(os.path.join(write_data_path, 'freq_set.json'), 'w')
for (key, value) in support_data_out:
    result_dict = {'set':None, 'sup':None}
    set_result = list(key)
    sup_result = value
    result_dict['set'] = set_result
    result_dict['sup'] = sup_result
    json_str = json.dumps(result_dict, ensure_ascii=False)
    freq_set_file.write(json_str+'\n')
freq_set_file.close()


# In[ ]:


data_set = get_data_set('result/130k.csv')
freq_set, support_data = apriori_data(data_set)
support_data_out = sorted(support_data.items(), key=lambda d: d[1], reverse=True)
write_data_path = 'result/130k/'
# 频繁项集输出到结果文件
freq_set_file = open(os.path.join(write_data_path, 'freq_set.json'), 'w')
for (key, value) in support_data_out:
    result_dict = {'set':None, 'sup':None}
    set_result = list(key)
    sup_result = value
    result_dict['set'] = set_result
    result_dict['sup'] = sup_result
    json_str = json.dumps(result_dict, ensure_ascii=False)
    freq_set_file.write(json_str+'\n')
freq_set_file.close()


# In[ ]:


# 用于评价生成的规则，并计算支持度，置信度，lift指标
def cal_conf(freq_set, h, support_data, big_rules_list):
    # 评估生成的规则
    prunedH = []
    for conseq in h:
        sup = support_data[freq_set]
        conf = sup / support_data[freq_set - conseq]
        lift = conf / support_data[freq_set - conseq]
        if conf >= min_confidence:
            big_rules_list.append((freq_set-conseq, conseq, sup, conf, lift))
            prunedH.append(conseq)
    return prunedH


# In[ ]:


# 递归的生成规则右部的结果项集
def rules_from_conseq(freq_set, h, support_data, big_rules_list):
    m = len(h[0])
    if len(freq_set) > (m+1):
        hmp1 = apriori_gen(h, m+1)
        hmp1 = cal_conf(freq_set, hmp1, support_data, big_rules_list)
        if len(hmp1) > 1:
            rules_from_conseq(freq_set, hmp1, support_data, big_rules_list)


# In[ ]:


# 产生强关联规则
"""
基于Apriori算法，首先从一个频繁项集开始，接着创建一个规则列表，其中规则右部只包含一个元素，然后对这些
规则进行测试。接下来合并所有的剩余规则列表来创建一个新的规则列表，其中规则右部包含两个元素。即分级法
"""
def generate_rules(l, support_data):
    # l是频繁项集
    # support_data:频繁项集对应的支持度
    # return：强关联规则列表
    big_rules_list = []
    for i in range(1, len(l)):
        for freq_set in l[i]:
            h1 = [frozenset([item]) for item in freq_set]
            # 只获取有两个或者更多元素的集合
            if i > 1:
                rules_from_conseq(freq_set, h1, support_data, big_rules_list)
            else:
                cal_conf(freq_set, h1, support_data, big_rules_list)
    return big_rules_list


big_rules_list = generate_rules(freq_set, support_data)
big_rules_list = sorted(big_rules_list, key = lambda x: x[3], reverse=True)
rules_file = open(os.path.join(write_data_path, 'rules.json'), 'w')
for result in big_rules_list:
    result_dict = {'X_set':None, 'Y_set': None, 'sup':None, 'conf':None, 'lift':None}
    X_set, Y_set, sup, conf, lift = result
    result_dict['X_set'] = list(X_set)
    result_dict['Y_set'] = list(Y_set)
    result_dict['sup'] = sup
    result_dict['conf'] = conf
    result_dict['lift'] = lift
    json_str = json.dumps(result_dict, ensure_ascii=False)
    rules_file.write(json_str + '\n')
rules_file.close()

