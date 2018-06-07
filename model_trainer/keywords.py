#!/usr/bin/env python
#encoding: utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("../")
import util
import json
import pyprind
from collections import Counter
import config
import re


# 根据Paper.csv，PaperAuther.csv获取每个作者的top k个keywords
def get_dict_auther_keywords(paper_path, paper_author_path, k, to_file):

    data_paper = util.read_dict_from_csv(paper_path)
    dict_paper_author= json.load(open(paper_author_path), encoding="utf-8")
    #print(dict_paper_author["1048576"])
    dict_auther_keywords = {}
    print("start...")
    bar = pyprind.ProgPercent(len(data_paper))
    for item in data_paper:
        paperId = int(item["Id"])
        title = item["Title"]
        keywords = item["Keyword"]
        key = util.get_string_splited(title + " " + keywords)

        for authorId in dict_paper_author[str(paperId)]:
            if authorId not in dict_auther_keywords:
                dict_auther_keywords[authorId]=[]
            dict_auther_keywords[authorId].extend(key)
        bar.update()

    print "dump..."
    json.dump(dict_auther_keywords, open(to_file, "w"), encoding="utf-8")




#计算关键字出现频率
def get_dict_author_keyword_freq(dict_author_keywords_path,to_file):
    dict_author_keywords_freq = {}
    dict_author_keywords = json.load(open(dict_author_keywords_path),encoding="utf-8")
    bar = pyprind.ProgPercent(len(dict_author_keywords))
    for autherId in dict_author_keywords:
        if autherId not in dict_author_keywords_freq:
            dict_author_keywords_freq[autherId] = Counter()
        keywords = dict_author_keywords[autherId]
        for keyword in keywords:
            dict_author_keywords_freq[autherId][keyword] += 1
    bar.update()


    print "dump..."
    json.dump(dict_author_keywords_freq, open(to_file, "w"), encoding="utf-8")


if __name__ == '__main__':
    k = 10
    get_dict_auther_keywords(
        os.path.join(config.DATASET_PATH, "Paper.csv"),
        os.path.join(config.DATA_PATH, "dict_paperId_to_authors.json"),
        k,
        os.path.join(config.DATA_PATH, "auther_keywords.json"))


    get_dict_author_keyword_freq(
        os.path.join(config.DATA_PATH, "auther_keywords.json"),
        os.path.join(config.DATA_PATH, "dict_auther_keywords_freq.json"))
