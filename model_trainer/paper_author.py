#!/usr/bin/env python
#encoding: utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("../")
import util
import json
from collections import Counter
import config
import pyprind

# 根据PaperAuthor.csv，获取每篇论文的作者列表
def get_top_k_coauthors(paper_author_path, k, to_file):

    data = util.read_dict_from_csv(paper_author_path)

    dict_paperId_to_authors = {}
    bar = pyprind.ProgPercent(len(data))
    for item in data:
        paperId = int(item["PaperId"])
        authorId = int(item["AuthorId"])
        if paperId not in dict_paperId_to_authors:
            dict_paperId_to_authors[paperId] = []
        dict_paperId_to_authors[paperId].append(authorId)
        bar.update()



    print "dump..."
    json.dump(dict_paperId_to_authors, open(to_file, "w"), encoding="utf-8")


if __name__ == '__main__':
    k = 10
    get_top_k_coauthors(
        os.path.join(config.DATASET_PATH, "PaperAuthor.csv"),
        k,
        os.path.join(config.DATA_PATH, "dict_paperId_to_authors.json"))

