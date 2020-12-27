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
def get_authorId_to_paperId(paper_author_path, to_file):

    data = util.read_dict_from_csv(paper_author_path)

    dict_authorId_to_paperId= {}
    bar = pyprind.ProgPercent(len(data))
    for item in data:
        paperId = int(item["PaperId"])
        authorId = int(item["AuthorId"])
        if authorId not in dict_authorId_to_paperId:
            dict_authorId_to_paperId[authorId] = []
        dict_authorId_to_paperId[authorId].append(paperId)
        bar.update()



    print "dump..."
    json.dump(dict_authorId_to_paperId, open(to_file, "w"), encoding="utf-8")


if __name__ == '__main__':

    get_authorId_to_paperId(
        os.path.join(config.DATASET_PATH, "PaperAuthor.csv"),
        os.path.join(config.DATA_PATH, "dict_authorId_to_paperId.json"))

