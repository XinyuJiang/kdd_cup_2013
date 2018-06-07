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


# 得到每个作者写过的论文所在的会议id， 期刊id
def get_dict_auther_conference_journal(paper_path, conference_path, journal_path, paper_author_path, to_file):

    data_paper = util.read_dict_from_csv(paper_path)
    #dict_conference = util.read_dict_from_csv(conference_path)
    #dict_journal = util.read_dict_from_csv(journal_path)
    dict_paper_author = json.load(open(paper_author_path), encoding="utf-8")


    print("start...")
    bar = pyprind.ProgPercent(len(data_paper))
    dict_auther_conference_journal = {}
    for item in data_paper:
        paperId = int(item["Id"])
        journalId = item["JournalId"]
        confereceId = item["ConferenceId"]
        authorIds = list(map(int,(dict_paper_author[str(paperId)])))
        for authorId in authorIds:
            if authorId not in dict_auther_conference_journal:
                dict_auther_conference_journal[authorId]={"conferenceId": Counter(), "journalId":Counter()}

            dict_auther_conference_journal[authorId]["conferenceId"][confereceId] += 1
            dict_auther_conference_journal[authorId]["journalId"][journalId] += 1


        bar.update()

    print "dump..."
    json.dump(dict_auther_conference_journal, open(to_file, "w"), encoding="utf-8")







if __name__ == '__main__':
    k = 10
    get_dict_auther_conference_journal(
        config.PAPER_FILE,
        config.CONFERENCE_FILE,
        config.JOURNAL_FILE,
        config.DICTPAPERAUTHOR_FILE,
        os.path.join(config.DATA_PATH, "dict_auther_conference_journal.json"))

