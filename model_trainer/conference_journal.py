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
import pandas as pd
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



def get_cj_keywords(author_cj_path, conference_path, journal_path, author_paper_path, to_file):
    author_cj_keywords = {}
    author_cj = json.load(open(author_cj_path), encoding="utf-8")
    conferences = pd.read_csv(conference_path)
    journals = pd.read_csv(journal_path)
    author_paper = json.load(open(author_paper_path), encoding="utf-8")
    bar = pyprind.ProgPercent(len(author_cj))
    for authorId in author_cj:
        author_cj_keywords[authorId] = Counter()
        for conferenceId in author_cj[authorId]["conferenceId"]:
            wordsc = []
            if int(conferenceId) > 0:
                #print(conferenceId)
                values = conferences[conferences["Id"] == int(conferenceId)]["FullName"].values
                if len(values)>0:
                    wordsc = str(values[0]).split(" ")
            for word in wordsc:
                word = word.lower()
                author_cj_keywords[authorId][word]+=author_cj[authorId]["conferenceId"][conferenceId]
        for journalId in author_cj[authorId]["journalId"]:
            wordsj = []
            if int(journalId) > 0 :
                #print(journalId)
                values = journals[journals["Id"]==int(journalId)]["FullName"].values
                if len(values)>0:
                    wordsc = str(values[0]).split(" ")

            for word in wordsc:
                word = word.lower()
                author_cj_keywords[authorId][word]+=author_cj[authorId]["journalId"][journalId]
        bar.update()




    print "dump..."
    json.dump(author_cj_keywords, open(to_file, "w"), encoding="utf-8")



if __name__ == '__main__':

    # get_dict_auther_conference_journal(
    #     config.PAPER_FILE,
    #     config.CONFERENCE_FILE,
    #     config.JOURNAL_FILE,
    #     config.DICTPAPERAUTHOR_FILE,
    #     os.path.join(config.DATA_PATH, "dict_auther_conference_journal.json"))

    get_cj_keywords(
        config.AUTHOR_CONFERENCE_JOURNAL_FILE,
        config.CONFERENCE_FILE,
        config.JOURNAL_FILE,
        config.AUTHOR_PAPER,
        config.AUTHOR_CJ_KEYWORDS
    )