#!/usr/bin/env python
#encoding: utf-8
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import util
import numpy as np
import re


#2. coauthor信息
# 很多论文都有多个作者，根据paperauthor统计每一个作者的top 10（当然可以是top 20或者其他top K）的coauthor，
# 对于一个作者论文对（aid，pid），计算ID为pid的论文的作者有没有出现ID为aid的作者的top 10 coauthor中，
# (1). 可以简单计算top 10 coauthor出现的个数，
# (2). 还可以算一个得分，每个出现pid论文的top 10 coauthor可以根据他们跟aid作者的合作次数算一个分数，然后累加，
# 我简单地把coauthor和当前aid作者和合作次数作为这个coauthor出现的得分。


# 1. 简单计算top 10 coauthor出现的个数
def coauthor_1(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, dict_author_conference_journal, dict_author_cj_keywords, Conference, Journal, PaperAuthor, Author, Paper):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    # 从PaperAuthor中，根据paperId找coauthor。
    curr_coauthors = list(map(str, list(PaperAuthor[PaperAuthor["PaperId"] == int(paperId)]["AuthorId"].values)))
    #
    top_coauthors = dict_coauthor[authorId].keys()

    # 简单计算top 10 coauthor出现的个数
    nums = len(set(curr_coauthors) & set(top_coauthors))

    return util.get_feature_by_list([nums])


# 2. 还可以算一个得分，每个出现pid论文的top 10 coauthor可以根据他们跟aid作者的合作次数算一个分数，然后累加，
def coauthor_2(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, dict_author_conference_journal, dict_author_cj_keywords, Conference, Journal, PaperAuthor, Author, Paper):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    # 从PaperAuthor中，根据paperId找coauthor。
    curr_coauthors = list(map(str, list(PaperAuthor[PaperAuthor["PaperId"] == int(paperId)]["AuthorId"].values)))

    # {"authorId": 100}
    top_coauthors = dict_coauthor[authorId]

    score = 0
    for curr_coauthor in curr_coauthors:
        if curr_coauthor in top_coauthors:
            score += top_coauthors[curr_coauthor]

    return util.get_feature_by_list([score])

''' String Distance Feature'''
# 1. name-a 与name1##name2##name3的距离，同理affliction-a 和 aff1##aff2##aff3的距离
def stringDistance_1(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, dict_author_conference_journal, dict_author_cj_keywords, Conference, Journal, PaperAuthor, Author, Paper):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    key = "%s|%s" % (paperId, authorId)
    name = str(dict_paperIdAuthorId_to_name_aff[key]["name"])
    aff = str(dict_paperIdAuthorId_to_name_aff[key]["affiliation"])

    T = list(Author[Author["Id"] == int(authorId)].values)[0]
    a_name = str(T[1])
    a_aff = str(T[2])
    if a_name == "nan":
        a_name = ""
    if a_aff == "nan":
        a_aff = ""

    feat_list = []

    # 计算 a_name 与 name 的距离
    feat_list.append(len(longest_common_subsequence(a_name, name)))
    feat_list.append(len(longest_common_substring(a_name, name)))
    feat_list.append(Levenshtein_distance(a_name, name))
    # 计算 a_aff 与 aff 的距离
    feat_list.append(len(longest_common_subsequence(a_aff, aff)))
    feat_list.append(len(longest_common_substring(a_aff, aff)))
    feat_list.append(Levenshtein_distance(a_aff, aff))

    return util.get_feature_by_list(feat_list)


# 2. name-a分别与name1，name2，name3的距离，然后取平均，同理affliction-a和,aff1，aff2，aff3的平均距离
def stringDistance_2(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, dict_author_conference_journal, dict_author_cj_keywords, Conference, Journal, PaperAuthor, Author, Paper):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    key = "%s|%s" % (paperId, authorId)
    name = str(dict_paperIdAuthorId_to_name_aff[key]["name"])
    aff = str(dict_paperIdAuthorId_to_name_aff[key]["affiliation"])

    T = list(Author[Author["Id"] == int(authorId)].values)[0]
    a_name = str(T[1])
    a_aff = str(T[2])
    if a_name == "nan":
        a_name = ""
    if a_aff == "nan":
        a_aff = ""

    feat_list = []

    # 计算 a_name 与 name 的距离
    lcs_distance = []
    lss_distance = []
    lev_distance = []
    for _name in name.split("##"):
        lcs_distance.append(len(longest_common_subsequence(a_name, _name)))
        lss_distance.append(len(longest_common_substring(a_name, _name)))
        lev_distance.append(Levenshtein_distance(a_name, _name))

    feat_list += [np.mean(lcs_distance), np.mean(lss_distance), np.mean(lev_distance)]

    # 计算 a_aff 与 aff 的距离
    lcs_distance = []
    lss_distance = []
    lev_distance = []
    for _aff in aff.split("##"):
        lcs_distance.append(len(longest_common_subsequence(a_aff, _aff)))
        lss_distance.append(len(longest_common_substring(a_aff, _aff)))
        lev_distance.append(Levenshtein_distance(a_aff, _aff))

    feat_list += [np.mean(lcs_distance), np.mean(lss_distance), np.mean(lev_distance)]

    # # feat_list
    # feat_list = [feat_list[0],feat_list[1], feat_list[3],feat_list[4]]

    return util.get_feature_by_list(feat_list)


#包含两个特征，分别是关键字和标题单词重合数，以及根据频率得到的分数
def keywords_1(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, dict_author_conference_journal, dict_author_cj_keywords, Conference, Journal, PaperAuthor, Author, Paper):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    #该作者以前写过的论文的keywords集合
    fomer_keywords = dict_author_keywords[authorId].keys()
    fomer_dict_keywords = dict_author_keywords[authorId]

    #当前论文的keywords集合current_key
    title = Paper[Paper["Id"]==int(paperId)]["Title"].values
    keywords = Paper[Paper["Id"]==int(paperId)]["Keyword"].values
    if len(title) :
        title=str(title[0])
    else:
        title = ' '
    keywords = str(keywords[0])
    if keywords=="nan":
        keywords = ' '
    curr_keywords = util.get_string_splited(title + " " + keywords)

    #统计关键字相同的个数
    nums = len(set(curr_keywords) & set(fomer_keywords))

    #统计分数
    score = 0
    for word in curr_keywords:
        if word in fomer_dict_keywords:
            score += fomer_dict_keywords[word]
    #print nums, score
    return util.get_feature_by_list([nums, score])




# def conference_journal_3(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, PaperAuthor, Author, Paper,dict_author_conference_journal,conferences,journals,dict_author_paperid):
#     authorId = AuthorIdPaperId.authorId#目前作者的id
#     paperId = AuthorIdPaperId.paperId#目前paper的id
#
#
#     #这篇论文所在conference和journal情况
#     conferenceId = Paper[Paper["Id"]==int(paperId)]["ConferenceId"].values
#     journalId = Paper[Paper["Id"]==int(paperId)]["JournalId"].values
#     conferenceId = str(conferenceId[0])
#     journalId = str(journalId[0])
#
#     feat_list = []
#
#     thesameurl = 0
#     temp = "a"
#     temp2 = "b"
#
#     #如果目前这篇confereceid和journalid均为0，那么返回[0]
#     if conferenceId == "0" and journalId == "0":
#         #feat_list = [ max(dict_author_conference_journal[authorId]["conferenceId"][conferenceId],dict_author_conference_journal[authorId]["journalId"][journalId]) ]
#         feat_list = [0]
#
#     #如果目前这篇conferenceid或journalid不为0，且该id作者之前没发过，那么查找该作者之前发的paper所在conference情况;其中如果作者发过该期刊，则直接拿该期刊的发布次数作为score；如果没有发过该期刊，那么求最近期刊和当前期刊的距离运算后作为权重乘以分数作为分数
#     if conferenceId != "0":
#         temp = conferences[conferences["Id"]==int(conferenceId)]["HomePage"].values
#         #print ("aaaa",type(temp))
#         if temp != None:
#             str_conference = str(temp[0])
#         else:
#             str_conference = "0"
#
#         for conference in dict_author_conference_journal[authorId]["conferenceId"]:
#         #比较目前的conference和该作者dict里的conference，找到属于同一个主url的conference
#             temp2 = conferences[conferences["Id"]==int(conference)]["HomePage"].values
#             if temp2 != None:
#                 str_tempconference = str(temp2[0])
#             else:
#                 str_tempconference = "1"
#             if in_thesame_major_website(str_tempconference,str_conference):
#                 #如果属于同一个主域，那么将作者发过的这个会议或者期刊的次数记录下来累加
#                 thesameurl += dict_author_conference_journal[authorId]["conferenceId"][conference]
#         feat_list = [thesameurl]
#
#         thesameurl = 0
#
#     if  journalId != "0":
#         temp = journals[journals["Id"]==int(journalId)]["HomePage"].values
#         if temp != None:
#             str_journal = str(temp[0])
#         else:
#             str_journal = "0"
#
#         for journal in dict_author_conference_journal[authorId]["journalId"]:
#             temp2 = journals[journals["Id"]==int(journal)]["HomePage"].values
#             if temp2 != None:
#                 str_tempjournal = str(temp2[0])
#             else:
#                 str_tempjournal = "1"
#
#             if in_thesame_major_website(str_tempjournal,str_journal) :
#                 thesameurl += dict_author_conference_journal[authorId]["journalId"][journal]
#         feat_list = [thesameurl]
#
#
#
#     return util.get_feature_by_list(feat_list)

def conference_journal_1(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, dict_author_conference_journal, dict_author_cj_keywords, Conference, Journal, PaperAuthor, Author, Paper ):

    authorId = AuthorIdPaperId.authorId#目前作者的id

    paperId = AuthorIdPaperId.paperId#目前paper的id


    # 该作者以前写过的论文的会议期刊名称keywords集合
    fomer_cj_keywords = dict_author_cj_keywords[authorId].keys()
    fomer_dict_cj_keywords = dict_author_cj_keywords[authorId]

    # 当前论文所在会议期刊名称的keywords集合current_key
    cid = Paper[Paper["Id"] == int(paperId)]["ConferenceId"].values
    jid = Paper[Paper["Id"] == int(paperId)]["JournalId"].values
    words = []
    cid = list(map(int,cid))
    jid = list(map(int,jid))
    #print(cid, "/t", jid)
    if type(cid) == list:
        cid = cid[0]
    if type(jid) == list:
        jid = jid[0]
    # if len(cid):
    #     cid = int(cid[0])
    # else:
    #     cid = 0
    # if len(cid):
    #     cid = int(cid[0])
    # else:
    #     cid = 0
    if cid==0 and jid==0:
        return  util.get_feature_by_list([0,0])
    if cid > 0:
        # print(conferenceId)
        values = Conference[Conference["Id"] == cid]["FullName"].values
        #print("values: ", values)
        if len(values) > 0:
            wordsc = str(values[0]).split(" ")
            words.extend([word.lower() for word in wordsc])
    if jid > 0:
        # print(conferenceId)
        values = Journal[Journal["Id"] == jid]["FullName"].values
        if len(values) > 0:
            wordsj = str(values[0]).split(" ")
            words.extend([word.lower() for word in wordsj])
    #print(words)
    # 统计关键字相同的个数
    nums = len(set(words) & set(fomer_cj_keywords))

    # 统计分数
    score = 0
    for word in words:
        if word in fomer_dict_cj_keywords:
            score += fomer_dict_cj_keywords[word]
    #print nums, score
    return util.get_feature_by_list([nums, score])


#根据会议期刊名字的关键字评分
def conference_journal_2(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, dict_author_conference_journal, dict_author_cj_keywords, Conference, Journal, PaperAuthor, Author, Paper ):

    authorId = AuthorIdPaperId.authorId#目前作者的id

    paperId = AuthorIdPaperId.paperId#目前paper的id


    #dict_author_conference_journal表示作者之前paper所在会议情况

    conferenceId = Paper[Paper["Id"]==int(paperId)]["ConferenceId"].values

    journalId = Paper[Paper["Id"]==int(paperId)]["JournalId"].values


    conferenceId = str(conferenceId[0])

    journalId = str(journalId[0])

    score = [dict_author_conference_journal[authorId]["conferenceId"][conferenceId] + dict_author_conference_journal[authorId]["journalId"][journalId]]



    return util.get_feature_by_list(score)


def yeardistance(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, dict_author_keywords, PaperAuthor, Author, Paper,dict_author_conference_journal,conference,journal ,dict_author_paperid):
    authorId = AuthorIdPaperId.authorId#目前作者的id
    paperId = AuthorIdPaperId.paperId#目前paper的id

    #当前Paper年份附近年份该作者有没有发过Paper 如果作者已经很久没有发过paper那么有理由相信这篇paper是这个作者发的可能性比较小
    minyear = 0
    feat_list = []
    #当前paper发的时间
    curyear = (Paper[Paper["Id"]==int(paperId)]["Year"].values)[0]
    #根据作者的id从paperauthor数据集中找到他发过的所有paper
    """for item in dict_author_paperid[authorId]:
                    #print ("item:",item)
                    #计算每一个paper所发的时间
                    tempyear = int(Paper[Paper["Id"]==int(item)]["Year"])
                    yearlist.append(abs(int(curyear) - int(tempyear)))
                yearlist = [np.min(yearlist)]"""

    minyear = np.min(Paper[Paper["Id"].isin (dict_author_paperid[authorId])]["Year"])
    if minyear < 8 :
        feat_list = [1]
    else :
        feat_list = [-1]
    return util.get_feature_by_list(feat_list)


''' 一些距离计算方法 '''

# 最长公共子序列（LCS）, 获取是a, b的最长公共子序列
def longest_common_subsequence(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            x -= 1
            y -= 1
    return result


# 最长公共子串（LSS）
def longest_common_substring(a, b):
    m = [[0] * (1 + len(b)) for i in range(1 + len(a))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(a)):
        for y in range(1, 1 + len(b)):
            if a[x - 1] == b[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return a[x_longest - longest: x_longest]


# 编辑距离
def Levenshtein_distance(input_x, input_y):
    xlen = len(input_x) + 1  # 此处需要多开辟一个元素存储最后一轮的计算结果
    ylen = len(input_y) + 1

    dp = np.zeros(shape=(xlen, ylen), dtype=int)
    for i in range(0, xlen):
        dp[i][0] = i
    for j in range(0, ylen):
        dp[0][j] = j

    for i in range(1, xlen):
        for j in range(1, ylen):
            if input_x[i - 1] == input_y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[xlen - 1][ylen - 1]


if __name__ == '__main__':
    # print Levenshtein_distance("abc","ab")
    print (Levenshtein_distance("abc", "ab"))


