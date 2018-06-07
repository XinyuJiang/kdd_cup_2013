#!/usr/bin/env python
#encoding: utf-8
import os
import socket
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# 当前工作目录
#CWD = "/home/username/KDD/KDD_benchmark" # Linux系统目录
CWD = "E:\study\junior\Datamining\kdd_cup_contest\kdd_2013\KDD_Benchmark" # Windows系统目录

DATA_PATH = os.path.join(CWD, "data")
DATASET_PATH = os.path.join(DATA_PATH, "dataset")

# 训练和测试文件（训练阶段有验证数据，测试阶段使用测试数据）
TRAIN_FILE = os.path.join(DATASET_PATH, "train_set", "Train.csv")
TEST_FILE = os.path.join(DATASET_PATH, "valid_set", "Valid.csv")
GOLD_FILE = os.path.join(DATASET_PATH, "valid_set", "Valid.gold.csv")

# 模型文件
MODEL_PATH = os.path.join(CWD, "model", "kdd.model")
# 训练和测试特征文件
TRAIN_FEATURE_PATH = os.path.join(CWD, "feature", "train.feature")
TEST_FEATURE_PATH = os.path.join(CWD, "feature", "test.feature")
# 分类在测试集上的预测结果
TEST_RESULT_PATH = os.path.join(CWD, "predict", "test.result")
# 重新格式化的预测结果
TEST_PREDICT_PATH = os.path.join(CWD, "predict", "test.predict")


COAUTHOR_FILE = os.path.join(DATASET_PATH, "coauthor.json")
PAPERIDAUTHORID_TO_NAME_AND_AFFILIATION_FILE = os.path.join(DATASET_PATH, "paperIdAuthorId_to_name_and_affiliation.json")
PAPERAUTHOR_FILE = os.path.join(DATASET_PATH, "PaperAuthor.csv")
DICTPAPERAUTHOR_FILE = os.path.join(DATA_PATH,"dict_paperId_to_authors.json")
PAPER_FILE = os.path.join(DATASET_PATH, "Paper.csv")
AUTHOR_FILE = os.path.join(DATASET_PATH, "Author.csv")
AUTHOR_KEYWORDS = os.path.join(DATA_PATH, "dict_auther_keywords_freq.json")
JOURNAL_FILE = os.path.join(DATASET_PATH, "Journal.csv")
CONFERENCE_FILE = os.path.join(DATASET_PATH, "Conference.csv")
CONFERENCE_JOURNAL = os.path.join(DATA_PATH,'dict_auther_conference_journal.json')
AUTHER_PAPERID = os.path.join(DATA_PATH,'dict_authorId_to_paperId.json')
