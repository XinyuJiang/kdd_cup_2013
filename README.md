# KDD 数据挖掘大作业

## Benchmark 程序

### 一、程序使用python27进行开发，需要安装以下包：
* [numpy](http://www.numpy.org/)
* [sklearn](http://scikit-learn.org/stable/)
* [pandas](http://pandas.pydata.org/)
* [pyprind](https://pypi.python.org/pypi/PyPrind): 用于显示进度条。位于make_feature_file.py中，用于显示抽取特征的进度，不需要的同学可以注释掉相应的行)

 建议通过安装 [Anaconda2](https://www.continuum.io/downloads "anaconda2") 来获得python27 以及上面相关的包。
	
### 二、运行方式：

1. 下载data数据，解压后放到项目根目录下:

	```
	链接: https://pan.baidu.com/s/1qY6K7EW 密码: wh12
	```

2. 修改配置文件config.py中的CWD（Current Working Directory）变量的值，将其改成当前项目所在的目录，如：
	
	```
	#当前工作目录 
	CWD = "/home/jianxiang/pycharmSpace/KDD_benchmark"
	``` 
3. 进入model_trainer文件夹，使用以下命令运行程序：

	```
	python trainer.py
	```
	
	程序将对 config.py 中 TRAIN\_FILE 对应的训练文件，抽取特征，构建分类器。对
TEST\_FILE 变量对应的测试文件，抽取特征，并使用在训练集上训练得到的模型，对测试集进行预测。

	```
	# 训练和测试文件
	TRAIN_FILE = os.path.join(DATASET_PATH, "train_set", "Train.csv")
	TEST_FILE = os.path.join(DATASET_PATH, "valid_set", "Valid.csv")
	```
	
	模型对测试集的预测结果文件，对应于config.py的TEST\_PREDICT\_PATH变量所指的文件。
	
	```
	TEST_PREDICT_PATH = os.path.join(CWD, "predict", "test.predict")
	```
	
4. 评估脚本

	使用下面的命令获取评估结果，**Accuracy** 为最终的评估标准。
	
	```
		python evalution.py gold_file_path pred_file_path
	```
	
	gold\_file\_path 为标准答案所在的路径，pred\_file\_path 为预测文件所在的路径
	
	
### 三、目录介绍


> data: 数据目录

>> dataset

>>> train_set: 训练集所在文件夹
>>>>* Train.authorIds.txt: 训练集的所有作者列表
>>>>* Train.csv：训练集

>>> valid_set: 验证集所在文件夹
>>>>* Valid.authorIds.txt: 验证集的所有作者列表
>>>>* Valid.csv：验证集
>>>>* Valid.gold.csv：验证集的标准答案

>>* Author.csv: 作者数据集

>>* coauthor.json: 共作者数据

>>* Conference.csv: 会议数据集

>>* Journal.csv: 期刊数据集

>>* Paper.csv：论文数据集

>>* PaperAuthor.csv: 论文-作者 数据集

>>* paperIdAuthorId_to_name_and_affiliation.json：(paperId，AuthorId)对 到(name1##name2; aff1##aff2)的映射关系
 
> feature: 特征文件夹

> model: 模型文件夹

>model_trainer: 模型训练器

>* coauthor.py: 获取共作者

>* data_loader.py: 加载数据

>* evalution.py: 评估脚本

>* feature_functions.py: 特征函数

>* make_feature_file.py: 生成特征文件

>* stringDistance.py: 获取字符串距离信息

>* trainer.py: 模型训练器，**主函数**

>predict: 预测结果文件夹

>authorIdPaperId.py: (作者，论文) 对类定义

>classifier.py： 分类器，使用了策略模式。

>config.py：配置文件

>confusion_matrix.py： 评估脚本所使用的包

>example.py： 样本类定义

>feature.py：特征类定义

>README.md: 说明文件

>util.py: 小工具类



## 任务介绍
####1. 目标：给定作者ID和论文ID，判断该作者是否写了这篇论文。


####2. 数据集描述：

1. 作者数据集: **Author.csv**。包含作者的编号（Id），名字（Name），隶属单位（affliation）信息。相同的作者可能在Author.csv数据集中出现多次，因为作者在不同会议／期刊上发表的论文，其名字可能有多个版本。例如：J. Doe, Jane Doe, 和 J. A. Doe 指的均是同一个人。

	|  字段名称 | 数据类型  | 注释 |
	|:--------------	|:-----------| ----------:|
	| Id				| int			|    作者编号 |
	| Name				| string		|    作者名称 |
	| Affiliation		| string     	|    隶属单位 |


2. 论文数据集: **Paper.csv**。包含论文的标题(title), 会议／期刊信息, 关键字(keywords)。相同的论文，可能会通过不同的数据来源获取，因此，在Paper.csv中会存在多个副本。

	|  字段名称 | 数据类型  | 注释 |
	|:--------------	|:-----------|:----------|
	| Id				| int			|    论文编号 |
	| Title			| string		|    论文标题 |
	| Year				| int     	|    论文年份 |
	| ConferrenceId	| int     	|    论文发表的会议Id |
	| JournalId		| int     	|    论文发表的期刊Id |
	| Keywords		| string     	|    关键字|
	
3. (论文-作者)数据集: **Paper-Author.csv**。包含 (论文Id-作者Id)对 的信息。该数据集是包含噪声的(noisy)，存在不正确的(论文Id-作者Id)对。也就是说，在Paper-Author.csv中的(论文Id-作者Id)，该作者Id并不一定写了该论文Id。因为，作者名字存在歧义（存在同名的人），和作者名字存在多个版本（如上面的例子：J. Doe, Jane Doe, 和 J. A. Doe 指的均是同一个人）。


	|  字段名称 		| 数据类型		| 注释       |
	|:--------------	|:-----------| ----------:|
	| PaperId			| int			|    论文编号 |
	| AuthorId		| int			|    作者编号 |
	| Name				| string		|    作者名称 |
	| Affiliation		| string     	|    隶属单位 |
	

4. 由于每篇论文要么发表在会议上，要么发表在期刊上。因此，这里提供关于会议和期刊的信息：**Conference.csv**, **Journal.csv** 

	|  字段名称 		| 数据类型		| 注释       |
	|:--------------	|:-----------| :----------|
	| Id				| int			|    会议／期刊 编号 |
	| ShortName		| string		|    简称|
	| Fullname		| string		|    全称 |
	| Homepage		| string     	|    主页 |
	
5. 训练集, **Train.csv**。ComfirmedPaperIds列对应的论文，表示该作者写了这些论文。DeletedPaperIds列对应的论文，表示该作者没有写这些论文。目前，已给出验证集（Valid.csv）的标准答案：**Valid.gold.csv**。Valid.gold.csv文件的格式与Train.csv格式相同。

	|  字段名称 		| 数据类型		| 注释       |
	|:--------------	|:-----------| :----------|
	| AuthorId		| int		   |      作者ID|
	| ComfirmedPaperIds		| string		|    以空格分割的论文(PaperId) 列表|
	| DeletedPaperIds		| string		|    以空格分割的论文(PaperId) 列表 |
	
	

6. 验证集和测试集，**Valid.csv, Test.csv**。测试集Test.csv，将在之后发布。格式如下：
	
	|  字段名称 		| 数据类型		| 注释       |
	|:--------------	|:-----------| :----------|
	| AuthorId		| int		   |      作者ID|
	| PaperIds		| string		|    以空格分割的论文(PaperId) 列表，待测的论文列表|
	
	
7. **coauthor.json**, 从 Paper-Author.csv 中抽取的共同作者的信息。该文件可以通过运行 model_trainer 下的 coauthor.py 来获取。
	
	```
	python coauthor.py
	```
	目前，coauthor.json存取的是每个作者合作频率最高的10个共同作者。可以通过修改coauthor.py 中 get\_top\_k\_coauthors (paper\_author\_path, k, to\_file)方法的 **k** 值来获取top k 的共同作者：
	
	```
	k = 10
    get_top_k_coauthors(
        os.path.join(config.DATASET_PATH, "PaperAuthor.csv"),
        k,
        os.path.join(config.DATA_PATH, "coauthor.json"))
	```
	
	coauthor.json的内容格式形如：
	
	```
	{"A作者ID": {"B1作者ID": 合作次数, "B2作者ID": 合作次数}}
	```
	
	第一层的key为作者的ID，对应的value为共同作者信息（同样为key-value形式，key为共同作者的ID，value为合作次数）。
	
	例如，获取作者ID为 ‘742736’ 的共同作者信息，可以通过以下代码获取，coauthor["742736"] 值对应的是ID为 ‘742736’ 作者的共同作者信息。```u'823230': 3``` 表示 ID为 ‘742736’ 的作者 和 ID为 ‘823230’ 的作者共合作过 3 次：
	
	```
	>>> import json
	>>> coauthor = json.load(open("coauthour.json"))
	>>> coauthor["742736"]
	{u'823230': 3, u'647433': 3, u'1691202': 3, u'891164': 3, 		u'1910552': 3, u'607259': 3, u'2182818': 7, u'1355775': 4, 		u'2097154': 3, u'1108518': 3}
		
	```
	
8. **paperIdAuthorId\_to\_name\_and\_affiliation.json**, 存储的是 **IDEAs** 下 **1. 字符串距离** 中描述的信息。该文件可以通过运行 model\_trainer 下的 stringDistance.py 来获取：

	```
	python stringDistance.py
	```
	
	文件内容为key-value形式，key 为论文Id和作者ID的pair对：'paperid|authorid', value为 ```{"name": "name1##name2##name3", "affiliation": "aff1##aff2##aff3"}```。 
	
	例如，获取ID为 ‘1156615’ 的论文和ID为 ‘2085584’  的作者 name 和 affiliation 信息：
	
	```
	>>> import json
	>>> pa_name_aff = json.load(open("paperIdAuthorId_to_name_and_affiliation.json"))
	>>> pa_name_aff['1156615|2085584']
	    {u'affiliation': u'Huawei##Microsoft Research Asia', u'name': u'Hang Li##Hang Li'}
	```
	

####3. 提交格式：
最终提交的的文件为对**“测试集”**的预测结果。预测结果文件的格式与训练集的格式相同，包含AuthorId、ComfirmedPaperIds、DeletedPaperIds 字段。

####4. 评估标准：
使用在“测试集”上的准确率 **Accuracy**，作为最后的评估标准。
 
评估脚本位于model\_trainer文件夹下，名为 evalution.py，通过运行该脚本可以获得评估结果。
	
```
python evalution.py gold_file_path pred_file_path
```
其中，gold\_file\_path 为标准答案所在的路径，pred\_file\_path 为预测文件所在的路径


大家可以尝试不同的特征和不同的算法来提升性能，目前系统已经实现如下算法：

* Decision Tree
* Naive Bayes
* KNN
* SVM
* Logister Regreation
* Random Forest
* AdaBoost
* VotingClassifier（ensemble）
 
 
 
####5. 数据集统计

| 数据集  		| (作者-论文)对 个数 |
|:-----------|:---------------------| 
| 训练集（Train.csv）	   | 11,263                |     
| 验证集（Valid.csv）	   | 2,347		          |   
| 测试集（Test.csv）	   | 每个队伍的测试集不同, 约1,300; |   


## IDEAs

####1. 字符串距离 
首先在paperauthor里面是又噪音的，同一个（authorid,paperid）可能出现多次，我做的是把同一个（authorid,paperid）对的多个name和多个affiliation合并起来。例如

 aid,pid,name1,aff1 <br/>
 aid,pid,name2,aff2 <br/>
 aid,pid,name3,aff3 <br/>
 
得到aid,pid,name1##name2##name3,aff1##aff2##aff3,“##”为分隔符。由paperauthor里可以知道论文的name和affiliation，另一个方面我们可以根据（authorid,paperid）对中的authorid到author表里找到对应的name和affiliation，假设当前的作者论文对是(aid,pid),从paperauthor里得到的name串和affiliation串分别为name1##name2##name3,aff1##aff2##aff3,根据aid从author表找到的name和affliction分别为name-a，affliction-a，这样我们可以算字符串的距离。

算法有两种：

*  name-a 与,name1##name2##name3的距离，同理affliction-a和,aff1##aff2##aff3的距离
*  name-a分别与name1，name2，name3的距离，然后取平均，同理affliction-a和,aff1，aff2，aff3的平均距离
距离的度量：编辑距离（levenshtein distance），最长公共子序列（LCS），最长公共子串（LSS）。
这样我们就得到关于作者name和作者affiliation的字符串相似度的多个特征。

####2. coauthor信息
很多论文都有多个作者，根据paperauthor统计每一个作者的top 10（当然可以是top 20或者其他top K）的coauthor，对于一个作者论文对（aid，pid），计算ID为pid的论文的作者有没有出现ID为aid的作者的top 10 coauthor中，可以简单计算top 10 coauthor出现的个数，还可以算一个得分，每个出现pid论文的top 10 coauthor可以根据他们跟aid作者的合作次数算一个分数，然后累加。

####3. journalid，conferenceid，year
把paper表的journalid，conferenceid和year也作为特征加进去，我的理解journalid和conferenceid可以看做是论文的一个类标签（label），年份year也可以看做是一个label。

####4. keyword信息
作者A写过的论文的keyword构成一个集合X，一篇论文B的keyword构成一个集合Y，这里说的keyword是论文的title和keyword分词后得到的单词，对于一个作者论文对（A，B）计算他们的keyword的交集：X∩Y。
每个单词可以类似tf-idf的分数，最后把属于X∩Y的单词的分数累加起来作为一维新的特征。

####5.其他
后面做了一下model的ensemble。把knn，svm，sgd分类器，rf随机森林，gbdt，logistic regression，adaboost的结果合并。













	
	
	


		



