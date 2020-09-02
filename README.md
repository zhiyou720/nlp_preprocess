##

输入HTML格式文章

输出 断句结果，标点符号矫正， 段落关键词，段落摘要，段落摘要的bert向量512维

配置见test.py


conf文件包含配置，模型文件，较大，单独上传到了nas算法库中，直接下载解压就可以


- 调整停用词
`./stopwords.txt`

- 调整分词pos标签
`jieba_considered_tags.txt`

- 调整用户词典
`user_dict.txt`