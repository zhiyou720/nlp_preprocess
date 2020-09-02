#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

from nlp_pipline import PipLine
from sentence_transformers import SentenceTransformer
from keyword_main import KeyWordExtractor
from seg_main import Cutter

# punctuation and segmentation parameters
PUNCS_PATH = 'conf/punctuation.dat'
SEG_MODEL_PATH = 'conf/seg_model'
PUNC_MODEL_PATH = 'conf/punc_model'
MAX_CUT_BATCH = 502
MAX_SENTENCE_LENGTH = 128

# key word parameters
SEED = 1024
TENCENT_EMBEDDING_PATH = "conf/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.bin"
USER_DICT_PATH = 'conf/user_dict.txt'
CONSIDER_TAGS_PATH = 'conf/jieba_considered_tags.txt'
STOP_WORDS_PATH = 'conf/stopwords.txt'
SIF_WEIGHT_PATH = 'conf/sif_model/dict.txt'
BERT_MODEL_NAME = 'distiluse-base-multilingual-cased'
SIF_EMBEDDING_PATH = r'conf/sif_model/zhs.model/'
KEY_WORD_TOPICS = 3
KEY_WORD_TOP_N = 1

print('LOAD MODELs')
CUTTER = Cutter(puncs_path=PUNCS_PATH,
                seg_model_path=SEG_MODEL_PATH,
                punc_model_path=PUNC_MODEL_PATH,
                max_cut_batch=MAX_CUT_BATCH,
                max_sentence_length=MAX_SENTENCE_LENGTH)

SENT_TRANS_MODEL = SentenceTransformer(BERT_MODEL_NAME)

KEY_WORD_EXTRACTOR = KeyWordExtractor(user_word_path=USER_DICT_PATH,
                                      puncs_path=PUNCS_PATH,
                                      considered_tags_path=CONSIDER_TAGS_PATH,
                                      stop_word_path=STOP_WORDS_PATH,
                                      sif_embedding_path=SIF_EMBEDDING_PATH,
                                      sif_weight_path=SIF_WEIGHT_PATH,
                                      tencent_embedding_path=TENCENT_EMBEDDING_PATH,
                                      random_seed=SEED)

print('FINISHED LOAD')

SEG_PROCESSOR = CUTTER.seg_processor
KEY_WORD_PROCESSOR = KEY_WORD_EXTRACTOR.key_word_func



pip_line = PipLine(seg_processor=SEG_PROCESSOR,
                   sentence_transformer=SENT_TRANS_MODEL,
                   key_word_func=KEY_WORD_PROCESSOR,
                   keyword_max_k=KEY_WORD_TOPICS, keyword_top_n=KEY_WORD_TOP_N
                   )

content = "<div class=\"RichText ztext Post-RichText\">\n <p>对于罗杰克劳利和他的“地中海史诗三部曲”，早知大名，如今终于读罢，" \
          "感觉酣畅淋漓，克劳利让人激赏的不光是他的历史学识沉淀，更有精湛的文笔，当然，也要感谢译者的漂亮翻译，让这种文笔气质得以在中文版重现。" \
          "从君士坦丁堡陷落到地中海海洋混战，一直到威尼斯海洋霸权的史诗，其实这三本书的角度和视角并不一致，但是总有一种主题和精神气质贯穿始终，" \
          "基督文明和伊斯兰文明在地中海的碰撞，献上了交织着血与泪的史诗。</p>\n <p class=\"ztext-empty-paragraph\"><br></p>\n " \
          "<figure style=\"margin-left: 0px;margin-right: 0px\">\n  \n   <img src=\"https://pic2.zhimg.com/v2-63cc" \
          "0151b07dc27ec1265b18210952fd_b.jpg\" data-rawwidth=\"892\" data-rawheight=\"587\" class=\"origin_image zh" \
          "-lightbox-thumb\" width=\"892\" data-original=\"https://pic2.zhimg.com/v2-63cc0151b07dc27ec1265b18210952fd" \
          "_r.jpg\">\n  \n </figure>\n <p class=\"ztext-empty-paragraph\"><br></p>\n <p>尽管下笔聚焦点力度不同，但是克劳" \
          "利的视角一直从时间到空间上站得高，看的广，哪怕这个系列中聚焦点最为集中的《1453：君士坦丁堡之战》。围绕这次战役，克劳利做了足够的" \
          "铺垫，从君士坦丁堡的历次危局，到双方皇帝的背景与性格、气质，还有基督世界的分裂，技术进步对战争的改变，克劳利给予了大信息流，但是又" \
          "逻辑缜密顺畅的衔接。克劳利对战争进程的描写是其文笔技法中，我个人感觉最为出色的，从格局切换的应对自如，到战争细节的聚焦描写，都很有" \
          "代入感。在他的文字组织下，你既能感觉战争中英雄主义的光辉，又能体会到个体在战争中挣扎的血泪，比如对一次次攻城战，从双方对垒、" \
          "兵器使用、战斗细节都场面感十足，从大局到细节构成了一种战争史诗的效果。</p>\n <p class=\"ztext-empty-paragraph\"><br>" \
          "</p>\n <p>事实上，攻城战在克劳利的三部曲中，写了太多场，整套书，由于整体处于伊斯兰势力进攻，欧洲人防守的局面，所以整个氛围常常是" \
          "一群欧洲人，孤独抵抗强大的攻城军队的悲壮故事。通常，在第三方看来，守城一方容易得到更多同情，毕竟他们是在守卫家园，抵御入侵，而且通" \
          "常处于一种实力不对等的战争中，所以悲剧感也是扑面而来。而且，克劳利也特别注意将视角从激烈的攻城战现场，转移到基督文明的其它地域，往" \
          "往，这里的故事更加凸显守城军民的悲剧性命运。我们看到面对强敌，基督文明世界整体上一盘散沙，内斗严重，在增援决策上效率低下，相互推诿" \
          "，常常调子很高，却轻轻放下，这倒是与同样以信仰为名战斗，但是却整体形成高效战争机器的伊斯兰势力形成鲜明对比。也正是这样的背景下，克" \
          "劳利给了很多勇敢的，甚至自身前往被困城市的个体以更多笔墨，突出了这些人的英雄之举。</p>\n <p class=\"ztext-empty-paragrap" \
          "h\"><br></p>\n <figure style=\"margin-left: 0px;margin-right: 0px\">\n  \n   <img src=\"https://pic4.zhi" \
          "mg.com/v2-3186a2f38722e21c191e8322bebc77a7_b.jpg\" data-rawwidth=\"300\" data-rawheight=\"441\" class=\"co" \
          "ntent_image\" width=\"300\">\n  \n </figure>\n <p>但是，如果你认为克劳利站在西方视角，刻意强化这种悲壮就错了，克劳利对" \
          "当时的伊斯兰文明世界给予了充分客观的描写，从他们的治理结构，到科学、商业、文化的成就。比如君士坦丁堡之战，土耳其苏丹默罕默德二世" \
          "，在作者笔下就是一位年轻英武，谋断能力极强的君主。虽然他也时有暴虐之举，此后直到他死亡更是成为欧洲人的噩梦，但是在这场战役中，他" \
          "体现的更是一种坚韧和果决，在战后，尽管也有城市惨遭洗劫的悲剧（克劳利也在书中承认，在攻城洗劫上，欧洲人与之相比也是半斤对八两），但" \
          "是也是他及时制止，并将这座城市建设成为新的首都，继续其征伐之旅。而与他交战的君士坦丁十一世，一个史书上常常给予弱势刻画的人物，在本" \
          "书中也有了更加生动的形象，在那个特别时期，他的勇气与勇敢，直到最后坚守到死的壮举，都无愧为一个皇帝的荣耀。</p>\n <p class=\"zt" \
          "ext-empty-paragraph\"><br></p>\n <figure style=\"margin-left: 0px;margin-right: 0px\">\n  \n   <img src=\"ht" \
          "tps://pic3.zhimg.com/v2-b6529b2a2638cdc61b1c18aa8c90043a_b.jpg\" data-rawwidth=\"300\" data-rawheight=\"44" \
          "1\" class=\"content_image\" width=\"300\">\n  \n </figure>\n <p>当然，围绕地中海，主角还是海洋，正如君士坦丁堡被" \
          "攻陷后，对欧洲人的震动象征意义大于经济意义，此后，大航海时代逐步开始，让君士坦丁堡从地缘政治上的地位变得这真的更具象征意义。而大航" \
          "海时代的基础，正是地中海，这一文明交流与碰撞的“文明之湖”的长期航海历史所孕育。本系列的另外两部《海洋帝国》、《财富之城》更聚焦于" \
          "海上的战斗，在《海洋帝国》中，伊斯兰世界的几位海盗显得残暴，但是其统治力令人印象深刻，而基督文明显得松散但充满韧性的抵抗和反击，也" \
          "带来了几场悲壮的战役。而事实上，明知不可无而为之堪称悲壮，但当时的另一个现实却是基督文明下的一个个城市这样的“点”被攻击的时候，" \
          "整个基督文明的各方势力却无法形成一个“面”去支持，让这些抗争置身于一种明知可为却不为的孤独大背景下，这，又让人感觉是一场悲剧。</" \
          "p>\n <p class=\"ztext-empty-paragraph\"><br></p>\n <figure style=\"margin-left: 0px;margin-right: 0px\"" \
          ">\n  \n   <img src=\"https://pic2.zhimg.com/v2-38f6022c8d7c4e03337ec930c309bcd1_b.jpg\" data-rawwidth=\"" \
          "318\" data-rawheight=\"417\" class=\"content_image\" width=\"318\">\n  \n </figure>\n <p>《财富之城》聚焦威" \
          "尼斯共和国的历史，威尼斯共和国在地缘政治影响力上，从表面上，似乎有些后来的大不列颠王国、今天的美国的感觉，并不追求绝对的领土控" \
          "制，而是以经济利益为纽带进行征伐、怀柔与治理。但是，正如历史上对威尼斯人的脸谱化的不佳刻画一样，威尼斯的扩张却少了很多软实力的同步" \
          "跟进。第四次十字军东征可谓一个典型，作为威尼斯共和国扮演重要角色的东征，这是一次那个时期罕见的基督文明的强势进击，但是攻打的却是同" \
          "是基督文明下的君士坦丁堡。那时的威尼斯执政官丹多洛，确实堪称传奇人物，但是其身上也体现了那个时候威尼斯对外政策的特点：势利。是的，" \
          "不是“务实”，而是一种有时候到了见利忘义，目光短浅程度的“势利”，这种软实力的虚弱，也使得威尼斯共和国始终无法构筑强势的联盟，伴随着" \
          "时代的变迁，也只能走向衰落。</p>\n <p class=\"ztext-empty-paragraph\"><br></p>\n <figure style=\"margin-left: " \
          "0px;margin-right: 0px\">\n  \n   <img src=\"https://pic1.zhimg.com/v2-6c4b657d4dd526a1bc9bcfde9f7909ec_b.j" \
          "pg\" data-rawwidth=\"840\" data-rawheight=\"480\" class=\"origin_image zh-lightbox-thumb\" width=\"840\" da" \
          "ta-original=\"https://pic1.zhimg.com/v2-6c4b657d4dd526a1bc9bcfde9f7909ec_r.jpg\">\n  \n </figure>\n <p>地中海" \
          "史诗三部曲是一套可读性极强的历史读物，一种代入感极强，让你感觉可触摸的历史氛围营造之下，你可以体察那个时代的动荡与不安。曾经，" \
          "两个文明殊死搏斗，仿佛必须有一方被征服，但是我们发现，尽管那个时代伊斯兰势力咄咄逼人，基督教艰难抵抗，但是地理上的格局似乎也" \
          "就僵持在地中海周围，形成微妙的均衡。无疑，那是个让当时欧洲人容易感到绝望的时代，而今天，我们发现，尽管没有刀光剑影，相反，倒" \
          "是很多流离失所的中东难民由地中海的涌入，以及欧洲穆斯林人口比例的增加引发了新的争议和对未来的不安。这种不安，在规模不一，但" \
          "是明显更加频繁的发生于欧洲的恐怖袭击事件之下，显得更加真实。在此背景下，回顾曾经围绕地中海的那从文明冲突或许能给我们一些思" \
          "考的养料，至少，那段岁月之后，欧洲没有完蛋，倒是在融合中进入了大航海时代，形成了欧洲国家的一个世界性扩张的强势周期。历史并" \
          "不见得会重复，但是，我依然相信在不时发生的文明碰撞之中，伴随人类整体文明程度的提高，激烈程度会比以往降低，而在纠结与冲突中，" \
          "新的平衡总会达到，而置身其中的人，在今天也将有更多理性思考和行动的空间。 </p>\n <p></p>\n</div>"

start_time = time.time()

content = pip_line.process_data(content,
                                generate_key_word=True,
                                generate_summary=True,
                                generate_summary_bert_vector=True)
end_time = time.time()

print("Time used: {}s".format(round(end_time - start_time), 4))

[print(x) for x in content]

question = '你们喜欢去地中海旅游吗？，有没有什么好玩的地方和好吃的东西推荐一下。'
answer = [
    '喜欢去地中海旅游，喜欢吹海风，海风很凉爽。',
    '不太喜欢地中海，气候太闷热潮湿了。听说那边风土人情不是很好。',
    '喜欢地中海，但是没钱旅游。',
    '我去过地中海旅游，那边海鲜非常好吃。',
    '可以去君士坦丁堡，感受那边的历史文化。'

]
start_time = time.time()

content = pip_line.process_question_post(question=question,
                                         answers=answer,
                                         generate_key_word=True,
                                         generate_summary=True,
                                         generate_summary_bert_vector=True)
end_time = time.time()
print("Time used: {}s".format(round(end_time - start_time), 4))
[print(x) for x in content]



# content = """<div>
#  <div class="rich_media_content " id="js_content" style="visibility: hidden;">
#   <section style="font-size: 16px;" data-mpa-powered-by="yiban.io">
#    <section style="text-align: center;margin-right: 0%;margin-left: 0%;" powered-by="xiumi.us">
#     <section style="vertical-align: middle;display: inline-block;line-height: 0;border-width: 1px;width: 95%;box-shadow: rgb(160, 160, 160) 3.5px 3.5px 2px;border-radius: 0px;border-style: solid;border-color: rgb(160, 160, 160);">
#      <img data-ratio="1.988" data-src="https://mmbiz.qpic.cn/sz_mmbiz_jpg/2vxkcwfFLHaI0OBBEcFicCiaxYdUHm5XxiaAooCdk4vwXaCRyCh3A053ddmVLDpcD8Xfne4lH6l5yBiaWlcmtJtibsg/640?wx_fmt=jpeg" data-type="jpeg" data-w="750" style="vertical-align: middle;width: 100%;box-sizing: border-box;">
#     </section>
#    </section>
#   </section>
#   <p style="font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif;letter-spacing: 0.544px;white-space: normal;background-color: rgb(255, 255, 255);"><span style="font-size: 16px;color: rgb(2, 30, 170);"></span></p>
#   <p style="font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif;letter-spacing: 0.544px;white-space: normal;background-color: rgb(255, 255, 255);"><span style="font-size: 16px;color: rgb(2, 30, 170);">本文为作者应中国科协约稿所作，原发于科协公众号“科创中国”</span><br></p>
#   <p style="font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif;letter-spacing: 0.544px;white-space: normal;background-color: rgb(255, 255, 255);"><br></p>
#   <p style="font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif;letter-spacing: 0.544px;white-space: normal;background-color: rgb(255, 255, 255);"><span style="font-size: 18px;">近来，美国对华为等中国科技企业的限制和制裁，越来越升级。简单来说，就是从半导体行业尤其是芯片制造上对中国企业进行各种限制。今天，我们就来说说芯片、半导体和其他行业的技术发展情况。<br></span></p>
#   <p style="font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif;letter-spacing: 0.544px;white-space: normal;background-color: rgb(255, 255, 255);"><br></p>
#   <section data-tools="新媒体排版" data-id="916998" data-style-type="undefined" style="font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif;letter-spacing: 0.544px;white-space: normal;background-color: rgb(255, 255, 255);">
#    <section data-id="92360">
#     <section style="border-width: 0px;border-style: none;border-color: initial;">
#      <section style="text-align: center;">
#       <section style="margin-right: auto;margin-left: auto;padding-right: 10px;padding-left: 10px;display: inline-block;border-width: 20px;border-style: solid;-webkit-border-image: url(&quot;https://mmbiz.qpic.cn/mmbiz_png/uN1LIav7oJibnXRTwzul99YfpqjjVFWoFFzb3SHIMbibQzTLznChicrNSdMpiatIgQRCf6GWVJLdKq6Pua5aF3GN3g/640?wx_fmt=png&quot;) 20 fill;">
#        <section data-brushtype="text" style="font-size: 18px;color: rgb(255, 255, 255);line-height: 25px;">
#         <strong>国家很早就重视半导体，为什么现在我们还被限制？</strong>
#        </section>
#       </section>
#      </section>
#     </section>
#    </section>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">中国早就在大力推动半导体行业的发展，为什么现在还被美国限制，且受影响面如此大呢？回顾历史，中国半导体行业的起步并不算太晚。1953年，中国第一个五年计划期间，苏联援建的北京电子管厂（774厂）一度是亚洲最大的半导体晶体管厂。它也是京东方的前身。</span></p>
#    <p><br></p>
#    <p style="text-align: center;"><img class="rich_pages" data-ratio="0.66375" data-s="300,640" data-type="jpeg" data-w="800" data-src="https://mmbiz.qpic.cn/mmbiz_jpg/xx8bMlKiajpAicGTCMUwDia2w2YO2Yt5ZJgu5kZKjDyIV33o0VaVJRA57NxCk9N98EcEUu9s97IVpFRRAy9KVNU9Q/640?wx_fmt=jpeg" style="box-sizing: border-box !important;width: 677px !important;visibility: visible !important;"></p>
#    <p><br><span style="font-size: 18px;color: rgb(0, 0, 0);"></span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">改革开放后又先后进行了发展集成电路的“908”和“909”工程。虽然，我们在相当长时期与世界先进水平一直有差距，但在美国开始全面在半导体领域对中国限制之前，这个差距已经大大缩小了。但是，因为多方面的原因，这个领域中国追赶世界先进水平的困难远大于其他领域，原因如下：</span></p>
#    <p><br></p>
#    <p><span style="color: rgb(61, 170, 214);"><strong><span style="font-size: 18px;">技术发展太快</span></strong></span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">可以说，</span><span style="font-size: 18px;color: rgb(110, 170, 215);"><strong>半导体芯片是当今世界科技发展速度最快的领域。</strong></span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">根据</span><span style="font-size: 18px;color: rgb(110, 170, 215);"><strong>摩尔定律</strong></span><span style="font-size: 18px;color: rgb(0, 0, 0);">，芯片的集成度、运算能力等每18个月提升一倍。并且，这个定律从半导体集成电路技术出现的一开始，一直到2010年之后都与实际发展过程高度吻合。2010年之后虽然摩尔定律的速度有所放慢，但只是倍增的周期有所拉长，芯片行业仍在以</span><span style="font-size: 18px;color: rgb(110, 170, 215);"><strong>指数规律</strong></span><span style="font-size: 18px;color: rgb(0, 0, 0);">发展。</span></p>
#    <p><br></p>
#    <p><strong><span style="font-size: 18px;color: rgb(110, 170, 215);">这样的发展速度在人类科技史上是空前的。</span></strong><span style="font-size: 18px;color: rgb(0, 0, 0);">它也带动了整个IT业以指数的速度迅猛发展。正因为这个领域的技术发展太快，使得学习模仿的后发优势难以得到发挥——学习的速度跟不上领先者进步的速度。</span></p>
#    <p><br></p>
#    <p style="text-align: center;"><img class="rich_pages" data-ratio="0.65125" data-s="300,640" data-type="jpeg" data-w="800" data-src="https://mmbiz.qpic.cn/mmbiz_jpg/xx8bMlKiajpAicGTCMUwDia2w2YO2Yt5ZJgicV9o6CFNryoU9K2SxpibCviahzaUAB0sHnRJbLD7e8afKpSkWibuEia6Zg/640?wx_fmt=jpeg" style="box-sizing: border-box !important;width: 677px !important;visibility: visible !important;"></p>
#    <p><br><span style="font-size: 18px;color: rgb(0, 0, 0);"></span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">虽然表面看来芯片集成度以规律性的速度在提升，但并不意味着只是单纯的技术指标进步，而是每过一段时间，基本技术架构、生产技术工艺体系等都可能发生颠覆性的变化，并不是原来技术体系改进性的进步就可以长期满足需要的。一旦在相应颠覆性技术进步时方向判断失误，会使原来的芯片巨头很快陷入衰落。</span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">现在中国在芯片领域呈现了超越的势头，一方面是中国技术全面的进步和积累，另一方面也是芯片技术进步的速度在放慢，这给了赶超者更好的条件。</span></p>
#    <p><br></p>
#    <p><span style="color: rgb(61, 170, 214);"><strong><span style="font-size: 18px;">涉及领域太广</span></strong></span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">芯片上游产业涉及的领域太广，使得单纯某些领域的进步甚至突破，远远不足以跟上整个芯片行业的发展。例如，现在中国刻蚀设备已经世界领先，可以做到5纳米了，芯片设计能力也不错，封装能力也很高。</span></p>
#    <p><br></p>
#    <p style="text-align: center;"><img class="rich_pages" data-ratio="0.65125" data-s="300,640" data-type="jpeg" data-w="800" data-src="https://mmbiz.qpic.cn/mmbiz_jpg/xx8bMlKiajpAicGTCMUwDia2w2YO2Yt5ZJgDoCStd3Ux9DG4LIJZ0l9vZgmnhZ5567DzCCjGfvPG5VAox35qBjFtg/640?wx_fmt=jpeg" style="box-sizing: border-box !important;width: 677px !important;visibility: visible !important;"></p>
#    <p><br><span style="font-size: 18px;color: rgb(0, 0, 0);"></span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">但芯片生产涉及的设备、材料太多，上下游产业链太长。因此，如果不能全面突破，只要有一个关键点上存在薄弱点，理论上就存在被限制的可能。又因为技术发展太快，带动整个产业链所有环节都跟着一起在技术上高度变化，这使后发者要进行追赶的确困难重重。</span></p>
#    <p><br></p>
#    <p><span style="color: rgb(61, 170, 214);"><strong><span style="font-size: 18px;">美国领先并最为重视</span></strong></span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">从半导体技术出现，直到现在的芯片，主要技术发明和驱动者都是美国。芯片是信息产业的基石，而信息产业又是今天信息社会的基石，是大量其他领域技术进步的核心驱动力，因此具有重大的战略价值。包括军事技术的进步等，主要都体现在芯片和信息技术的进步上。</span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">因此，美国从政府到产业界、金融界对芯片技术都极为重视，并且大力推动其技术进步。其他国家或地区发展部分芯片产业是可以的，一旦呈现全面发展的态势，美国就会全力打压。例如，光刻机现在是荷兰ASML居领先地位，芯片生产台积电最领先，日本和韩国都在芯片部分领域领先，这些美国可以接受。</span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">但当年美日贸易战时，也配合进行了科技战，其中半导体芯片就是美国重点打击日本的领域之一，就是因为日本在芯片领域呈现出了全面超越的趋势。现在美国对中国芯片业全面进行打压，也是因为</span><strong><span style="font-size: 18px;color: rgb(110, 170, 215);">中国开始在芯片领域呈现出了全面超越的迹象。</span></strong></p>
#    <p><br></p>
#    <p style="text-align: center;"><img class="rich_pages" data-ratio="0.66625" data-s="300,640" data-type="jpeg" data-w="800" data-src="https://mmbiz.qpic.cn/mmbiz_jpg/xx8bMlKiajpAicGTCMUwDia2w2YO2Yt5ZJgic6JTibQwx7WfqP9H0yGicse55iaEicWzu753EAlrQSPNZszicSSW2tKF8jg/640?wx_fmt=jpeg" style="box-sizing: border-box !important;width: 677px !important;visibility: visible !important;"></p>
#    <p><br></p>
#    <section data-id="92360">
#     <section style="border-width: 0px;border-style: none;border-color: initial;">
#      <section style="text-align: center;">
#       <section style="margin-right: auto;margin-left: auto;padding-right: 10px;padding-left: 10px;display: inline-block;border-width: 20px;border-style: solid;-webkit-border-image: url(&quot;https://mmbiz.qpic.cn/mmbiz_png/uN1LIav7oJibnXRTwzul99YfpqjjVFWoFFzb3SHIMbibQzTLznChicrNSdMpiatIgQRCf6GWVJLdKq6Pua5aF3GN3g/640?wx_fmt=png&quot;) 20 fill;">
#        <section data-brushtype="text" style="font-size: 18px;color: rgb(255, 255, 255);line-height: 25px;">
#         <strong>芯片技术的赶超，为啥特别难？</strong>
#        </section>
#       </section>
#      </section>
#     </section>
#    </section>
#    <p><span style="font-size: 18px;"></span><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">正如前面所介绍的，中国在芯片技术上赶超的过程特别艰难，是有其特殊产业原因的。作为对比，其他技术领域中国赶超起来要容易很多。在国防、军事等尖端科技领域，从我们建国到现在，始终受到“巴统”——“瓦森纳协议”等限制的，例如2013年美国推动日本对中国高模量的碳纤维进行禁运。</span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(110, 170, 215);"><strong>尖端技术不仅无法通过引进学习，甚至产品都难以买到。</strong></span><span style="font-size: 18px;color: rgb(0, 0, 0);">中国在长期的受限制甚至禁运的环境中，不断完成了绝大多数科技领域接近甚至赶超世界先进水平的历史积累。芯片是剩下的美国可以有效对中国进行限制的几乎唯一领域。</span></p>
#    <p><span style="font-size: 18px;"></span></p>
#    <section data-id="92360">
#     <section style="border-width: 0px;border-style: none;border-color: initial;">
#      <section style="text-align: center;">
#       <section style="margin-right: auto;margin-left: auto;padding-right: 10px;padding-left: 10px;display: inline-block;border-width: 20px;border-style: solid;-webkit-border-image: url(&quot;https://mmbiz.qpic.cn/mmbiz_png/uN1LIav7oJibnXRTwzul99YfpqjjVFWoFFzb3SHIMbibQzTLznChicrNSdMpiatIgQRCf6GWVJLdKq6Pua5aF3GN3g/640?wx_fmt=png&quot;) 20 fill;">
#        <section data-brushtype="text" style="font-size: 18px;color: rgb(255, 255, 255);line-height: 25px;">
#         <strong>我们有哪些技术世界领先？</strong>
#        </section>
#       </section>
#      </section>
#     </section>
#    </section>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">从总体上说，中国科技已经处于越来越跟紧美国，并开始甩开第二梯队的欧洲与日本的地位。有部分领域的超越，如在大型的工程机械、高铁、5G系统、导弹、量子技术、太阳能、风能等。</span></p>
#    <p><br></p>
#    <p style="text-align: center;"><img class="rich_pages" data-ratio="0.64375" data-s="300,640" data-type="jpeg" data-w="800" data-src="https://mmbiz.qpic.cn/mmbiz_jpg/xx8bMlKiajpAicGTCMUwDia2w2YO2Yt5ZJgdgNrQ3RiaMn9q8xgxFyE2NabtRAeSJwp6meTGhDibdSAIgEtziayk9o6g/640?wx_fmt=jpeg" style="box-sizing: border-box !important;width: 677px !important;visibility: visible !important;"></p>
#    <p><br><span style="font-size: 18px;color: rgb(0, 0, 0);"></span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">客观地说，现在中国超越美欧的领域，一部分是通过改进性的提升（功能更多更适用），量上的扩大（如把机械设备做得更大），产能更大，成本更低等获得的，真正原创性的超越还是比较少的。</span></p>
#   </section>
#   <section data-tools="新媒体排版" data-id="916998" data-style-type="undefined" style="font-family: -apple-system-font, BlinkMacSystemFont, &quot;Helvetica Neue&quot;, &quot;PingFang SC&quot;, &quot;Hiragino Sans GB&quot;, &quot;Microsoft YaHei UI&quot;, &quot;Microsoft YaHei&quot;, Arial, sans-serif;letter-spacing: 0.544px;white-space: normal;background-color: rgb(255, 255, 255);">
#    <p><br></p>
#    <section data-id="92360">
#     <section style="border-width: 0px;border-style: none;border-color: initial;">
#      <section style="text-align: center;">
#       <section style="margin-right: auto;margin-left: auto;padding-right: 10px;padding-left: 10px;display: inline-block;border-width: 20px;border-style: solid;-webkit-border-image: url(&quot;https://mmbiz.qpic.cn/mmbiz_png/uN1LIav7oJibnXRTwzul99YfpqjjVFWoFFzb3SHIMbibQzTLznChicrNSdMpiatIgQRCf6GWVJLdKq6Pua5aF3GN3g/640?wx_fmt=png&quot;) 20 fill;">
#        <section data-brushtype="text" style="font-size: 18px;color: rgb(255, 255, 255);line-height: 25px;">
#         <strong>我们自身该如何发展？</strong>
#        </section>
#       </section>
#      </section>
#     </section>
#    </section>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">如此说来，似乎中国的科技很快就会领先世界了。是否如此呢？可以说有这个潜力，但很多人也会感觉差距还非常远，可以说两种判断都正确，原因何在？如果单纯从一个又一个技术领域来分析，可以说中国与世界最领先的水平差距都不远了。只要再努力一下，超上或超过距离都不太远。但另一方面，就是这“最后一百米”，却又是难如登天的。关键原因在于：</span></p>
#    <p><br></p>
#    <p><span style="color: rgb(61, 170, 214);"><strong><span style="font-size: 18px;">追赶与超越是完全不同的两个概念</span></strong></span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">追赶时既有</span><span style="font-size: 18px;color: rgb(110, 170, 215);"><strong>后发劣势</strong></span><span style="font-size: 18px;color: rgb(0, 0, 0);">，也有</span><span style="font-size: 18px;color: rgb(0, 0, 0);">后发优势</span><span style="font-size: 18px;color: rgb(0, 0, 0);">。后发劣势有很多，它换句话说就是</span><strong><span style="font-size: 18px;color: rgb(110, 170, 215);">领先者优势</span></strong><span style="font-size: 18px;color: rgb(0, 0, 0);">。如，领先者会吸引更多人才，使落后者更加落后。领先者有更多资金，可进行更多投入。会率先建立专利保护带，有利于自己的行业规则。拥有进一步技术发展的话语权。率先研究清楚并控制相应的产业链。率先建立各种行业保护带。无人拦阻，反过来可以对追赶者进行各种遏制、专利打压等。</span></p>
#    <p><br></p>
#    <p style="text-align: center;"><img class="rich_pages" data-ratio="0.61875" data-s="300,640" data-type="jpeg" data-w="800" data-src="https://mmbiz.qpic.cn/mmbiz_jpg/xx8bMlKiajpAicGTCMUwDia2w2YO2Yt5ZJgQIiaP9KQgoyNLRdDI4kzJusEFhasfg33ckCtKyyrSic1xjI2opmxHzaQ/640?wx_fmt=jpeg" style="box-sizing: border-box !important;width: 677px !important;visibility: visible !important;"></p>
#    <p><br><span style="font-size: 18px;"></span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">当然，追赶者也有</span><span style="font-size: 18px;color: rgb(110, 170, 215);"><strong>后发优势</strong></span><span style="font-size: 18px;color: rgb(0, 0, 0);">。</span><span style="font-size: 18px;color: rgb(110, 170, 215);"><strong>学习优势</strong></span><span style="font-size: 18px;color: rgb(0, 0, 0);">，这是追赶者最大的后发优势。学习有充分的可参考的知识信息，避免各种领先者探索的风险，避免领先者已经犯过的错误。</span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);"></span><strong><span style="font-size: 18px;color: rgb(110, 170, 215);">产业化优势</span></strong><span style="font-size: 18px;color: rgb(0, 0, 0);">，新的产业开创需要新开拓很多产业链，一个企业往往难以去建立所有的产业环节。因此，领先战略的长期执行往往并不是单一企业可以完成的，它需要大量产业链条上各个环节的共同创新和进步。这对领先者是一个巨大的挑战。对于追赶者来说，整个产业链已经成熟了，因此可以直接利用整个产业链的成熟资源，自己只要做一个最有优势的环节就可以了。</span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">&nbsp;&nbsp;&nbsp;</span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">中国在从追赶瞬间转向领先者战略时，需要从学习和模仿国外具体的先进技术，转向学习和模仿其原创能力。原创能力它是有规律的，我们现在最需要的就是研究和掌握其内在规律，使中国的科技战略从跟随型尽快转向原创型。</span></p>
#    <section data-id="92360">
#     <section style="border-width: 0px;border-style: none;border-color: initial;">
#      <section style="text-align: center;">
#       <section style="margin-right: auto;margin-left: auto;padding-right: 10px;padding-left: 10px;display: inline-block;border-width: 20px;border-style: solid;-webkit-border-image: url(&quot;https://mmbiz.qpic.cn/mmbiz_png/uN1LIav7oJibnXRTwzul99YfpqjjVFWoFFzb3SHIMbibQzTLznChicrNSdMpiatIgQRCf6GWVJLdKq6Pua5aF3GN3g/640?wx_fmt=png&quot;) 20 fill;">
#        <section data-brushtype="text" style="font-size: 18px;color: rgb(255, 255, 255);line-height: 25px;">
#         <strong>未来几年最大的风口</strong>
#        </section>
#       </section>
#      </section>
#     </section>
#    </section>
#    <p><br></p>
#    <p><span style="font-size: 18px;">对于未来的科技原创一个最重要的方面，是判断出主要的科技发展方向在哪里。我们今天经常用“</span><strong><span style="font-size: 18px;color: rgb(110, 170, 215);">风口</span></strong><span style="font-size: 18px;">”一词来表达，原因在于</span><span style="font-size: 18px;color: rgb(110, 170, 215);"><strong>全社会的科技推动力往往存在于科技发展最为迅速的领域。</strong></span><span style="font-size: 18px;">因为其发展最为迅猛，它的技术能力进步会进而带动其他领域的技术进步。</span></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;">信息技术的进步速度在摩尔定律的作用下带动了整个社会的技术跟着一起进步。未来是否还会出现新的这类风口技术，或者是否就只是一个领域具备这样的能力，我们很难去下结论。但有一些基本的判断是可以初步分析一下的。</span><br></p>
#    <p><br></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">芯片领域虽然越来越遇到发展的极限约束，但离真正的不可能再提升的物理极限还有一定空间。如从7纳米发展到2纳米。另外各类固体存储技术的发展，如新型闪存、相变存储、磁存储等很可能会引发另一场IT领域的革命性改变。</span></p>
#    <p><br></p>
#    <p><span style="color: rgb(110, 170, 215);"><strong><span style="font-size: 18px;">新能源领域正面临一场革命性突破的前夜。</span></strong></span><span style="font-size: 18px;color: rgb(0, 0, 0);">太阳能和风能的成本在2020年已经低于火电，并且未来有很大的成本下降空间。这样也会使新能源替代化石能源的进程大大加快。纯电动车离平价进入市场也不远，并且其改进潜力巨大。</span></p>
#    <p><br></p>
#    <p style="text-align: center;"><img class="rich_pages" data-ratio="0.66625" data-s="300,640" data-type="jpeg" data-w="800" data-src="https://mmbiz.qpic.cn/mmbiz_jpg/xx8bMlKiajpAicGTCMUwDia2w2YO2Yt5ZJgY7LB988iaPRnWdytJHstEemOxgGBHFc446gw3kznpjW7wYjhJj05oTw/640?wx_fmt=jpeg" style="box-sizing: border-box !important;width: 677px !important;visibility: visible !important;"></p>
#    <p><br><span style="font-size: 18px;color: rgb(0, 0, 0);"></span></p>
#    <p><span style="font-size: 18px;color: rgb(0, 0, 0);">这些都会使能源领域和汽车领域发生革命性的变化，并使这些变革漫延到其他领域。能源和交通是全社会的基础性产业，因此其影响极为深远。</span></p>
#   </section>
#   <p><br></p>
#   <p><br></p>
#   <section style="font-size: 16px;">
#    <section style="margin: 10px 0%;" powered-by="xiumi.us">
#     <section style="display: inline-block;vertical-align: top;width: 50%;padding-right: 5px;">
#      <section style="text-align: center;margin-right: 0%;margin-bottom: 10px;margin-left: 0%;" powered-by="xiumi.us">
#       <section style="vertical-align: middle;display: inline-block;line-height: 0;border-width: 0px;border-radius: 6px;border-style: none;border-color: rgb(62, 62, 62);overflow: hidden;box-shadow: rgb(0, 0, 0) 0px 0px 0px;">
#        <img data-ratio="1.0604839" data-src="https://mmbiz.qpic.cn/sz_mmbiz_png/2vxkcwfFLHaI0OBBEcFicCiaxYdUHm5Xxia7BWox8s2zplaBaYRaqh0iciaM6cl88Qr8bK2oR0liaKAgTqjLFTPNkuoA/640?wx_fmt=gif" data-type="gif" data-w="1984" style="vertical-align: middle;box-sizing: border-box;">
#       </section>
#      </section>
#      <section style="transform: translate3d(0px, 0px, 1px) rotateY(180deg);margin-right: 0%;margin-left: 0%;" powered-by="xiumi.us">
#       <section style="display: inline-block;vertical-align: middle;width: 15px;height: auto;align-self: center;">
#        <section style="text-align: center;margin-right: 0%;margin-left: 0%;font-size: 0px;" powered-by="xiumi.us">
#         <section style="vertical-align: middle;display: inline-block;line-height: 0;">
#          <img data-ratio="1.0948905" data-src="https://mmbiz.qpic.cn/mmbiz_svg/6mXOeYa4HU8ssOg2CQiatE76udOzgWoibN6KxKe5aOibib1E15W4bEcK7xvziacjltOckianVU9n8ZIZxJqRduxD0yHgib6hc3l6h1w/640?wx_fmt=svg" data-type="svg" data-w="137" style="vertical-align: middle;width: 300px;box-sizing: border-box;">
#         </section>
#        </section>
#       </section>
#       <section style="display: inline-block;vertical-align: middle;width: auto;align-self: center;min-width: 10%;height: auto;">
#        <section style="transform: translate3d(0px, 0px, 1px) rotateY(180deg);" powered-by="xiumi.us">
#         <section style="text-align: center;color: rgb(62, 62, 62);padding-right: 10px;padding-left: 10px;letter-spacing: 1px;line-height: 1;">
#          <p>长按识别二维码，可进入购买页面</p>
#         </section>
#        </section>
#       </section>
#      </section>
#      <section style="transform: translate3d(-12px, 0px, 0px);margin-right: 0%;margin-left: 0%;" powered-by="xiumi.us">
#       <section style="padding-right: 12px;padding-left: 12px;font-size: 12px;color: rgb(108, 108, 108);">
#        <p style="white-space: normal;"><br></p>
#       </section>
#      </section>
#     </section>
#     <section style="display: inline-block;vertical-align: top;width: 50%;padding-left: 5px;">
#      <section style="margin-right: 0%;margin-left: 0%;" powered-by="xiumi.us">
#       <section style="display: inline-block;vertical-align: middle;width: 15px;height: auto;align-self: center;">
#        <section style="text-align: center;margin-right: 0%;margin-left: 0%;font-size: 0px;" powered-by="xiumi.us">
#         <section style="vertical-align: middle;display: inline-block;line-height: 0;">
#          <img data-ratio="1.0948905" data-src="https://mmbiz.qpic.cn/mmbiz_svg/6mXOeYa4HU8ssOg2CQiatE76udOzgWoibNS0iaO0oQC5sx3wwxTKl0logPEd8Qic4CSEmJbwek4jnAhzXusEr2RmXK8jDloWlZIG/640?wx_fmt=svg" data-type="svg" data-w="137" style="vertical-align: middle;width: 300px;box-sizing: border-box;">
#         </section>
#        </section>
#       </section>
#       <section style="display: inline-block;vertical-align: middle;width: auto;align-self: center;min-width: 10%;height: auto;">
#        <section style="text-align: center;color: rgb(0, 0, 0);padding-right: 10px;padding-left: 10px;letter-spacing: 1px;line-height: 1;" powered-by="xiumi.us">
#         <p><strong>《科学经济学原理：看见看不见的手》</strong></p>
#        </section>
#       </section>
#      </section>
#      <section style="transform: translate3d(12px, 0px, 0px);margin-right: 0%;margin-left: 0%;" powered-by="xiumi.us">
#       <section style="padding-right: 12px;padding-left: 12px;font-size: 12px;color: rgb(108, 108, 108);">
#        <p style="white-space: normal;">汪涛著</p>
#       </section>
#      </section>
#      <section style="text-align: center;margin-top: 10px;margin-right: 0%;margin-left: 0%;" powered-by="xiumi.us">
#       <section style="vertical-align: middle;display: inline-block;line-height: 0;border-width: 0px;border-radius: 6px;border-style: none;border-color: rgb(62, 62, 62);overflow: hidden;width: 90%;height: auto;">
#        <img data-ratio="1.004065" data-src="https://mmbiz.qpic.cn/sz_mmbiz_png/2vxkcwfFLHaI0OBBEcFicCiaxYdUHm5Xxia1xLiankSBcJCKGOXst6QOxv5KfrcDfFWzm0RB8TvD5yhEN3VmfuNncA/640?wx_fmt=png" data-type="png" data-w="246" style="vertical-align: middle;width: 100%;box-sizing: border-box;">
#       </section>
#      </section>
#     </section>
#    </section>
#   </section>
#   <p><br></p>
#  </div>
# </div>"""
# content = pip_line.process_data(content,
#                                 generate_key_word=True,
#                                 generate_summary=True,
#                                 generate_summary_bert_vector=True)
#
# [print(x) for x in content]
#
# content = """<div>
#  <div class="RichContent-inner">
#   <span class="RichText ztext CopyrightRichText-richText" itemprop="text"><p>1949年前后中国发生了什么？</p><p>对现代史有点了解的好像都能回答这个问题。</p><p>根据历史记载，在1949年前后，中共以及人民解放军一路高歌猛进，从三大战役到百万雄师过大江席卷全国，然后是新中国成立，然后就是抗美援朝战争。</p><p>从传统的历史记载来看，在那段时间我们从一个胜利走向另一个胜利。</p><p>但是，你可能不知道的是，<b>那段历史表面上顺风顺水，其实是暗流涌动，危机四伏。这个危机不是来源于国内，而是国外。</b></p><p>准确的说，在1949年前后，全球两个头号强国美苏联手为中共以及后来的新中国挖了一系列大坑，稍有不慎，新中国就会面临国家分裂、国土沦丧的结局。</p><p>我们的历史书还是格局太低，视角总是局限于国内那一亩三分地，所以，大家了解的历史总有很大的局限性。现在我就站在全球的视野，给大家讲讲这段历史。</p><p class="ztext-empty-paragraph"><br></p><p><b><i>1 雅尔塔的枷锁</i></b></p><p>斯大林有两个身份，一个是苏联的国家领袖，另一个是国际共运的领袖。</p><p>在这两个身份中，斯大林真正认同的还是第一个身份，至于国际共运的领袖身份在很多时候都是为苏联国家利益服务的工具。</p><p>1945年2月，美国为了拉拢苏联对日宣战，在雅尔塔签订秘密协议，其中涉及中国主权部分包括：</p><p><b>外蒙古(蒙古人民共和国)</b>的现状须予维持；</p><p>维护苏联在<b>大连商港</b>的优先权益，并使该港国际化；</p><p>恢复<b>旅顺港口</b>苏俄海军基地的租借权；</p><p>中苏设立公司共同经营合办<b>中长铁路、南满铁路</b>，并保障苏联的优先利益；</p><p>同时维护中华民国在<b>满洲</b>完整的主权。</p><p>以上就是苏联对日宣战将拿到的红利。</p><p>这些红利是苏联梦寐以求心心念念上千年的目标！</p><p>简单的讲解一下这些红利对苏联意味着什么。</p><p>为啥要把外蒙古分裂出去？<b>因为外蒙古距离苏联远东的大动脉——西伯利亚铁路太近！</b></p><p>如果在外蒙古边境集结一只军队，理论上就具有随时掐断西伯利亚铁路的可能。苏联期望获得外蒙古除了保障西伯利亚铁路的战略安全，还有更险恶的心机。</p><p>外蒙古独立从地缘政治上看，是从新疆到东北的整个中国北方的中间地带撕开了一个大缺口，接踵而至的就是西进新疆和东进东三省，并对中国北京长期保持高压态势。</p><p>一旦中国东北或新疆出现不利于苏联的事态，苏方就会以最短的距离和最快的速度直插北京。20世纪60、70年代中苏关系紧张时期，苏联在中苏和中蒙边境屯兵近百万，就曾对中国北方安全形成重大压力。</p><p><b>苏联最有价值的目标是旅顺与大连港。</b>这是面向太平洋的出海口，是<b>苏俄梦寐以求上千年的不冻港！</b>苏联这个国家很有意思，疆域面积世界第一，偏偏就没有一个面对全球主要海贸路线的出海口，虽然从满清手里抢去了海参崴，但是海参崴维度太高，不是不冻港，商业价值并不大。</p><p>在俄罗斯的历史上，唯一曾经获得过出海口不冻港的就是旅顺与大连，后来日俄战争中战败，这个港口又被日本人抢走了。现在借着雅尔塔协议，苏联又拿回了旅顺与大连，这次是无论如何不会再放手了。</p><p>按：苏俄对出海口不冻港的渴望参见历史文章<b><a class=" wrap external" target="_blank" rel="nofollow noreferrer">《俄罗斯的千年港口梦》</a></b>，顺便说一句，是否具有面向主要海贸路线的出海口，是能否进入发达国家俱乐部的必要条件之一。</p><p>但是旅顺与大连在中国境内，与苏联并不接壤。怎么才能长期占有这两个宝贵的港口呢？</p><p>历史上俄国人在东北搞了一条中长铁路，这条铁路从苏联境内经哈尔滨、长春直达旅顺，这是占领旅顺与大连的生命线。所以，要拿下旅顺与大连，就必须确保对中长线的控制。</p><p>从上图可以看出，如果让苏联占领旅顺、大连港，再控制了中长线，我们东北的主权其实被切割两块，<b>雅尔塔协议所谓的“维护中华民国在满洲完整的主权”不过是一句空话。</b></p><p>旅大港最重要的意义还不仅仅是主权问题，失去旅大港对于中国最要命的就是渤海门户大开，外国侵略者可以长驱直入在大沽口登陆，直接威胁北京——历史上第二次鸦片战争中的英法联军，庚子事变中的八国联军都是从这个路线攻陷了北京。</p><p><b>外蒙古+旅大港+中长铁路就是苏联通过雅尔塔协议给中国脖子上套的的锁链，这条锁链不砸碎，中国不要说保持主权完整，连国家安全都完全没有保障。</b></p><p>即便如此，在抗战胜利前夕，国民党政府还是与苏俄签订《中苏友好同盟条约》，被迫对雅尔塔协议进行背书。虽然国民党政府在主权问题上作出了重大牺牲，但是，总算得到了苏联支持国民党政府接收东北的承诺。</p><p>为啥国民党政府不惜牺牲主权也要换取苏联支持国民党接收东北的承诺？因为东北太重要了！</p><p>当时东北工业实力冠绝全国，以钢产量为例，在抗战时期，除东北其他地区钢产量还不到10万吨，而东北满铁最高峰时期钢产量达到133万吨！可以这样讲，国共之争，谁拿到东北谁就有争天下的本钱。但是，国民党如愿拿到东北了吗？</p><p><b><i>2 出尔反尔的苏联</i></b></p><p>1945年10月12日，国民党东北行营主任兼政治委员会主任委员熊式辉、经济委员张嘉墩、外交特派员蒋经国一行飞抵长春，启动接收东北的实际工作。</p><p>17日，熊式辉等在和驻东北苏军总司令马林诺夫斯基的第二次会谈中，提出接收日本和伪满政府独营与满日合营之产业，但马氏居然称这些产业应视为<b>“苏军战利品”</b>，应由苏方处理。</p><p>这个要求给兴冲冲指望通过《中苏友好同盟条约》就可以顺利拿到东北的国民党政府兜头一桶冷水。与此同时，乘坐美军军舰的国民党军队准备在大连登陆，结果被大连苏军直接拒绝。</p><p>熊式辉等人在与苏方交涉中屡屡受挫——该死的俄国佬，为什么出尔反尔？很快，苏联人就揭示了这个谜底。</p><p>11月14日，苏军经济顾问斯拉特科夫斯基向张嘉墩提出，苏俄在东北的商业机构拟向中国政府立案，并以没收敌产作为苏方财产与中国合作经营，这是苏俄第一次向国民党政府提出东北经济合作问题。</p><p>20日，斯拉特科夫斯基向张嘉墩正式提出苏俄关于经济合作的设想：组织中苏合办之股份公司，经营 “满业”和“满电”的产业；股本双方各半，苏方以两会社日本资产的一半作为己方股本；中方担任总裁，苏方担任总经理。</p><p>斯氏在谈话中特意表示，“政治问题与经济问题须同时解决”，将其以经济合作交换苏俄支持国民政府接收东北的条件。</p><p>你看看，<b>苏联当时胃口有多大——拿到了旅大、控制了中长铁路还不够，还想彻底控制东北的经济命脉！</b></p><p>为了实现这个目的，苏联在东北翻手是云覆手为雨，以支持（限制）当时中共领导的东北民主联军在东北的发展作为压迫国民党政府屈服的筹码。</p><p>当与国民党谈判不顺利时，就放手让民主联军在东北发展；而一旦与国民党谈判进展顺利，对于中共这个小兄弟就换了一副嘴脸。</p><p>45年11月，中共刚刚进入沈阳，苏军就通知中共东北局，沈阳要移交给国民党政府，要东北局机关与军队立刻撤出沈阳——<b>“不走？就用坦克赶你们走！”</b>这就是“老大哥”在涉及国家利益时对共产主义小兄弟的态度！</p><p>最后国民党与苏联谈判失败，原因主要就是美国佬坚决反对。</p><p>马歇尔就不止一次对时任外交部长的王世杰表示，对苏俄经济合作要求“不必立予解决。”1946年2月11日，美国大使馆照会王世杰：中苏东北经济合作“将被认为违反门户开放之原则，明显歧视美国望获得参加满洲工业发展机会的人民，并可能对树立未来满洲贸易关系上，置美国商业利益于显著不利地位。”</p><p><b>因为与国民党交涉失败，最后苏联才改变了态度，开始支持中共在东北的发展。</b>但是苏联最看重的还是雅尔塔协议拿到的旅大、中长权益。</p><p>所以，斯大林在日本投降后还致电要求毛泽东去重庆谈判并不能打内战。毛泽东多年间对此事一直积愤在胸，曾指责斯大林在中国犯了“不许革命”这样的大错误。</p><p><b><i>3 解放战争拖不得</i></b></p><p>伟大领袖曾经有句名言：<b>抗日战争快不得，解放战争拖不得。</b></p><p>为啥解放战争拖不得？因为当时中国外部环境太差，说群狼环视并不夸张。</p><p>美国坚决反共自不待言，就算是共产主义的老大哥苏联对于中共也别有怀抱。二战之后，苏联在亚洲主要的战略诉求不是扶持中共夺取政权，而是维持雅尔塔体系，固化苏联通过雅尔塔协议拿到的红利。</p><p>所以，一个统一的共产党领导的中国并不符合苏联的利益，分裂混乱的中国才是苏联最想看到的。因为苏联的潜在阻扰，加上美国的干涉，让解放战争的前景充满了危险与不确定性，直到1948年6月才出现了转机。</p><p>这个转机就是<b>柏林危机事件</b>。</p><p>1948年6月18日，美、英、法三国公布了“关于改革植国货币制度的法令”；6月21日，正式在西占区实行货币改革，发行了“B”记德国马克。</p><p><b>这一行动成为第一次柏林危机爆发的导火线。</b></p><p>苏联得知该计划后，于6月19日提出抗议，占领军长官索洛科夫斯基发布《告德国民众书》，书中称英、美、法三国欲分解德国。6月22日，苏占区也实行货币改革，发行新的D记号马克，并于6月24日，全面切断西占区与柏林的水陆交通及货运，只保留从西德往柏林三条走廊通道。</p><p>与此同时，苏联驻扎在东部德国地区的三十多万军队摆出了强烈的战争姿态。</p><p>美、英等国对此反应强烈，立即向西部德国紧急调动兵力，准备全面迎战。一时间，整个欧洲再度陷入危机，似乎“第三次世界大战”一触即发。</p><p>这就是历史上震惊全世界的“柏林危机”。</p><p>最终，由于双方都不愿诉诸武力，经过谈判，达成了妥协。</p><p>1949年5月12日，苏联宣布解除对柏林的封锁，第一次柏林危机结束。</p><p>美苏因为柏林危机无暇东顾，毛泽东立刻敏锐感到了这是一个难得的窗口期，立即组织三大战役，对国民党政权进行军事总摊牌。</p><p>柏林危机1948年6月爆发，辽沈战役1948年9月打响。等到1949年5月柏林危机结束，美苏回头一看，不但三大战役已经结束，解放军连南京都拿下来了——国民党政府事实上已经总崩溃。这TM还怎么干涉？这就是解放战争拖不得！</p><p><b><i>4 渡江前后的中苏博弈</i></b></p><p>1949年1月三大战役胜利结束，国民党政权摇摇欲坠。在这个关键时刻，斯大林却致电毛泽东反对解放军渡江南下，希望中共与国民党划江而治。</p><p>斯大林这个建议的理由却是，<b>解放军渡江美国一定会干涉。</b></p><p>当时美国怎么干涉？在柏林美苏双方剑拔弩张，战争一触即发。美国佬除非是脑袋进了水才会在柏林危机没解决前去干涉一个东方大国的政权更替。</p><p>当毛泽东毅然决定渡江南下时，斯大林再次致电，希望毛泽东寻求美国的国际承认。</p><p>为啥斯大林开始不希望解放军渡江，后来又要求中共寻求美国佬的国际承认？因为三大战役胜利之后中共发表了一个公开声明——<b>新中国将废除国民党政府对外签订一切卖国条约。</b></p><p>注意，声明说的是<b>“一切”</b>。</p><p>这个一切不仅包括与美国签订的《中美友好通商航海条约》，也包括与苏联签订的《中苏友好同盟条约》。</p><p>而在1949年2月3日，美国国务卿迪安·艾奇逊就指示司徒雷登发布了一个公开声明：中国新政府继续继承现存的中外条约义务是美国予以承认的前提。</p><p>所以，如果当时新中国要寻求获得美国的承认，就必须继承那个《中美友好通商航海条约》，只要新中国承认继承《中美友好通商航海条约》，自然也只能继续继承《中苏友好同盟条约》。</p><p>这就是斯大林的算盘！<b>斯大林心心念念的还是要保住旅顺、大连以及中长铁路的权益。</b></p><p>1949年4月，解放军百万雄师渡过长江，并于当月解放南京。在这个国民党政权已经总崩溃的时刻，美苏两国做出了迥然不同的选择。</p><p>美国大使司徒雷登留在南京，向中共伸出橄榄枝；而苏联政府大使馆默不作声地收拾行李居然跟随即将崩溃的国民党政府去了广州。</p><p>司徒雷登留在南京是希望与中共做场交易——以承认新中国为条件获取中共对过去条约的继承。而苏联呢？只要国民党政府没有咽下最后一口气，就一定要死死抓住国民党政权，以确保《中苏友好同盟条约》的合法性。</p><p>最后美苏都没如愿，司徒雷登黯然离开中国，苏联大使馆最后看着国民党政府转进台湾，也只能灰溜溜回到苏联。1949年10月1日，中华人民共和国成立。</p><p>那么，到了现在，中共的外部危机解除了吗？不！更大的危机才刚刚开始。新中国脖子上那道雅尔塔锁链还没有砸碎。</p><p><b><i>5 与斯大林艰苦的较量</i></b></p><p>1949年12月6日，毛泽东动身访问苏联，这是伟大领袖第一次出国访问。因为他肩负着一个艰巨的使命——<b>从斯大林手里拿回我们的旅大港与中长铁路。</b></p><p>1949年12月16日，毛泽东刚到苏联第一天会谈就直奔主题，讨论新中国收回旅大港以及中长铁路的事宜，但斯大林突然抛出雅尔塔协定作为理由给顶了回去。斯大林向毛泽东解释说：1945年的那个与国民党政府签订的旧条约是根据苏、美、英三国缔结的《雅尔塔协定》签订的，而苏联正是通过《雅尔塔协定》才在远东得到了千岛群岛、南库页岛和中长铁路旅顺口以及蒙古这个战略屏障等。</p><p>如果改动经过美国和英国同意的中苏条约，“哪怕改动一款,都可能给美国和英国提出修改条约中的涉及千岛群岛、南库页岛等等条款的问题提供法律上的借口”。</p><p>因此，经过慎重考虑后，苏联才“决定暂不改动这项条约的任何条款”。<b>对于斯大林的强词夺理，毛泽东感到非常失望。</b>但很快他就要求与斯大林举行第二次会谈。</p><p>12月24日，第二次会谈如期举行。但这次斯大林对条约根本就不予理睬。毛泽东感到非常恼火，公开向苏方表示不满，并从此闭门不出，不参加任何活动。中苏首脑的会谈就陷入僵局。怎么办？新中国怎么才能拿回自己的旅大港与中长铁路？<b>关键时刻，美国佬送来一记神助攻。</b></p><p>1950年1月5日，杜鲁门总统发表声明称：“在1943年12月1日的《开罗宣言》中，美、英、中三国元首申明他们的目的是使日本窃取于中国的领土，如台湾，归还中国。过去四年来，美国和其他盟国也都承认中国对该岛行使主权。美国对台湾或中国其他领土从无进行掠夺的野心，也不准备以武装部队干预中国现在的局势。美国政府不准备采取任何足以把美国卷入中国内战的行为。”</p><p><b>对于美国总统拉拢中国的讲话，苏联的反应异乎寻常。</b>1月7日凌晨一时,苏联外交部长维辛斯基在莫斯科紧急约见正在苏联访问的中华人民共和国中央人民政府主席毛泽东。建议中国外交部发表一个给联合国安理会的声明，否认前国民党政府继续为安理会中国代表的合法地位。</p><p>中国发表声明后,如果国民党政府代表继续留在安理会，苏联就采取行动，将拒绝出席安理会。美国人怕中国没注意到或不满足，1950年1月12日，美国国务卿艾奇逊在全美新闻俱乐部发表题为《中国的危机》的演讲，除了指责苏联占据中国旅大、中长铁路外，公开称国民党不是在战场上被打倒的，而是被中国人民抛弃了。</p><p><b>赤裸裸要拉拢新中国。</b></p><p>而且除了前几天提到的台湾外，这次把朝鲜半岛也作为价码抛出来吸引中国，声称远东防御圈不包括朝鲜半岛和台湾。</p><p>苏联对此的反应更为激烈。当时斯大林要毛泽东发表一个官方声明反驳，苏联和蒙古也同时发表。据师哲的回忆，毛泽东问清楚了官方声明就是要外交部发表正式声明，却故意只让胡乔木以新闻署长的名义发表一个非官方的与记者谈话来应付。</p><p>声明见报后，斯大林与莫洛托夫都非常生气，把毛泽东找去责问，说这种私人性质的谈话“一文不值”。毛泽东却不予理睬，甚至要师哲收回为缓和僵持紧张气氛而请斯大林去住所做客的话，“不请他”。</p><p>此前毛泽东为表示不满，闭门谢客拒绝外出参观等活动，并向苏联方面前来探访的人提到正在跟美国的盟友英国接近。</p><p>西方舆论对中苏关系现状也议论纷纷，英国报纸甚至说毛被软禁了。<b>由于这一系列压力，斯大林最终被迫让步。</b></p><p>1950年1月26日，在莫斯科的中国代表团向苏联提交了中国方面有关大连、旅顺和中长铁路协定的方案。1月28日，经过艰难紧张的谈判，苏联方面回复中国代表团，基本上同意了中国的方案，但加上了一条：苏联有权自由利用中长铁路运兵和军用物资。</p><p>对此，中国要求对等地利用苏联的西伯利亚铁路从东北至新疆自由运兵和军用物资。这不是要把刚分裂出去的蒙古给包围了吗？</p><p>这彻底激怒了苏方，最后这一条也基本上去掉了。</p><p>那么，这个条约签订是否意味着斯大林真打算把旅大港与中长铁路归还给中国？不是。斯大林还另有算计。</p><p class="ztext-empty-paragraph"><br></p><p><b><i>6 斯大林的算盘</i></b></p><p>1月30日（也就是中苏基本达成协议的同时），斯大林通过驻北朝鲜大使斯蒂科夫给金日成发了一封密电。在这封密电里，斯大林通知金日成，苏联准备支持金日成统一朝鲜的行动。接到密电之后，金日成兴奋异常——因为在此之前，金日成多次向斯大林请求支持他去统一朝鲜，而斯大林过去都毫不犹豫的拒绝了金日成的请求。</p><p>那么，这次为什么斯大林的态度突然有了180度的变化呢？<b>这是斯大林最凶狠的算计！</b></p><p>斯大林利用金日成在中国边境点一把火，再把中国拉进与美国直接军事对抗的第一线。由于中国当时海空力量基本为0，面对强大的美国海空军事压力，中国为了防守渤海，护住大沽口这个要害，就只能邀请苏联海空军长驻旅大，就只能把中长线交给苏联。</p><p>这样，苏联就可以达到长期占据旅大港，控制中长线的目的。</p><p>斯大林算计中国的时候，美国政局也出现了不利于新中国的变化。1950年2月9日，美国威斯康星州参议员麦卡锡在俄亥俄县的共和党妇女俱乐部发表了题为“国务院里的共产党”的演讲，声称在他手中，有“一份205 人的名单”，“这些人全都是共产党和间谍网的成员”。“国务卿知道名单上这些人都是共产党员，但这些人至今仍在草拟和制定国务院的政策。”麦卡锡的演说有如晴天霹雳，令美国上下一片哗然。</p><p>此前声名狼藉的麦卡锡则一夜之间成为名震全国的政治明星。<b>麦卡锡的这个演讲改变了美国政治气候，从此反共产主义成为美国政坛的政治正确。</b></p><p>这个政治氛围也绑架了杜鲁门政府对外政策，让杜鲁门政府对华政策出现极大的转变——从拉拢新中国反苏变成极端反华。</p><p>1950年6月25日朝鲜战争爆发。1950年6月25日—7月7日，联合国在苏联代表缺席的情况下连续通过三个决议，不但将北朝鲜定义为“侵略者”，还授权美国组建联合国军采取一切必要的手段阻止北朝鲜的“侵略”行为。</p><p>1950年6月27日，杜鲁门宣布台湾未来地位未定，命令第七舰队进入台湾，阻止任何对台湾的进攻。斯大林的阴谋得逞了！新中国已经被斯大林的算计推到与美国军事对抗的第一线，南面台湾是美军第七舰队，北面朝鲜是气势汹汹的“联合国军”。</p><p>新中国怎么办？</p><p>1950年9月15日，美军在仁川登陆，然后合围了北朝鲜人民军主力，人民军全面溃败，朝鲜形势急剧恶化。</p><p>这个时候，毛泽东为了避免战争还在继续努力。美国拿下汉城之后，周总理通过各种渠道给美国传信，美军不能过三八线，否则中国就要出兵。</p><p>但是，如日中天的美国佬把新中国警告完全当做一个笑话。麦克阿瑟公开放话，要打到鸭绿江边过圣诞节！</p><p><b>现在毛泽东就面临一个艰难的选择——出不出兵？</b>出兵就是直接与美军对抗，当时从任何角度来看，新中国实力与美国都不是一个等级。不出兵？那么我们最重要的东北工业基地就在美军军事威胁之下，而且，旅大港、中长线就拿不回来了！</p><p>这个决策非常痛苦，让毛泽东连续2天2夜都没下床。新中国面临空前的危机，一着不慎，满盘皆输！该死的斯大林，与美国佬联手挖了一个大坑，正笑吟吟的等着新中国跳进去。怎么办？</p><p><b><i>7 上帝视角的时间线</i></b></p><p>现在，让我们站在上帝的视角回顾一下1950年的时间线，你就会明白，斯大林的深谋远虑的算计有多狠。</p><p>1949年12月16日毛泽东访问苏联，提出归还旅大与中长线的诉求，斯大林拒绝了，24日，毛泽东再次提出，斯大林不予理睬。从此毛泽东闭门不出以示抗议。</p><p>1950年1月5日，美国佬送来神助攻——杜鲁门发表声明拉拢新中国；12日艾奇逊再次开出更高的价码拉拢新中国。</p><p>1月13日，苏联驻联合国代表马立克突然提出驱逐国民党议案，被联大否决，马立克以此为理由退出联合国安理会。</p><p>1月28日，苏联在中美双重压力下勉强同意了归还旅大与中长线。</p><p>1月30日，斯大林给金日成发送密电，同意金日成统一朝鲜的要求，并给与朝鲜10个师的武器装备。</p><p>6月25日朝鲜战争爆发，当日联合国通过谴责北朝鲜的议案。</p><p>6月27日，联合国秘书长赖伊向苏联代表马立克通报了25日议案，并且邀请马立克重返联合国，但是却被马立克拒绝。</p><p>6月27日，在苏联缺席下，安理会通过美国提出的“紧急制裁案”，建议会员国向南朝鲜提供必要的援助以击退武装进攻。同日，因为有联大的授权，美国第七舰队进入台湾。</p><p>7月7日，在苏联缺席下，安理会通过组建联合国军去朝鲜的提案，要求会员国提供军队和其他援助，交由美国领导的统一司令部使用。</p><p>1950年8月1日，苏联代表团重返安理会，并担任轮值。</p><p>你看看，苏联掐着点等到美国佬在联大把所有的活都干完了，才回到联合国安理会。那么，<b>对于苏联在这个敏感时期缺席安理会——其实就是放弃否决权，</b>斯大林是如何解释的呢？</p><p>6月29日，莫斯科针对27日联合国决议作出了公开声明：“……莫斯科此时只能采取置身事外的立场，因为如果苏联代表返回安理会，必将陷入二难选择的困境——不使用否决权（或弃权）就意味着对朝鲜乃至社会主义阵营的背叛，使用否决权则等于承认在平壤背后站着莫斯科，从而导致与美国和世界舆论的直接对立……”</p><p>苏联所谓“两难”的理由是不是很奇葩？</p><p>“使用否决权则等于承认在平壤背后站着莫斯科”——尼玛，<b>全世界都知道没有苏联的支持，北朝鲜怎么敢挑起朝鲜战争？</b></p><p>在朝鲜战场上人民军装备的可是全套的苏式武器！包括前期打得南韩军队落花流水的就是苏联T—34型坦克！</p><p>1950年10月19日中国人民志愿军入朝参战，1951年2月1日，联合国通过决议，将新中国打成“侵略者”。这个决议，苏联同样没有使用否决权。<b>从以上时间线可以清晰的看出，将新中国拖入了朝鲜战争泥潭的正是社会主义老大哥苏联。</b></p><p><b><i>8 屌丝的逆袭</i></b></p><p>在新中国最危险的时刻，毛泽东毅然做出了最伟大的决策：生死看淡，不服就干！中国，参战！<b>“我们认为应当参战，必须参战，参战利益极大，不参战损害极大”——毛泽东</b>1950年10月19日当伟大的中国人民志愿军入朝参战之后，东亚乃至世界的格局为之一变！</p><p>满怀共产主义理想和庄严正义感的中国志愿军士兵，爆发出惊人的战斗力，<b>将一支装备简陋的轻步兵推到人类军事史的巅峰！</b></p><p>在这支轻步兵的冲击下，武装到牙齿的强大的美军居然被打得溃不成军，一泻千里。在云山重创了美军王牌第一骑兵师，在清川江畔打烂了美二师和二十五师，在风雪弥天的长津湖边把美海军陆战一师以及第七步兵师打得灰飞烟灭。</p><p>中国人民志愿军气贯长虹，威震敌胆，让全世界都目瞪口呆。</p><p>志愿军辉煌战绩打出了军威，打出了国威，打出了新中国的国际地位。虽然那时中国工业基础基本为0，但是志愿军的战绩让全世界都承认新中国已经是一个军事强国。</p><p><b>这个新兴军事强国对于当时的苏联具有重大的战略价值。</b>冷战两大阵营对抗，苏联其实压力很大。二战后的苏联半个国家被打成废墟，人口死亡2000万，经济总量还不到美国一半，拿什么去与美国长期对抗？但是冉冉升起的中国却让斯大林看到了机会。</p><p>那就是——<b>扶持中国，中苏抱团共同对抗美国！</b></p><p>所以，因为朝鲜战争，斯大林对中国的态度来了个180度大转弯，虽然这个弯子转得有点大，但是形势逼人——志愿军在朝鲜战场的胜利不仅引起了西方国家的震动，也在东欧一票社会主义国家获得了极高的威望。</p><p>那时中国代表只要出现在东欧国家的集会，基本就是满堂欢呼与掌声——<b>朝鲜战争不仅为中国打出了国威，也让苏联一票社会主义小伙伴建立了对抗西方世界的信心。</b></p><p>不仅如此，整个社会主义大家庭都知道，北朝鲜在苏联支持下挑起了战争但是差点被灭国，是中国志愿军把“联合国军”赶回了三八线，这就是维护了苏联这个老大的颜面。</p><p>在这样形势下，斯大林也不得不拿出最大的诚意。<b>大连港、中长线直接就提前归还中国。</b>旅顺有点特殊，因为麦克阿瑟叫嚣要将战争扩大到中国境内，所以中国主动邀请苏联海空军驻扎旅顺港，护住渤海这个要害。</p><p>只要中长线在我们手里，就不怕苏联未来不归还旅顺。</p><p>事实也是如此，<b>朝鲜停战协议签订后，苏联就归还了旅顺。</b>另外就是苏联<b>对中国的工业化建设的援助——这就是著名的156工程。</b></p><p>这个156工程我们出兵朝鲜之后就开始谈，一开始只有50个项目，随着志愿军节节胜利，苏联对华援助的项目也一路攀升，最后谈定是156个大型项目。</p><p>其实，按照当时的形势，只要我们提要求，苏联基本就是有求必应，只不过我们当时底子太差，拿到156个项目已经是中国接受能力的极限。</p><p>1949年新中国成立时，全国只有10万名工程师，合格的只有4万。产业工人也不够。当时基础最好的东北接受了156工程几十个项目，结果连产业工人都凑不齐，后来紧急从上海抽调了10万产业工人，才勉强凑够了人数。</p><p>产业工人可以凑，工程师想凑也凑不了。于是苏联又提供技术专家，高峰时期苏联专家在华人数达到5000人。相当于我们全国合格工程师总人数的12%。给人还给设备。156工程苏联提供的设备一律按照成本价算，还不用付钱，都是苏联贷款买单。除了给人给设备还给技术，156工程以重工业项目为主，一个项目涉及技术专利就是几十万，156个大型项目涉及专利技术就是几百万项——这些技术如果全用钱买，估计新中国全国人民当掉裤子也买不起。</p><p>现在因为抗美援朝战争的胜利，一律免费！如果没有朝鲜战争，新中国靠自身积累要拿下156工程至少也要10年以上。</p><p>也就是说，<b>因为抗美援朝战争的红利，让我们的工业化建设至少也提速了10年！</b></p><p>提速10年是个什么概念？如果我们把共和国70年历史看做是一部工业史——</p><p>对，<b>猫哥作为坚定的工业党永远是从工业化的角度去观察一个国家的历史</b>——</p><p>不要给我扯前三十年这运动那运动，站在工业党的角度，不管是前三十年还是后四十年，中国的工业化进程一直在前进！</p><p>如果没有这开挂般的10年提速，后来一切历史不变，我们今天人均GDP就不是1万美元而是5000美元，我们今天的整体生活水平会倒退10年！</p><p>现在让我们再一次学习伟大领袖1950年的表态——<b>“我们认为应当参战，必须参战，参战利益极大，不参战损害极大”！</b></p><p>真是暮鼓晨钟具有穿透历史的力量！</p><p>我们后人开了上帝的视角也只能对毛泽东感佩不已。因为伟大领袖的决策，因为我们志愿军战士浴血奋战，新中国不但打碎了雅尔塔枷锁，而且成为朝鲜战争最大的赢家，几乎是平白拿到了一个重工业体系。</p><p><b><i>9 伟大复兴</i></b></p><p>为什么老百姓把毛泽东比喻为东方红？</p><p><b>因为天不生润之，华夏万古如长夜！</b></p><p>为什么志愿军将士是最可爱的人？</p><p><b>因为没有他们卧冰嚼雪英勇牺牲，就没有今天的一切！</b></p><p>新中国的成立与发展从来都不是一帆风顺的，从1949到2019我们周围都是危机四伏群狼环视，苏联给我们套上一条雅尔塔枷锁，美国给我们套上一条第一岛链。但是中华民族岂能是这两条锁链所能束缚的？毛泽东领导志愿军击碎了雅尔塔枷锁；现在我们又打破了第一岛链的束缚；未来谁还能阻挡中华民族的伟大复兴？诸君，加油吧！</p><p><b><i>10 写在文章的最后</i></b></p><p>写完本文后，我突然找来《亮剑》主题曲播放。</p><p>当那激昂的音乐响起的时候，我的脑海是一幕幕画面：</p><p>多灾多难的近代中国……</p><p>雅尔塔协议，被肢解的中国主权……</p><p>叼着烟斗斯大林阴冷的眼神……</p><p>毛泽东在床上辗转难眠，黑夜里烟头忽明忽灭……</p><p>志愿军雄赳赳气昂昂跨过鸭绿江……</p><p>在云山，在清川江畔，在风雪弥天长津湖边，无数志愿军士兵前赴后继发起决死的冲锋……</p><p>在松骨峰，在龙源里，满身是火的志愿军士兵与美军在地上翻滚扭打……</p><p>溃退的美军长长的车队……</p><p>垂头丧气的美军俘虏高举着双手……</p><p>沸腾的新中国处处歌声嘹亮……</p><p>不知不觉之中，我已经是泪流满面。</p><p><b>————————————</b></p><p><b>本文来源于公众号“猫哥的视界”，深度文章会第一时间发表在该公众号上，欢迎大家订阅。</b></p><p>原文链接：</p><a target="_blank" data-draft-node="block" data-draft-type="link-card" data-image="https://pic2.zhimg.com/v2-9370874019c6824253b89044e768e5a4_180x120.jpg" data-image-width="896" data-image-height="378" class="LinkCard LinkCard--hasImage"><span class="LinkCard-backdrop" style="background-image:url(https://pic2.zhimg.com/v2-9370874019c6824253b89044e768e5a4_180x120.jpg)"></span><span class="LinkCard-content"><span class="LinkCard-text"><span class="LinkCard-title" data-text="true">原创 | 危机四伏：1949年前后发生了什么？</span><span class="LinkCard-meta"><span style="display:inline-flex;align-items:center">​
#         <svg class="Zi Zi--InsertLink" fill="currentColor" viewbox="0 0 24 24" width="17" height="17">
#          <path d="M13.414 4.222a4.5 4.5 0 1 1 6.364 6.364l-3.005 3.005a.5.5 0 0 1-.707 0l-.707-.707a.5.5 0 0 1 0-.707l3.005-3.005a2.5 2.5 0 1 0-3.536-3.536l-3.005 3.005a.5.5 0 0 1-.707 0l-.707-.707a.5.5 0 0 1 0-.707l3.005-3.005zm-6.187 6.187a.5.5 0 0 1 .638-.058l.07.058.706.707a.5.5 0 0 1 .058.638l-.058.07-3.005 3.004a2.5 2.5 0 0 0 3.405 3.658l.13-.122 3.006-3.005a.5.5 0 0 1 .638-.058l.069.058.707.707a.5.5 0 0 1 .058.638l-.058.069-3.005 3.005a4.5 4.5 0 0 1-6.524-6.196l.16-.168 3.005-3.005zm8.132-3.182a.25.25 0 0 1 .353 0l1.061 1.06a.25.25 0 0 1 0 .354l-8.132 8.132a.25.25 0 0 1-.353 0l-1.061-1.06a.25.25 0 0 1 0-.354l8.132-8.132z"></path>
#         </svg></span>mp.weixin.qq.com</span></span><span class="LinkCard-imageCell"><img class="LinkCard-image LinkCard-image--horizontal" alt="图标" src="https://pic2.zhimg.com/v2-9370874019c6824253b89044e768e5a4_180x120.jpg"></span></span></a><p>延伸阅读：</p><a target="_blank" data-draft-node="block" data-draft-type="link-card" data-image="https://pic2.zhimg.com/v2-80d72887b0063f40664c03f9a4d8c0ec_180x120.jpg" data-image-width="498" data-image-height="212" class="LinkCard LinkCard--hasImage"><span class="LinkCard-backdrop" style="background-image:url(https://pic2.zhimg.com/v2-80d72887b0063f40664c03f9a4d8c0ec_180x120.jpg)"></span><span class="LinkCard-content"><span class="LinkCard-text"><span class="LinkCard-title" data-text="true">原创 | 屌丝的逆袭：志愿军凭啥能赢得抗美援朝的胜利？</span><span class="LinkCard-meta"><span style="display:inline-flex;align-items:center">​
#         <svg class="Zi Zi--InsertLink" fill="currentColor" viewbox="0 0 24 24" width="17" height="17">
#          <path d="M13.414 4.222a4.5 4.5 0 1 1 6.364 6.364l-3.005 3.005a.5.5 0 0 1-.707 0l-.707-.707a.5.5 0 0 1 0-.707l3.005-3.005a2.5 2.5 0 1 0-3.536-3.536l-3.005 3.005a.5.5 0 0 1-.707 0l-.707-.707a.5.5 0 0 1 0-.707l3.005-3.005zm-6.187 6.187a.5.5 0 0 1 .638-.058l.07.058.706.707a.5.5 0 0 1 .058.638l-.058.07-3.005 3.004a2.5 2.5 0 0 0 3.405 3.658l.13-.122 3.006-3.005a.5.5 0 0 1 .638-.058l.069.058.707.707a.5.5 0 0 1 .058.638l-.058.069-3.005 3.005a4.5 4.5 0 0 1-6.524-6.196l.16-.168 3.005-3.005zm8.132-3.182a.25.25 0 0 1 .353 0l1.061 1.06a.25.25 0 0 1 0 .354l-8.132 8.132a.25.25 0 0 1-.353 0l-1.061-1.06a.25.25 0 0 1 0-.354l8.132-8.132z"></path>
#         </svg></span>mp.weixin.qq.com</span></span><span class="LinkCard-imageCell"><img class="LinkCard-image LinkCard-image--horizontal" alt="图标" src="https://pic2.zhimg.com/v2-80d72887b0063f40664c03f9a4d8c0ec_180x120.jpg"></span></span></a><a target="_blank" data-draft-node="block" data-draft-type="link-card" data-image="https://pic4.zhimg.com/v2-1f5ce21285d41a68265a86a51d3cb064_180x120.jpg" data-image-width="1080" data-image-height="606" class="LinkCard LinkCard--hasImage"><span class="LinkCard-backdrop" style="background-image:url(https://pic4.zhimg.com/v2-1f5ce21285d41a68265a86a51d3cb064_180x120.jpg)"></span><span class="LinkCard-content"><span class="LinkCard-text"><span class="LinkCard-title" data-text="true">原创：苦难的行军（上）</span><span class="LinkCard-meta"><span style="display:inline-flex;align-items:center">​
#         <svg class="Zi Zi--InsertLink" fill="currentColor" viewbox="0 0 24 24" width="17" height="17">
#          <path d="M13.414 4.222a4.5 4.5 0 1 1 6.364 6.364l-3.005 3.005a.5.5 0 0 1-.707 0l-.707-.707a.5.5 0 0 1 0-.707l3.005-3.005a2.5 2.5 0 1 0-3.536-3.536l-3.005 3.005a.5.5 0 0 1-.707 0l-.707-.707a.5.5 0 0 1 0-.707l3.005-3.005zm-6.187 6.187a.5.5 0 0 1 .638-.058l.07.058.706.707a.5.5 0 0 1 .058.638l-.058.07-3.005 3.004a2.5 2.5 0 0 0 3.405 3.658l.13-.122 3.006-3.005a.5.5 0 0 1 .638-.058l.069.058.707.707a.5.5 0 0 1 .058.638l-.058.069-3.005 3.005a4.5 4.5 0 0 1-6.524-6.196l.16-.168 3.005-3.005zm8.132-3.182a.25.25 0 0 1 .353 0l1.061 1.06a.25.25 0 0 1 0 .354l-8.132 8.132a.25.25 0 0 1-.353 0l-1.061-1.06a.25.25 0 0 1 0-.354l8.132-8.132z"></path>
#         </svg></span>mp.weixin.qq.com</span></span><span class="LinkCard-imageCell"><img class="LinkCard-image LinkCard-image--horizontal" alt="图标" src="https://pic4.zhimg.com/v2-1f5ce21285d41a68265a86a51d3cb064_180x120.jpg"></span></span></a><a target="_blank" data-draft-node="block" data-draft-type="link-card" class="LinkCard LinkCard--noImage"><span class="LinkCard-content"><span class="LinkCard-text"><span class="LinkCard-title" data-text="true">苦难的行军（下）</span><span class="LinkCard-meta"><span style="display:inline-flex;align-items:center">​
#         <svg class="Zi Zi--InsertLink" fill="currentColor" viewbox="0 0 24 24" width="17" height="17">
#          <path d="M13.414 4.222a4.5 4.5 0 1 1 6.364 6.364l-3.005 3.005a.5.5 0 0 1-.707 0l-.707-.707a.5.5 0 0 1 0-.707l3.005-3.005a2.5 2.5 0 1 0-3.536-3.536l-3.005 3.005a.5.5 0 0 1-.707 0l-.707-.707a.5.5 0 0 1 0-.707l3.005-3.005zm-6.187 6.187a.5.5 0 0 1 .638-.058l.07.058.706.707a.5.5 0 0 1 .058.638l-.058.07-3.005 3.004a2.5 2.5 0 0 0 3.405 3.658l.13-.122 3.006-3.005a.5.5 0 0 1 .638-.058l.069.058.707.707a.5.5 0 0 1 .058.638l-.058.069-3.005 3.005a4.5 4.5 0 0 1-6.524-6.196l.16-.168 3.005-3.005zm8.132-3.182a.25.25 0 0 1 .353 0l1.061 1.06a.25.25 0 0 1 0 .354l-8.132 8.132a.25.25 0 0 1-.353 0l-1.061-1.06a.25.25 0 0 1 0-.354l8.132-8.132z"></path>
#         </svg></span>mp.weixin.qq.com</span></span><span class="LinkCard-imageCell">
#       <div class="LinkCard-image LinkCard-image--default">
#        <svg class="Zi Zi--Browser" fill="currentColor" viewbox="0 0 24 24" width="32" height="32">
#         <path d="M11.991 3C7.023 3 3 7.032 3 12s4.023 9 8.991 9C16.968 21 21 16.968 21 12s-4.032-9-9.009-9zm6.237 5.4h-2.655a14.084 14.084 0 0 0-1.242-3.204A7.227 7.227 0 0 1 18.228 8.4zM12 4.836A12.678 12.678 0 0 1 13.719 8.4h-3.438A12.678 12.678 0 0 1 12 4.836zM5.034 13.8A7.418 7.418 0 0 1 4.8 12c0-.621.09-1.224.234-1.8h3.042A14.864 14.864 0 0 0 7.95 12c0 .612.054 1.206.126 1.8H5.034zm.738 1.8h2.655a14.084 14.084 0 0 0 1.242 3.204A7.188 7.188 0 0 1 5.772 15.6zm2.655-7.2H5.772a7.188 7.188 0 0 1 3.897-3.204c-.54.999-.954 2.079-1.242 3.204zM12 19.164a12.678 12.678 0 0 1-1.719-3.564h3.438A12.678 12.678 0 0 1 12 19.164zm2.106-5.364H9.894A13.242 13.242 0 0 1 9.75 12c0-.612.063-1.215.144-1.8h4.212c.081.585.144 1.188.144 1.8 0 .612-.063 1.206-.144 1.8zm.225 5.004c.54-.999.954-2.079 1.242-3.204h2.655a7.227 7.227 0 0 1-3.897 3.204zm1.593-5.004c.072-.594.126-1.188.126-1.8 0-.612-.054-1.206-.126-1.8h3.042c.144.576.234 1.179.234 1.8s-.09 1.224-.234 1.8h-3.042z"></path>
#        </svg>
#       </div></span></span></a><p></p></span>
#  </div>
# </div>"""
#
# content = pip_line.process_data(content,
#                                 generate_key_word=True,
#                                 generate_summary=True,
#                                 generate_summary_bert_vector=True)
#
# [print(x) for x in content]
#
# content = "！"
# content = pip_line.process_data(content,
#                                 generate_key_word=True,
#                                 generate_summary=True,
#                                 generate_summary_bert_vector=True)
#
# [print(x) for x in content]

content = """#每日生活报告#
   我在上海生活，是个互联网从业者
   8:30 起床并赖床10分钟
   8:40-9:00 洗漱穿衣出门坐地铁去公司（花费4块钱）
   9:45-12:00 到公司查看昨日工作数据，并进行相关工作的调整，开会商讨近期活动上线事宜
   12:00-13:30 午饭时间，今天吃的是牛肉面，不好吃。（30块钱）
   13：30-16:00 下午工作很多，这期间都在认真工作
   16:00-16:15 带薪拉屎
   16:15-18:00 继续工作
   18:00-19:00 在公司吃饭，吃完再加会班
   19:00-20:00 回家，路上买了点水果（20块钱）
   20:00-22:00 逗猫，吃水果，看剧
   22:00-22:30 洗澡，上床躺着
   22:30-0:30 逛淘宝，然后睡觉（花费200块钱）
   今日花费254块钱"""
content = pip_line.process_data(content,
                                generate_key_word=True,
                                generate_summary=True,
                                generate_summary_bert_vector=True)

[print(x) for x in content]


"""
data = [
        {'paragraph': '段落1',
         'key_word': ['word1', 'word2', ...],
         'summary': '摘要',
         'summary_bert_embedding': np.array(512,)
        },
        {'paragraph': '段落2',
         'key_word': ['word1', 'word2', ...],
         'summary': '摘要',
         'summary_bert_embedding': np.array(512,)
        }
]

"""