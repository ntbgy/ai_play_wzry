"""
pip install --upgrade spark_ai_python
"""
from sparkai.core.messages import ChatMessage
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler

# 星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
# 星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = 'a04f84c1'
SPARKAI_API_SECRET = 'Nzk3YjUzZDA0NDUwNzI0NjY0M2IwNDJh'
SPARKAI_API_KEY = '41776999999388c89ed0b05252179233'
# 星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'generalv3.5'


def get_answer(question):
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    content = f"""
注意事项：
直接回答选择题答案，不需要答案解析，不需要其他字符。
参考资料：
王者荣耀官方网站及游戏内的英雄介绍页面。
问题：
{question}
        """.strip()
    messages = [ChatMessage(
        role="user",
        content=content
    )]
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    answer = a.generations[0][0].text
    print(content)
    print()
    print(answer)
    return answer


"""
注意事项：
直接回答选择题答案，包括选项内容，不需要答案解析，不需要其他字符。
参考资料：
王者荣耀官方网站及游戏内的英雄介绍页面。
问题：
干将莫邪每次击败英雄或获得助攻时，都会为自己增加哪种属性？
A.攻击速度
B.生命上限
C.法术强度
D.物理攻击
"""
