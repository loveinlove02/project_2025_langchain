from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=key, 
    model_name='gpt-4o-mini',	# 모델 : 사용할 모델을 지정합니다.
    temperature=0.1,		    # 창의성 : (0.0 ~ 2.0) 
    max_tokens=300,	            # 최대 토큰 수
)

# answer에 스트리밍 답변의 결과를 받습니다.
answer = llm.stream('삼성전자에서 만든 제품을 3가지 알려주세요.')
answer2 = ''

# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
for token in answer:
    print(token.content, end='', flush=True)
    answer2 += token.content