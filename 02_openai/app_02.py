from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
claude_key = os.getenv('CLAUDE_API_KEY')


# ChatAnthropic 객체를 생성합니다.
llm = ChatAnthropic(api_key=claude_key, model_name="claude-3-5-sonnet-20241022")

answer = llm.stream('삼성전자에서 만든 제품을 3가지 알려주세요.')
answer2 = ''

# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
for token in answer:
    print(token.content, end='', flush=True)
    answer2 += token.content