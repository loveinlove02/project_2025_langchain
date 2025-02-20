from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    api_key=key, 
    model_name='gpt-4o-mini',
    temperature=0.1
) 

def chatbot(state: State):
    answer = llm.invoke(state['messages'])

    print('==' * 50)
    print(f'chatbot() 실행\n')
    print(f"[1] state[messages]: \n{state['messages']}\n")
    # print(f'[2] chatbot answer: \n', answer , "\n")
    print(f'[2] chatbot answer: \n', answer.content)
    print('==' * 50)

    return {'messages': [answer]}


graph_builder = StateGraph(State)

graph_builder.add_node('chatbot', chatbot)      # 노드(함수) 이름을 인자로 받아서 chatbot 노드를 추가
graph_builder.add_edge(START, 'chatbot')        # 시작 노드에서 챗봇 노드(chatbot)로의 엣지 추가
graph_builder.add_edge('chatbot', END)          # chatbot 노드에서 END 노드로 엣지 추가


graph = graph_builder.compile()                 # 그래프 컴파일

question = '대한민국 계엄령에 대해서 알려줘'

for event in graph.stream({'messages': [('user', question)]}):
    print('==' * 50)
    print('[event]')

    for k, value in event.items():
        print(f'노드 이름: {k}')
        print()
        print(f"assistant 답변: ")
        print(f"{value['messages'][-1].content}")

    print('==' * 50)