from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph 
from langgraph.graph import START, END

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_teddynote.tools.tavily import TavilySearch

from langchain_openai import ChatOpenAI
from langchain_teddynote.graphs import visualize_graph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage

import json

memory = MemorySaver()

######### 1. 상태 정의 #########

# 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]

######### 2. 도구 정의 및 바인딩 #########

tool = TavilySearch(max_results=1)
tools = [tool]

######### 3. LLM을 도구와 결합 #########

llm = ChatOpenAI(
    api_key=key, 
    model_name='gpt-4o-mini',
    temperature=0.1
)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    print(f'=='*50)
    print('===== chatbot() 함수 시작 =====')
    print(f"chatbot() 으로 넘어온 메시지 :")
    print(state['messages'])
    print(f"메시지 개수 : {len(state['messages'])}")
    print()

    answer = llm_with_tools.invoke(state['messages'])

    print(f'[도구 사용 LLM 실행 결과 content]: {answer.content}')
    print(f'[도구 사용 LLM 실행 결과 answer]: {answer}')
    print(f'[도구 사용 LLM 실행 결과 additional_kwargs]: {answer.additional_kwargs}')

    print('===== chatbot() 함수  끝 =====')
    print(f'=='*50)
    print()

    return {'messages': [answer]}












graph_builder = StateGraph(State)

graph_builder.add_node('chatbot', chatbot)

tool_node = ToolNode(tools=[tool])

graph_builder.add_node('tools', tool_node)

graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition
)

######### 5. 엣지 추가 #########

# tools 에서 chatbot 으로
graph_builder.add_edge('tools', 'chatbot')

# START 에서 chatbot 으로
graph_builder.add_edge(START, 'chatbot')


# chatbot 에서 END 로
graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile(checkpointer=memory)


config = RunnableConfig(
    recursion_limit=10, 
    configurable={'thread_id': '1'}
)


question = ('대한민국 대구 동성로의 중앙떡볶이에 대해서 웹 검색을 해주세요.')

i = 1

for event in graph.stream({"messages": [("user", question)]}, config=config):
    print()
    print('===== 여기서 시작 =====')
    print(f'[event] 바깥 for 시작 {i}')
    print()

    for k, value in event.items():
        print(f'실행한 노드 이름: {k}')
        print()

        if isinstance(value['messages'][-1], HumanMessage):
            print('==================== HumanMessage ========================')
            print(f"[해당 노드 값] value : \n{value['messages'][-1]}")
            # print(f"[해당 노드 값] content: {value['messages'][-1].content}")
            # print(f"additional_kwargs: {value['messages'][-1].additional_kwargs}")
            print('==================== END HumanMessage ====================')
            print()
        elif isinstance(value['messages'][-1], AIMessage):
            print('==================== AIMessage ========================')
            # print(f"[해당 노드 값] value : \n{value['messages'][-1]}")
            print(f"[해당 노드 값] value content: {value['messages'][-1].content}")
            # print(f"addtional_kwargs: {value['messages'][-1].additional_kwargs}")

            if 'tool_calls' in value['messages'][-1].additional_kwargs:
                # print(f"additional_kwargs tool_calls: {value['messages'][-1].additional_kwargs['tool_calls']}")
                tool_calls = value['messages'][-1].additional_kwargs['tool_calls']

                for call in tool_calls:
                    if 'function' in call:
                        arguments = json.loads(call['function']['arguments'])
                        name = call['function']['name']

                        print(f"도구 이름 : {name}")
                        print(f"Arguments: {arguments}")
            else:
                print("additional_kwargs tool_calls: None")
            print('==================== END AIMessage ====================')    

        elif isinstance(value['messages'][-1], ToolMessage):
            print('==================== ToolMessage ========================')
            # print(f"[해당 노드 값] value : \n{value['messages'][-1]}")
            content = json.loads(value['messages'][-1].content)

            if content and isinstance(content, list) and len(content) > 0:
                print(f"[해당 노드 값] 제목: {content[0].get('title', 'No title')}")
                print(f"[해당 노드 값] URL: {content[0].get('url', 'No URL')}")
                print(f"[해당 노드 값] 내용: {content[0].get('content', 'No URL')}")
            else:
                print("No content or invalid content format in ToolMessage")

            print('==================== END ToolMessage ====================')   

        print()
        
    print('바깥 for 끝')
    i=i+1
    
    print('===== 여기서 끝 =====') 
    print()
print(f'전체 반복문 {i}번 실행')    