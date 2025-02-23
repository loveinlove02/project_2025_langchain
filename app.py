from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')


from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph 
from langgraph.graph import START, END

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

from langchain_teddynote.tools.tavily import TavilySearch
from langchain_teddynote.tools import GoogleNews

from langchain_teddynote.graphs import visualize_graph
from langchain_teddynote.messages import pretty_print_messages
from langchain_teddynote.messages import display_message_tree
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage

######### 1. 상태 정의 #########

# 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]
    dummy_data: Annotated[str, 'dummy']

######### 2. 도구 정의 및 바인딩 #########

@tool
def search_keyword(query: str) -> List[Dict[str, str]]:     # 키워드로 뉴스 검색하는 도구     
    """Look up news by keyword"""

    news_tool = GoogleNews()        
    return news_tool.search_by_keyword(query, k=2)


tools = [search_keyword]

######### 3. LLM을 도구와 결합 #########

llm = ChatOpenAI(
    api_key=key, 
    model_name='gpt-4o-mini',
    temperature=0.1
)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    answer = llm_with_tools.invoke(state['messages'])

    print('=================================================================================')
    print(f'chatbot() 실행\n')
    print(f"[1] state[messages]: \n{state['messages']}\n")
    # print(f'[2] chatbot answer: \n', answer , "\n")
    print(f'[2] chatbot answer: \n', answer.content)
    print(f'[3] answer.additional_kwargs: \n', answer.additional_kwargs)
    print('=================================================================================')
    
    return {
        'messages': [answer], 
        'dummy_data': '[chatbot] 호출, dummy_data'
    }

# 상태 그래프 생성
graph_builder = StateGraph(State)

# chatbot 노드 추가
graph_builder.add_node('chatbot', chatbot)

# 도구 노드 생성
tool_node = ToolNode(tools=[search_keyword])

# 도구 노드 추가
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

######### 6. 메모리 생성 #######

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()


graph = graph_builder.compile(checkpointer=memory)

######### 7. 그래프 컴파일 #######

graph = graph_builder.compile(checkpointer=memory)

question = '2024년 노벨 문학상 관련 뉴스를 알려주세요.'

input = State(
    dummy_data='테스트 문자열' ,
    messages=[('user', question)]
)

config = RunnableConfig(
    recursion_limit=10, 
    configurable={'thread_id': '1'},
    tags=['data-tag']
)


i = 1

for event in graph.stream(input=input, config=config):      # 모드에 따라 event가 딕셔너리 또는 리스트. 기본 모드는 evenv가 딕셔서리 이다.
    print(f'({i})')
    print('=== 반복문 바같 === 시작')
    for key, value in event.items():
        print('=== 반복문 === 시작')
        print(f'\n[노드 이름]: {key}\n')                    # key 에는 노드의 이름
        # print(f"\n해당 노드의 값[value]: \n{value['messages'][-1]}")            

        if value['messages'][-1] is not None:
            if isinstance(value['messages'][-1], HumanMessage):
                print('==================== HumanMessage ========================')
                print(f"해당 노드의 값 [value][messages][-1]: \n{value['messages'][-1]}")        # value에는 해당 노드의 값
                print('==================== END HumanMessage ====================')
                print() 
            elif isinstance(value['messages'][-1], AIMessage):
                print('==================== AIMessage ========================')
                print(f"해당 노드의 값 [value]: \n{value['messages'][-1]}")        # value에는 해당 노드의 값
                print('==================== END AIMessage ====================')     
                print()       
            elif isinstance(value['messages'][-1], ToolMessage):
                print('==================== ToolMessage ========================')
                print(f"해당 노드의 값 [value]: \n{value['messages'][-1]}")        # value에는 해당 노드의 값
                print(f'Tool Calls:')
                print(f"\t name: {value['messages'][-1].name}")
                print(f"\t tool_call_id: {value['messages'][-1].tool_call_id}")
                print('==================== END ToolMessage ====================')     
                print()       
        else:
            print(f"해당 노드의 값 [value]: 없음")
            print()

        print('=== 반복문 === 끝')
    print('=== 반복문 바같 === 끝\n\n')
    i=i+1


# for event in graph.stream(input=input, config=config):      # 모드에 따라 event가 딕셔너리 또는 리스트. 기본 모드는 evenv가 딕셔서리 이다.
#     for key, value in event.items():
#         print(f'\n[노드 이름]: {key}\n')                    # key 에는 노드의 이름

#         print(f'value: \n{value}')
#         if 'messages' in value:                             # value에는 해당 노드의 값
#             message = value['messages']
#             value['messages'][-1].pretty_print()