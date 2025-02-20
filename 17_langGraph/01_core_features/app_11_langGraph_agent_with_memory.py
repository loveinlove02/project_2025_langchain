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
    answer = llm_with_tools.invoke(state['messages'])

    print('=================================================================================')
    print(f'chatbot() 실행\n')
    print(f"[1] state[messages]: \n{state['messages']}\n")
    # print(f'[2] chatbot answer: \n', answer , "\n")
    print(f'[2] chatbot answer: \n', answer.content)
    print(f'[3] answer.additional_kwargs: \n', answer.additional_kwargs)
    print('=================================================================================')

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


question = ('`소프트웨어놀이터`에서 코딩강의를 하고 있는 이인환입니다.')

for event in graph.stream({"messages": [("user", question)]}, config=config):
    
    print('[event] 실행 결과')

    for value in event.values():
        print('[value]: ')
        print(value['messages'][-1])
        
        print(f"\t[content]: {value['messages'][-1].content}")

        print(f"\t[additional_kwargs]: ")
        print(f"\t\t{value['messages'][-1].additional_kwargs}")

        print('\t\t[token_usage]: ')
        # print(f"\t\t\t {value['messages'][-1].response_metadata['token_usage']}")
        print(f"\t\t\t[completion_tokens]: {value['messages'][-1].response_metadata['token_usage']['completion_tokens']}")
        print(f"\t\t\t[prompt_tokens]: {value['messages'][-1].response_metadata['token_usage']['prompt_tokens']}")
        print(f"\t\t\t[total_tokens]: {value['messages'][-1].response_metadata['token_usage']['total_tokens']}")
        print(f"\t\t\t[completion_tokens_details]: {value['messages'][-1].response_metadata['token_usage']['completion_tokens_details']}")
        print(f"\t\t\t[prompt_tokens_details]: {value['messages'][-1].response_metadata['token_usage']['prompt_tokens_details']}")

        print(f"\t\t[model_name]: {value['messages'][-1].response_metadata['model_name']}")
        print(f"\t\t[system_fingerprint]: {value['messages'][-1].response_metadata['system_fingerprint']}")
        print(f"\t\t[finish_reason]: {value['messages'][-1].response_metadata['finish_reason']}")        
        
        print('\t[id]: ')
        print(f"\t\t {value['messages'][-1].id}")

        
        print('\t[usage_metadata]: ')
        print(f"\t\t[input_tokens]: {value['messages'][-1].usage_metadata['input_tokens']}")
        print(f"\t\t[output_tokens]: {value['messages'][-1].usage_metadata['output_tokens']}")
        print(f"\t\t[total_tokens]: {value['messages'][-1].usage_metadata['total_tokens']}")
        print(f"\t\t[input_token_details]: {value['messages'][-1].usage_metadata['input_token_details']}")
        print(f"\t\t[output_token_details]: {value['messages'][-1].usage_metadata['output_token_details']}")

