import streamlit as st
from langchain_openai import ChatOpenAI

from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool

from langchain_teddynote.messages import AgentStreamParser
from langchain_teddynote.messages import AgentCallbacks

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os


load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

st.title("CSV 데이터 분석 Agent 챗봇")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

class MessageRole:
    """
    메시지 역할을 정의하는 클래스입니다.
    """

    USER = 'user'                           # 사용자 메시지 역할
    ASSISTANT = 'assistant'                 # 어시스턴트 메시지 역할
# =============================================================================

class MessageType:
    """
    메시지 유형을 정의하는 클래스입니다.
    """

    TEXT = 'text'                           # 텍스트 메시지
    FIGURE = 'figure'                       # 이미지 메시지
    CODE = 'code'                           # 코드 메시지
    DATAFRAME = 'dataframe'                 # 데이터프레임 메시지
# =============================================================================

# =============================================================================
def print_messages():
    """
    저장된 메시지를 화면에 출력하는 함수입니다.
    """

    for role, content_list in st.session_state['messages']:
        with st.chat_message(role):                                     # role 에 따라 user, assistant를 출력
            for content in content_list:                                # content_list는 리스트. MessageType, 메시지
                if isinstance(content, list):
                    message_type, message_content = content             # 메시지 타입, 메시지 내용을 얻는다

                    if message_type == MessageType.TEXT:                # 메시지 타입이 텍스트 메시지
                        st.markdown(message_content)                    # 텍스트 메시지 출력
                    elif message_type == MessageType.FIGURE:            # 메시지 타입이 이미지 메시지
                        st.pyplot(message_content)                      # 이미지 메시지 출력
                    elif message_type == MessageType.CODE:              # 메시지 타입이 코드 메시지
                        with st.status('코드 출력', expanded=False):
                            st.code(message_content, language='python') # 코드 메시지 출력
                    elif message_type == MessageType.DATAFRAME:         # 메시지 타입이 데이터프레임 메시지
                        st.dataframe(message_content)                   # 데이터프레임 메시지 출력
                else:
                    raise ValueError(f'알 수 없는 컨텐츠 유형입니다: {content}')
# =============================================================================

# =============================================================================
def add_message(role: MessageRole, content: List[Union[MessageType, str]]): 

    """
    새로운 메시지를 저장하는 함수입니다.

    Agrs:
        role (MessageRole): 메시지 역할 (사용자 or 어시스턴트)
        content (List[UnionType, str]): 메시지 내용
    """

    # role은 MessageRole 타입의 객체를 매개변수로 받는다
    # content는 List를 매개변수로 받는다. 리스트의 요소는 MessageType 객체 이거나 str(메시지 내용)이다. 
    
    # ====================================================================================================
    # 호출 : assistant_message = [MessageType.TEXT, "무엇을 도와드릴까요?"]   
    #        add_message(MessageRole.ASSISTANT, assistant_message)
    # 결과 : messages = [[MessageRole.ASSISTANT, [[MessageType.TEXT, "무엇을 도와드릴까요?"]]]]
    #   
    # 호출 : user_message = [MessageType.TEXT, "안녕하세요"] 
    #        add_message(MessageRole.USER, user_message)
    # 결과 : messages = [ [MessageRole.ASSISTANT, [[MessageType.TEXT, "무엇을 도와드릴까요?"]]],
    #                     [MessageRole.USER, [[MessageType.TEXT, "안녕하세요"]]]
    #                   ]
    # ====================================================================================================

    messages = st.session_state['messages']

    if messages and messages[-1][0] == role:
        messages[-1][1].extend([content])           # 같은 역할의 연속된 메시지는 하나로 합친다.
    else:
        messages.append([role, [content]])          # 새로운 역할의 메시지는 새로 추가한다.
# =============================================================================    

with st.sidebar:
    clear_button = st.button('대화 다시')
    uploaded_file = st.file_uploader('CSV 파일을 업로드 해주세요.', type=['csv'], accept_multiple_files=True) 
    selected_model = st.selectbox('OpenAI 모델을 선택해주세요.', ['gpt-4o-mini', 'gpt-4o'], index=0)
    apply_button = st.button('데이터 분석 시작')

# ============================================================================= 
def tool_callback(tool) -> None:
    """
    도구 실행 결과를 처리하는 콜백 함수입니다.

    Args:
        tool (dict) : 실행된 도구 정보
    """

    if tool_name := tool.get('tool'):                   # tool.get('tool') 에는 사용된 도구 이름이 있다.
        if tool_name == 'python_repl_ast':              # 사용 도구의 이름이 python_repl_ast 이면
            tool_input = tool.get('tool_input', {})     # 입력 쿼리 (딕셔너리 타입)
            query = tool_input.get('query') 

            if query:
                df_in_result = None

                with st.status('데이터 분석 중...', expanded=True) as status:
                    st.markdown(f"```python\n{query}\n```")

                    add_message(MessageRole.ASSISTANT, [MessageType.CODE, query])

                    if 'df' in st.session_state:
                        result = st.session_state['python_tool'].invoke({'query': query})

                        if isinstance(result, pd.DataFrame):
                            df_in_result = result
                    
                    status.update(label='코드 출력', state='complete', expanded=False)


                if df_in_result is not None:
                    st.dataframe(df_in_result)
                    add_message(MessageRole.ASSISTANT, [MessageType.DATAFRAME, df_in_result])
                
                if 'plt.show' in query:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    add_message(MessageRole.ASSISTANT, [MessageType.FIGURE, fig])

                
                return result
            else:
                st.error('데이터프레임이 정의되지 않았습니다. csv 파일을 먼저 업로드 해주세요.')                                              
# ============================================================================= 

# ============================================================================= 
def observation_callback(observation) -> None:
    """
    관찰 결과를 처리하는 콜백 함수입니다.

    Args:
        observation (dict) : 관찰 결과
    """

    if 'observation' in observation:
        obs = observation['observation']

        if isinstance(obs, str) and 'Error' in obs:
            st.error(obs)
            st.session_state['messages'][-1][1].clear()     # 에러 발생 시 마지막 메시지 삭제
            

# ============================================================================= 
def result_callback(result: str) -> None:
    """
    최종 결과를 처리하는 콜백 함수입니다.

    Args:
        result (str) : 최종 결과
    """
    
    pass

# ============================================================================= 

# ============================================================================= 
def create_agent(dataframe, selected_model='gpt-4o-mini'):           # agent 생성 함수
    """
    데이터프레임 에이전트를 생성하는 함수입니다.

    Agrs:
        dataframe (pd.DataFrame) : 분석할 데이터프레임
        selected_model (str, optional) : 데이터분석에 사용하는 OpeanAI 모델
    
    Returns:
        Agent: 생성된 데이터프레임 에이전트
    """

    llm = ChatOpenAI(
        api_key=key, 
        model=selected_model, 
        temperature=0
    )

    pandas_dataframe_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        verbose=False, 
        agent_type='tool-calling',
        allow_dangerous_code=True,  
        
        # 데이터 분석에 필요한 프롬프트를 사전에 가지고 있는데 거기에 추가적으로 요구 사항을 적어준다 
        prefix="You are a professional data analyst and expert in Pandas. "
        "You must use Pandas DataFrame(`df`) to answer user's request. "
        "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
        "If you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
        "I prefer seaborn code for visualization, but you can use matplotlib as well."
        "\n\n<Visualization Preference>\n"
        "- [IMPORTANT] Use `English` for your visualization title and labels."
        "- `muted` cmap, white background, and no grid for your visualization."
        "\nRecommend to set cmap, palette parameter for seaborn plot if it is applicable. "
        "The language of final answer should be written in Korean. "
        "\n\n###\n\n<Column Guidelines>\n"
        "If user asks with columns that are not listed in `df.columns`, you may refer to the most similar columns listed below.\n",
    ) 
    
    return pandas_dataframe_agent


def ask(query):                                                 # 질문 처리 함수
    """
    사용자 질문을 처리하고 응답을 생성하는 함수입니다.

    
    Args:
        query (str) : 사용자 질문
    """

    if 'agent' in st.session_state:                                 # agent가 생성되어 있으면
        st.chat_message('user').write(query)                        # user 이모티콘, 사용자 질문
        add_message(MessageRole.USER, [MessageType.TEXT, query])        

        agent = st.session_state['agent']                           # agent를 가져온다
        response = agent.stream({'input': query})                   

        ai_answer = ''

        parser_callback = AgentCallbacks(tool_callback, observation_callback, result_callback)
        stream_parser = AgentStreamParser(parser_callback)

        with st.chat_message('assistant'):
            for step in response:
                stream_parser.process_agent_steps(step)

                if 'output' in step:
                    ai_answer = ai_answer + step['output']
            
            st.write(ai_answer)
        

        add_message(MessageRole.ASSISTANT, [MessageType.TEXT, ai_answer])


if clear_button:
    st.session_state['messages'] = []

if apply_button and uploaded_file:
    loaded_data = []

    for file in uploaded_file:
        df = pd.read_csv(file, encoding='utf-8')                # csv 파일 로드     
        # loaded_data.append(df)
    
    st.session_state['df'] = df                        # 데이터프레임 저장

    st.session_state['python_tool'] = PythonAstREPLTool()       # python 실행 도구 생성
    st.session_state['python_tool'].locals['df'] = df  # 데이터프레임을 python 실행 환결에 추가
    st.session_state['agent'] = create_agent(df, selected_model)   # agent 생성
    st.success('설정이 완료되었습니다. 대화를 시작해 주세요.')

elif apply_button:
    st.warning('파일을 업로드 해주세요.')


print_messages()        # 저장된 메시지 출력


user_input = st.chat_input('궁금한 내용을 물어보세요')

if user_input:
    ask(user_input)