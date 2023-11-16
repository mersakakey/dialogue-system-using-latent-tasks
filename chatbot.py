import json
import streamlit as st
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_to_dict
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import messages_from_dict, messages_to_dict
from langchain import PromptTemplate

from latent_task_module import load_model, extract_latent_task, p_to_prompt

def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

openai_api_key=st.secrets.OpenAIApiKey.key

chat = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0613",openai_api_key = openai_api_key, request_timeout = 30)

# テンプレートを定義
template = """
あなたは対話履歴をもとにユーザーと雑談します。雑談は，以下のルールに必ず従ってください．

- 人間らしく,フレンドリーでかつ簡潔な返答を行ってください.回りくどい説明はせず，人間の雑談を想定してください.
- 簡潔であることは重要です.多くとも2文,できれば1文で返答してください.
- ただし,出力に"AI:"や"answer:"等の文字を付与する必要はありません.返答文のみを出力してください.

history: {chat_history}  
question: {input}
"""

# テンプレートを用いてプロンプトを作成
prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template=template
)


conversation = ConversationChain(
        llm=chat, 
        prompt=prompt,
        memory=state['memory']            
    )


st.set_page_config(
    page_title="対話実験サイト",
)

if "exam_process" not in st.session_state.keys():
    st.session_state.exam_process = 0

if "memory_dict" not in st.session_state.keys():
    st.session_state.memory_dict = []

if "plist" not in st.session_state.keys():
    st.session_state.plist = []

match st.session_state.exam_process:
    case 0:
        st.write("""
                    ## 実験について
                    このwebサイトは，対話システムの評価を行うための実験サイトです．以下の指示に従って実験を行ってください．
                    
                    ・実験は通信環境が整った場所で行ってください．また，一度実験を始めたら，途中でブラウザを閉じないでください．閉じた場合，最初からやり直しになってしまう場合があります．
                    ・実験では，対話システムと雑談を行います．  
                    ・お互い1回の発言を行うことを1セットとし，5セットで1対話とします．  
                    ・実験では，対話を8回行います．対話の前に，短い指示文が提示されますので，それに従って対話を行ってください．  
                    ・対話内容は，あなたの実際の情報でも構いませんし，フィクションでも構いません．  

                    ## 情報の取り扱いについて  
                    ・本実験内での対話は保存され，研究目的でのみ利用されます．  
                    ・本実験で得た情報は，個人を特定できないように加工された上で，研究成果として発表される可能性があります．  
                    ・本実験で得た情報は，研究目的以外に利用されることはなく，適切に管理されます．   
                    ・本実験で得た情報の保存期間は実験日〜2024年3月31日までとし，保存期間を過ぎた情報は廃棄します．  
                    """)
        

    case 1:
        st.write("雑談の中で，相手に自分の情報や経験，体験等を伝えてください．")
    case 2:
        st.write("相手のことを知る")
    case 3:
        st.write("共感を得る")
    case 4:
        st.write("相手に共感する")
    case 5:
        st.write("議論")
    case 6:
        st.write("会話終了")  
    case 7:
        st.write("明確なタスクがある")
    case 8:
        st.write("自由対話")
    case 9:
        st.write("""実験は終了です．まず，以下の\"ログをダウンロード\"をクリックして実験ログをダウンロードしてください．
                 その後，アンケートに回答してください．""")
        
        st.session_state.memory_dict.append({"p":st.session_state.plist})
        # st.write(st.session_state.memory_dict)
        json_string = json.dumps(st.session_state.memory_dict, indent=4, ensure_ascii=False)

        st.download_button(
    label="ログをダウンロード",
    file_name="history.json",
    mime="application/json",
    data=json_string,
)
        st.link_button("アンケートへ", "https://store.steampowered.com/app/105600/Terraria/?l=japanese")



with st.spinner("Loading..."):
    if "latent_task_model" not in st.session_state.keys():
        st.session_state.latent_task_model, st.session_state.latent_task_tokenizer = load_model()

if st.session_state.exam_process >= 1 and st.session_state.exam_process <= 8:

    first_talk = "お話ししよう！"

    # check for messages in session and create if not exists
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": first_talk}
        ]


    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if len(st.session_state.messages) < 10:
        user_input = st.chat_input()
    else:
        user_input = None
    

    if user_input is not None:
        p = extract_latent_task(user_input, st.session_state.latent_task_model, st.session_state.latent_task_tokenizer)

        latent_task_categories, latent_task_prompt = p_to_prompt(p)

        emb_prompt = f"""

        latent_task_categories: {latent_task_categories}
        prompt: {latent_task_prompt}
        """

        converted_list_of_p = p.tolist()

        user_prompt = user_input + emb_prompt
        st.session_state.plist.append({"p":converted_list_of_p, "user_prompt":user_prompt, "latent_task_categories":latent_task_categories})

    else:
        user_prompt = None

    if user_prompt is not None:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)


    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                ai_response = conversation.predict(input=user_prompt)
                print("history:",state["memory"])
                st.write(ai_response)
        new_ai_message = {"role": "assistant", "content": ai_response}
        st.session_state.messages.append(new_ai_message)


    if len(st.session_state.messages) >= 10:
        if st.button("次へ"):
            st.session_state.memory_dict.append(state["memory"].dict())
            print("appended!")

            del st.session_state.state
            st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")}

            st.session_state.messages = [
            {"role": "assistant", "content": first_talk}
        ]

            st.session_state.exam_process += 1
            raise st.rerun()
            
    st.write(st.session_state.exam_process, len(st.session_state.messages))

if st.session_state.exam_process == 0:
    if st.button("次へ"):
        st.session_state.exam_process += 1
        raise st.rerun()