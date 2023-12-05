import json
import os
import streamlit as st
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from latent_task_module import load_model, extract_latent_task, p_to_prompt

from huggingface_hub.hf_api import HfFolder

hf_token = os.getenv("Huggingface_token")

HfFolder.save_token(hf_token)

def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

def first_talk_maker(n: int):
    match n:
        case 1:
            talkbun = "よろしくお願いします！1"
        case 2:
            talkbun = "よろしくお願いします！私は22歳の大学生です．"
        case 3:
            talkbun = "よろしくお願いします！"
        case 4:
            talkbun = "よろしくお願いします！私は22歳の大学生です．最近，友達と喧嘩してしまいました．"
        case 5:
            talkbun = "よろしくお願いします！"
        case 6:
            talkbun = "よろしくお願いします！"
        case 7:
            talkbun = "よろしくお願いします！"
        case 8:
            talkbun = "よろしくお願いします！"
        case 9:
            talkbun = "よろしくお願いします！"
    return talkbun

openai_api_key=os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(temperature=0.7, model_name="gpt-4-0613",openai_api_key = openai_api_key, request_timeout = 30)

# テンプレートを定義
template = """
あなたは対話履歴をもとにユーザーと雑談します。雑談は，以下のルールに必ず従ってください．

- 人間らしく,フレンドリーでかつ簡潔な返答を行ってください.回りくどい説明はせず，人間の雑談を想定してください.
- 付け加えて，あなたは人間であるという前提で雑談を行ってください.例えば，「私はAIですので〜」というような返答はしないでください．
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

    # aa

match st.session_state.exam_process:
    case 0:
        st.write("""
                    ## 実験について
                 
                    このwebサイトは，対話システムの評価を行うための実験サイトです．以下の指示に従って実験を行ってください．
                    
                    - 実験はPCでのみ行うことができます．ブラウザはGoogle Chromeを推奨します．
                    - 実験は通信環境が整った場所で行ってください．また，一度実験を始めたら，途中でブラウザを閉じないでください．閉じた場合，最初からやり直しになってしまう場合があります．
                    - 実験では，対話システムと雑談を行います．しかし，対話する際は相手を人間として扱ってください．  
                    - お互い1回の発言を行うことを1セットとし，5セットで1対話とします．  
                    - 実験では，対話を8回行います．対話の前に，短い指示文が提示されますので，それに従って対話を行ってください．  
                    - 対話内容は，あなたの実際の情報でも構いませんし，フィクションでも構いません．  

                    ## 情報の取り扱いについて  
                 
                    - 本実験内での対話は保存され，研究目的でのみ利用されます．  
                    - 本実験で得た情報は，個人を特定できないように加工された上で，研究成果として発表される可能性があります．  
                    - 本実験で得た情報は，研究目的以外に利用されることはなく，適切に管理されます．   
                    - 本実験で得た情報の保存期間は実験日〜2024年3月31日までとし，保存期間を過ぎた情報は廃棄します．  
                    """)
        

    case 1:
        st.info("雑談の中で，相手に自分の情報や経験，体験等を伝えてください．", icon="ℹ️")
        first_talk = "よろしくお願いします！"
        state["memory"].save_context({"input": ""},{"output": first_talk})
    case 2:
        st.info("相手のことを知ろうとしてください．", icon="ℹ️")
    case 3:
        st.info("自身の感情を相手に伝え，共感してもらおうとしてください．", icon="ℹ️")
    case 4:
        st.info("相手に共感してください．", icon="ℹ️")
    case 5:
        st.info("何かについての議論をしようとしてください．（〜ってどう思う？，〜とはなんでしょう？等）", icon="ℹ️")
    case 6:
        st.info("会話を終了させようとしてください．", icon="ℹ️")
    case 7:
        st.info("明確なタスク（例えば，飛行機の予約やレストランの入店時など）がある前提で，ロールプレイしてください．", icon="ℹ️")
    case 8:
        st.info("自由に対話してください．", icon="ℹ️")
    case 9:
        st.write("""実験は終了です．まず，以下の\"ログをダウンロード\"をクリックして実験ログをダウンロードしてください．
                 その後，アンケートに回答してください．""")
        
        st.session_state.memory_dict.append({"p":st.session_state.plist})
        # st.write(st.session_state.memory_dict)
        json_string = json.dumps(st.session_state.memory_dict, indent=4, ensure_ascii=False)

        dld = st.download_button(
    label="ログをダウンロード",
    file_name="history.json",
    mime="application/json",
    data=json_string,
)
        if dld:
            st.link_button("アンケートへ", "https://forms.gle/Xbk6vk7mxFmb8oBLA")



with st.spinner("Loading..."):
    if "latent_task_model" not in st.session_state.keys():
        st.session_state.latent_task_model, st.session_state.latent_task_tokenizer = load_model(hf_token)

if st.session_state.exam_process >= 1 and st.session_state.exam_process <= 8:

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
                # print("history:",state["memory"])
                st.write(ai_response)
        new_ai_message = {"role": "assistant", "content": ai_response}
        st.session_state.messages.append(new_ai_message)


    if len(st.session_state.messages) >= 10:
        if st.button("次へ"):
            st.session_state.memory_dict.append(state["memory"].dict())
            # print("appended!")

            del st.session_state.state
            st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")}

            first_talk = first_talk_maker(st.session_state.exam_process + 1)

            st.session_state.messages = [
            {"role": "assistant", "content": first_talk}
        ]
            st.session_state.state["memory"].save_context({"input": ""},{"output": first_talk})

            st.session_state.exam_process += 1
            raise st.rerun()
            
    # st.write(st.session_state.exam_process, len(st.session_state.messages))

if st.session_state.exam_process == 0:
    if st.button("次へ"):
        st.session_state.exam_process += 1
        raise st.rerun()