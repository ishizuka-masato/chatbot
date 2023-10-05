import os
import streamlit as st
from streamlit_chat import message
# テキストの生成、言語の翻訳、質問への回答に使用(OpenAI GPT-3 言語モデル)
from langchain.embeddings.openai import OpenAIEmbeddings
# ユーザーと会話できるチャットボットを作成するために使用(OpenAI GPT-3 言語モデル)
from langchain.chat_models import ChatOpenAI
# データベースから情報を取得し、ユーザーのクエリに応答するために使用するチャットボットの一種
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
# ConversationalRetrievalChainが特定のクエリに対してもっとも関連性の高いドキュメントを見つけるために使用
from langchain.vectorstores import FAISS
# 一時ファイルやディレクトリを作成するのに利用
import tempfile
# テキストを文字に分割
from langchain.text_splitter import RecursiveCharacterTextSplitter
# プロンプトをテンプレート化し、プログラミングによりプロンプトを生成するための機能」です。
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from llama_index import download_loader

# サイドバーの作成
user_api_key = st.sidebar.text_input( 
    label="OpenAI API key",
    placeholder="Paste your openAI API key",
    type="password")


os.environ['OPENAI_API_KEY'] = user_api_key

# モデルの指定
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)

llm = select_model()

# PDFのファイルアップロード枠の作成
uploaded_file = st.sidebar.file_uploader("upload", type="pdf")

# チャンクサイズの制限を下回るまで再帰的に分割（セパレータのないテキストも分割可能）
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000, # チャンクの文字数
        chunk_overlap  = 100, # チャンクオーバーラップの文字数
        length_function = len,    
)


if uploaded_file :
    #  一時ファイルを作成し、そのファイルにアップロードされたPDFデータを書き込み
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file: # delete=False:ファイルが一時ファイルとして削除されないようにする
        tmp_file.write(uploaded_file.getvalue()) # getvalue()：バッファの全内容を含む bytes を返す
        tmp_file_path = tmp_file.name # 作成したファイルのディレクトリを保管

    def _load_documents(file_path: str) -> list:
        CJKPDFReader = download_loader("CJKPDFReader")
        loader = CJKPDFReader(concat_pages=False)
        document = loader.load_data(file=file_path)
        langchain_documents = [d.to_langchain_format() for d in document]
        return langchain_documents

    documents = _load_documents(tmp_file_path)
    
    # OpenAIの埋め込みモデルの初期化（テキストデータを埋め込みベクトルに変換するために使用）
    embeddings = OpenAIEmbeddings()
    # テキストデータと埋め込みモデルを使用して、FAISSを介して文書の埋め込みベクトルを生成
    vectorstore = FAISS.from_documents(documents, embeddings) # FAISS：Meta製の近似最近傍探索ライブラリ

    # ChatOpenAIと、生成された埋め込みベクトルを持つリトリーバを使用して、対話型リトリーバルチェーンを初期化。
    # ユーザーのクエリに対して適切な応答を生成するために使用
    chain_chat = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True # 回答文の作成に関連した元テキスト群についても示すように指定
        )

    # 入力したクエリを元にChatOpenAIモデルからの応答を返す関数
    def conversational_chat(query):

        # session_state：セッション内での更新や再実行の前後で保持したい値を辞書形式で保存
        result = chain_chat({"question": query, "chat_history": st.session_state['history']})
        # チャット履歴はタプルのリスト。各タプルはクエリとそのクエリから生成されたレスポンスを含む。
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = [] # ['history']:セッションの初期化

    # ['generated']:モデルから生成された回答を保存
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["読み込んだPDFファイルの名前は「" + uploaded_file.name + "」です。"]

    # ['past']:ユーザーが入力したメッセージを保存
    if 'past' not in st.session_state:
        st.session_state['past'] = ['読み込んだファイルの名前を教えて！']

    if 'summary' not in st.session_state:
        st.session_state['summary'] = []


    query_sum = "読み込んだPDFファイルの内容を300字以内で要約してください。"
    
    # 畳み込み可能な要約文の表示
    with st.expander(label= uploaded_file.name + "の要約", expanded=True): # expanded=False 初期で展開しない
        summary = conversational_chat(query_sum)
        #st.session_state['summary'].append(HumanMessage(content=summarize_prompt_template))
        # st.session_state['summary'].append(summary)
        st.write(summary)

    # チャット履歴の表示
    response_container = st.container()
    # ユーザーの入力とChatOpenAIモデルからの応答を表示するために使用
    container = st.container()

    with container:
        with st.form(
            key='my_form', # 要素を識別するためのキー
            clear_on_submit=True # フォームが送信された後にフォームの内容をクリアする設定
            ):
            
            # テキスト入力フィールドを提供
            user_input = st.text_input("Input:", placeholder="PDFデータに関するメッセージをご入力ください。", key='input')
            submit_button = st.form_submit_button(label='送信')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # ユーザーとチャットボットのメッセージを表示
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

            