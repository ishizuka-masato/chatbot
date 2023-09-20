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


# サイドバーの作成
user_api_key = st.sidebar.text_input( 
    label="OpenAI API key",
    placeholder="Paste your openAI API key here",
    type="password")
# PDFのファイルアップロード枠の作成
uploaded_file = st.sidebar.file_uploader("upload", type="pdf")
# 環境変数によるOpenAI APIキー入力
os.environ['OPENAI_API_KEY'] = user_api_key

# チャンクサイズの制限を下回るまで再帰的に分割
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000, # チャンクの文字数
        chunk_overlap  = 100, # チャンクオーバーラップの文字数
        length_function = len,
    
)

if uploaded_file :
    #  一時ファイルを作成し、そのファイルにアップロードされたPDFデータを書き込み
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file: # delete=False:ファイルが一時ファイルとして削除されないようにする
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(file_path=tmp_file_path)  
    data = loader.load_and_split(text_splitter)

    # OpenAIの埋め込みモデルの初期化（テキストデータを埋め込みベクトルに変換するために使用）
    embeddings = OpenAIEmbeddings()
    # テキストデータと埋め込みモデルを使用して、Faissを介して文書の埋め込みベクトルを生成
    vectors = FAISS.from_documents(data, embeddings)

    # ChatOpenAIと、生成された埋め込みベクトルを持つリトリーバを使用して、対話型リトリーバルチェーンを初期化。
    # このチェーンは、ユーザーのクエリに対して適切な応答を生成するために使用される。
    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo-16k'),
                                                                      retriever=vectors.as_retriever())

    # 入力したクエリを元にChatOpenAIモデルからの応答を返す関数
    def conversational_chat(query):

        result = chain({"question": query, "chat_history": st.session_state['history']})
        # チャット履歴はタプルのリストであり、各タプルはクエリとそのクエリから生成されたレスポンスを含む。
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = [] # ['history']:セッションの初期化

    # ['generated']:モデルから生成された回答を保存
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Feel free to ask about anything regarding this" + uploaded_file.name]

    # ['past']:ユーザーが入力したメッセージを保存
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]
        
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
            user_input = st.text_input("Input:", placeholder="Please enter your message regarding the PDF data.", key='input')
            submit_button = st.form_submit_button(label='Send')
            
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