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
from langchain.chains.summarize import load_summarize_chain

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

    # PDFドキュメントのの読み込み
    loader = PyPDFLoader(file_path=tmp_file_path) 
    # テキストをページごとに分割 
    documents = loader.load_and_split(text_splitter)

    #langchain_documents = [d.to_langchain_format() for d in documents]

    # OpenAIの埋め込みモデルの初期化（テキストデータを埋め込みベクトルに変換するために使用）
    embeddings = OpenAIEmbeddings()
    # テキストデータと埋め込みモデルを使用して、FAISSを介して文書の埋め込みベクトルを生成
    vectors = FAISS.from_documents(documents, embeddings) # FAISS：Meta製の近似最近傍探索ライブラリ

    # ChatOpenAIと、生成された埋め込みベクトルを持つリトリーバを使用して、対話型リトリーバルチェーンを初期化。
    # ユーザーのクエリに対して適切な応答を生成するために使用
    chain_chat = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectors.as_retriever()
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
        st.session_state['generated'] = ["読み込んだPDFファイルの名前：" + uploaded_file.name]

    # ['past']:ユーザーが入力したメッセージを保存
    if 'past' not in st.session_state:
        st.session_state['past'] = []

     
    # 要約用のプロンプトテンプレート
    summarize_prompt_template="""以下の文章を簡潔に要約してください。:
    {text}
    要約:"""    

    # 
    simplify_prompt_prefix = "元の文章の難しい表現を、元の意味を損なわないように注意して子ども向けのかんたんな表現に変換してください。語尾は「です」「ます」で統一してください。"

    # 簡単な日本語変換を行った例文
    simplify_examples = [
        {
            "元の文章": "綿密な計画のもと、彼は革命を起こし、王朝の支配に終止符を打った。",
                    "かんたんな文章": "よく考えられた計画で、彼は国の政治や社会の仕組みを大きく変えました。そして、王様の家族がずっと支配していた時代が終わりました。"
        },
        {
            "元の文章": "彼は無類の読書家であり、その博識ぶりは同僚からも一目置かれる存在だった。",
                    "かんたんな文章": "彼はたくさんの本を読むのが大好きで、たくさんのことを知っています。友達も彼の知識を尊敬しています。"
        },
        {
            "元の文章": "彼女は劇団に所属し、舞台で熱演を繰り広げ、観客を魅了していた。",
                    "かんたんな文章": "彼女は劇のグループに入っていて、舞台でとても上手に演じて、見ている人たちを楽しませています。"
        },
        {

            "元の文章": "宇宙の膨張は、エドウィン・ハッブルによって観測された銀河の運動から発見されました。",
                    "かんたんな文章": "宇宙がどんどん広がっていることは、エドウィン・ハッブルさんがたくさんの星が集まった大きなものが動いていることを見つけることでわかりました。"
        }
    ]
        
    simplify_example_formatter_template = """
    元の文章: {元の文章}\n
    かんたんな文章: {かんたんな文章}
    """

    simplify_prompt_suffix = "元の文章: {input}\nかんたんな文章:"

    # 簡単な日本語への変換
    simpify_system_message = "あなたは文章を子ども向けのかんたんな日本語に変換するのに役立つアシスタントです。"


    class PdfSimplySummarizer():
        def __init__(self, llm :ChatOpenAI):
            self.llm = llm

        # 要約
        def _summarize(self, langchain_documents: list) -> str:
            # プロンプトテンプレートの作成
            summarize_template = PromptTemplate(
                template=summarize_prompt_template,
                input_variables=["text"] # 代入する変数の指定
                )

            # 要約モデルの作成
            chain_sum = load_summarize_chain(
                llm = self.llm,
                chain_type="map_reduce", # 処理の分散方法の指定
                map_prompt=summarize_template,
                combine_prompt=summarize_template
            )

            # 文章の要約
            summary = chain_sum.run(
                inputs=langchain_documents,
                return_only_outputs=True)
            return summary

        def _simplify(self, summary: str) -> str:
            simplify_example_prompt = PromptTemplate(
                input_variables=["元の文章", "かんたんな文章"],
                template=simplify_example_formatter_template,
            )

            # FewShotPromptTemplate：教師データの入力に加えて、教師データをどのようなフォーマットで学習させるかも指定可能
            simplify_few_shot_prompt_template = FewShotPromptTemplate(
                examples=simplify_examples,
                example_prompt=simplify_example_prompt,
                prefix=simplify_prompt_prefix,
                suffix=simplify_prompt_suffix,
                input_variables=["input"],
                example_separator="\n\n",
            )

            simplify_few_shot_prompt = simplify_few_shot_prompt_template.format(
                input=summary)

            # チャットモデルの呼び出し
            messages = [
                SystemMessage(content=simpify_system_message),
                HumanMessage(content=simplify_few_shot_prompt),
            ]
            result = self.llm(messages)

            return result.content

        def run(self, langchain_documents: str):
            summary = self._summarize(langchain_documents)
            result = self._simplify(summary)
            return result

    # 畳み込み可能な要約文の表示
    with st.expander(label= uploaded_file.name + "の要約", expanded=False): # expanded=False 初期で展開しない
        PSS = PdfSimplySummarizer(llm)
        output_sum = PSS.run(vectors)
        st.write(output_sum)  


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