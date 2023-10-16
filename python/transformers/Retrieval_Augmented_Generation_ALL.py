# -*- mode: python; -*-

'''
The MIT License (MIT)
Copyright (C) 2023 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

'''
env
iphthon
%run Retrieval_Augmented_Generation_ALL.py
'''

################################################################################################
################################################################################################
# (出典) Retrieval Augmented Generation(RAG)アーキテクチャをHuggingFaceのモデルで作ってみよう！ ～Retrieval編
# https://techblog.cccmk.co.jp/entry/2023/08/08/120452
################################################################################################
################################################################################################


################################################################
# RAG用LLMチューニングデータ生成手順
################################################################

################################
# get_text_content - ウェブページから本文テキストを取得
################################

import urllib

from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader

textfile = "./temp.txt"

def get_text_content(url, class_name):
    """
    urlで示されるページを取得し、class名=class_nameであるdiv要素のテキスト情報を返す
    """
    # コンテント取得
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as res:
        html = res.read().decode()
    # print(f'html={html}')

    # BeautifulSoupで解析し、本文テキストを取得
    content = BeautifulSoup(html).find("div", attrs ={"class":class_name}).text

    # テキストファイルとして出力
    with open(textfile, "w") as f:
        f.write(content) 

    # テキストファイルから入力
    loader = TextLoader(textfile)
    data = loader.load()

    return {"url":url, "class_name":class_name, "content":data}

################################
# テキストを分割し、ベクトル化し、Vectorstoreに格納する
################################

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# テキスト分割のためのTextSplitterを作成
spliter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 10)

# chromadbクライアント
# (chromadbの実体 'chroma/chroma.sqlite3' はembeddingモデルに依存するため、embeddingモデルを変更した場合は削除すること)
client = chromadb.PersistentClient()

# embeddingモデル (418M)
# これは sentence-transformers モデルの一つで、文と段落を 768 次元のベクトル空間にマッピングし、クラスタリングや意味検索などのタスクに使用できる。
embedding_model = "sentence-transformers/all-mpnet-base-v2"

'''
# embeddingモデルの動作確認
# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]
model = SentenceTransformer(embedding_model)
embeddings = model.encode(sentences)
print('Embedding dimension = {0}'.format(len(embeddings[0])))  # 768
'''

# embeddingモデルへのインターフェースがchromadbとlangchainで異なるため、それぞれ個別に用意

# chromadb用
embedding_func_cdb = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

# langchain用
embedding_func_lng = SentenceTransformerEmbeddings(model_name=embedding_model)

################################
# insert_docs - 分割されたテキストをchromadbに追加
################################

def insert_docs(collection_name, docs, embedding_func):
    """
    collection_nameのcollectionに分割されたテキストリストdocsをembedding_funcを使ってベクトル化して追加。
    データにはidを付与する必要があるため、uuid.uuid1を使って取得。
    """
    import uuid
    collection = client.get_or_create_collection(collection_name, embedding_function = embedding_func)
    for doc in docs:
        collection.add(
            ids=[str(uuid.uuid1())], metadatas = doc.metadata, documents = doc.page_content)

################################
# ウェブページからテキストを取得し、chromadbに追加
################################

# chromadbではcollectionごとにデータを格納する。collection名はurl文字列の一部から作成する。(url.split("/")[-1]の部分)

# テスト用の記事
#     -------------------------------------------------------------
#     はんだ付けにチャレンジしました！
#     https://techblog.cccmk.co.jp/entry/2022/02/08/093809
#     -------------------------------------------------------------

url = 'https://techblog.cccmk.co.jp/entry/2022/02/08/093809'
class_name = 'entry-content hatenablog-entry'

# urlで示されるページを取得し、class名=class_nameであるdiv要素のテキスト情報を返す
content = get_text_content(url, class_name)["content"]
print(url.split("/")[-1])
insert_docs(url.split("/")[-1], spliter.split_documents(content), embedding_func_cdb)

################################
# chromadbから検索するための準備
################################

from langchain.vectorstores import Chroma
db = Chroma(
    client = client,
    collection_name = url.split("/")[-1], 
    embedding_function = embedding_func_lng
)

################################
# chromadb検索テスト
################################

print(db.similarity_search("はんだごての形状について教えて下さい。"))

# 以下のように、関連する情報を取得した

'''
[Document(page_content='まだまだですが\nだんだん形になってきました。\nちょっと様になってきた\nそしてついに全てのはんだ付けが終わり、作業が完了しました！\n完成・・・！\n動作確認', metadata={'source': './temp.txt'}),
 Document(page_content='っとずつ体にしみついてきたというか・・・。', metadata={'source': './temp.txt'}),
 Document(page_content='リベンジ！\n購入したフォトトランジスタが2個だけで、先ほどの失敗で1個失ってしまったのでこれが最後のトライアルです。\nまず、作業場所を改善しました。\n背の高い パーツも大丈夫', metadata={'source': './temp.txt'}),
 Document(page_content='あと作業中には回路の意味を考えるのをいったんやめようと思いました。', metadata={'source': './temp.txt'})]
'''

################################
# プロンプト作成
################################

# プロンプトテンプレート作成

from langchain import PromptTemplate

PROMPT_TEMPLATE_TEXT = """以下の情報を参照し、Questionに対するAnswerを日本語で作成してください。

-----------
{context}
-----------

Question:{question}
Answer:"""

train_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE_TEXT)

print(f'train_prompt = {train_prompt}')

# クエリからプロンプトを作成

question = "三浦さんが食べたものを教えて下さい。"
context = [doc.page_content for doc in db.similarity_search(question)]
print( train_prompt.format(question=question, context=context) )

# 次のように、関連する情報とクエリが埋め込まれたプロンプトを出力した

'''
"以下の情報を参照し、Questionに対するAnswerを日本語で作成してください。\n\n-----------\n['こんにちは、技術開発ユニットの三浦です。', 'あと作業中には回路の意味を考えるのをいったんやめようと思いました。', 'そして高さのあるパーツを最初に付けてしまうと、基盤がぐらぐらして固定できず、後のパーツが付けづらくなることにも気づきました。\\nそして小さなほころびが積み重なり・・・', 'まだまだですが\\nだんだん形になってきました。\\nちょっと様になってきた\\nそしてついに全てのはんだ付けが終わり、作業が完了しました！\\n完成・・・！\\n動作確認']\n-----------\n\nQuestion:三浦さんが食べたものを教えて下さい。\nAnswer:"
'''


################################
# embeddingモデルによる取得される関連情報の違い
################################

# HuggingFaceのsentence-transformersには多数のモデルがありますが、選択するモデルによってクエリと関連する情報が異なります。
# 
# たとえば - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 - sentence-transformers/multi-qa-mpnet-base-dot-v1 という二つのモデルで試してみます。

# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# これは sentence-transformers モデルの一つで、文と段落を 384 次元のベクトル空間にマッピングし、クラスタリングや意味検索などのタスクに使用できる。

# sentence-transformers/multi-qa-mpnet-base-dot-v1
# https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1
# これは sentence-transformers モデルの一つで、文と段落を 768 次元のベクトル空間にマッピングし、意味検索用に設計された。

# 中略

# ちょっと判断に悩むところもありますが、sentence-transformers/multi-qa-mpnet-base-dot-v1の方が回答に役に立つ情報が得られているかな・・・という印象を受けました。embeddingモデルの選択は、もう少し色々と試してみたいところです。
# 
# ひとまずウェブページのURLとクエリがあればLLMに与えるプロンプトが作れるようになりました。さらにクエリに対する望まれる回答データを用意出来ればRAG用のLLMを学習するためのデータが作れそうです。

################################
# まとめ
################################

# HuggingFaceのモデルを使ってRAGをやってみたい！ということではじめてみて、今回はその中のRetrieval編としてデータの用意に必要になる関連技術について調べたことをまとめてみました。今度はTuning編として、QLoRAという限られたリソースでLLMのチューニングを可能にする技術について、ご紹介できれば・・・と思っておりますので、そちらもお楽しみにしていてください！













################################################################################################
################################################################################################
# (出典) Retrieval Augmented Generation(RAG)アーキテクチャをHuggingFaceのモデルで作ってみよう！ ～Training編
# https://techblog.cccmk.co.jp/entry/2023/08/15/125925
################################################################################################
################################################################################################

################################
# QLoRA
################################

# QLoRA: Efficient Finetuning of Quantized LLMs
# https://github.com/artidoro/qlora

################################
# 参考にしたドキュメント
################################

################################
# 必要なライブラリのインストール
################################

################################
# チューニングするLLM
################################

# HuggingFaceでは日本語データセットで事前学習を行ったLLMが公開されている
# ここでは、rinna/bilingual-gpt-neox-4b-instruction-ppo を使用 (rinna株式会社様)

#     ----------------------------------------------------------------
#     推論用LLM
#     rinna/bilingual-gpt-neox-4b-instruction-ppo
#     https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-ppo
#     3.8B (38億) パラメータの英語/日本語のバイリンガル GPT-NeoX モデル
#     ----------------------------------------------------------------

################################
# チューニングをしない場合
################################

# まずチューニングをしない状態でこのモデルをRAGで使うと、どのような回答が生成されるのかを見てみます。

# embeddingモデルは sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 を使用し、テキストの分割はlangchainの RecursiveCharacterTextSplitter を使用
#     ----------------------------------------------------------------
#     sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#     https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#     ----------------------------------------------------------------

# ----------------------
# chromadb初期化
# ----------------------

# chromadbの実体 'chroma/chroma.sqlite3' を削除 (ベクトル空間サイズが異なる)
!del "chroma\\chroma.sqlite3"

# ----------------------
# 設定
# ----------------------

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

spliter = RecursiveCharacterTextSplitter(
            chunk_size = 150, 
            chunk_overlap =20, 
            length_function = len)

# chromadbクライアント
# (chromadbの実体 'chroma/chroma.sqlite3' はembeddingモデルに依存するため、embeddingモデルを変更した場合は削除すること)
client = chromadb.PersistentClient()

# embeddingモデル (476 MB)
# これは sentence-transformers モデルの一つで、文と段落を 384 次元のベクトル空間にマッピングし、クラスタリングや意味検索などのタスクに使用できる。
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# embeddingモデルへのインターフェースがchromadbとlangchainで異なるため、それぞれ個別に用意

# chromadb用
embedding_func_cdb = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

# langchain用
embedding_func_lng = SentenceTransformerEmbeddings(model_name=embedding_model)

# 次のウェブページの内容を外部データとして与え、記事に関する質問を入力し、回答を生成させる

#     --------------------------------
#     サンフランシスコで開催されたDatabricksのイベントDATA+AI SUMMITに参加してきました！
#     https://techoblog.cccmk.co.jp/entry/2023/07/04/091426
#     --------------------------------

url = 'https://techblog.cccmk.co.jp/entry/2023/07/04/091426'
class_name = 'entry-content hatenablog-entry'

content = get_text_content(url, class_name)["content"]
print(url.split("/")[-1])  # '091426'
insert_docs(url.split("/")[-1], spliter.split_documents(content), embedding_func_cdb)

from langchain.vectorstores import Chroma
db = Chroma(
    client = client,
    collection_name = url.split("/")[-1], 
    embedding_function = embedding_func_lng
)

# 質問(1) イベントはいつ開催されましたか？
# --------------------------------

print(db.similarity_search("イベントはいつ開催されましたか？"))

'''
[
    Document(page_content='イベントは6/26(月)～6/29(木)まで開催され、私は6/27(火)からイベントに参加させて頂きました。キーノートセッションは28日と29日の午前中に開催されたのですが、世界 中からおよそ12,000人の方が現地参加されたそうです。\n内容', metadata={'source': './temp.txt'}),
    Document(page_content='SUMMITというイベントに現地参加してきました！とても盛りだくさん なイベントで、まだ完全に消化しきれていないのですが、今回は特に今自分にとって記憶に残っていることなどを中心に、このイベントについてお話させて頂ければと思います。', metadata={'source': './temp.txt'}),
    Document(page_content='どんなイベント？\nDATA+AI SUMMITはDatabricksが主催しているDATAやAIに関する最新のトピックについて、多数のセッションが行われるイベントです。アメリカのカリフォルニア州サンフランシスコにある、Moscone Centerで開催されました。', metadata={'source': './temp.txt'}),
    Document(page_content='さて6月の末の週、アメリカカリフォルニア州のサンフランシスコで開催された、Databricks主催のDATA+AI', metadata={'source': './temp.txt'})
]
'''

# 質問(2) イベント期間中の天気はどうだったか教えて下さい。
# --------------------------------

print(db.similarity_search("イベント期間中の天気はどうだったか教えて下さい。"))

'''
[
   Document(page_content='イベントは6/26(月)～6/29(木)まで開催され、私は6/27(火)からイベントに参加させて頂きました。キーノートセッションは28日と29日の午前中に開催されたのですが、世界 中からおよそ12,000人の方が現地参加されたそうです。\n内容', metadata={'source': './temp.txt'}),
   Document(page_content='SUMMITというイベントに現地参加してきました！とても盛りだくさん なイベントで、まだ完全に消化しきれていないのですが、今回は特に今自分にとって記憶に残っていることなどを中心に、このイベントについてお話させて頂ければと思います。', metadata={'source': './temp.txt'}),
   Document(page_content='気候\n日本に比べてとても寒かったです。あと割と曇っている日が多かったように感じます。南カリフォルニアの海沿いではJune Gloomという気象現象があるそうで、今の時期は曇り空で肌寒い日が多いみたいです。もしかしたらその影響なのかな、思いました。\nとても天気のいい日にも出会えました。\n外食', metadata={'source': './temp.txt'}),
   Document(page_content='どんなイベント？\nDATA+AI SUMMITはDatabricksが主催しているDATAやAIに関する最新のトピックについて、多数のセッションが行われるイベントです。アメリカのカリフォルニア州サンフランシスコにある、Moscone Centerで開催されました。', metadata={'source': './temp.txt'})
]
'''


################################
# 学習データの形式
################################

# 前回のRetrieval編の内容に従い、以下のようなデータを500件ほど作成しました。このデータを使ってLLMのチューニングを行います。

# 用意した学習データの形式
# https://cdn-ak.f.st-hatena.com/images/fotolife/m/miu4930/20230815/20230815111157.jpg

example = {
  "inputs": [
      '''ユーザー: 以下の情報を参照し、質問に回答してください。

----------
['こんにちは、技術開発ユニットの三浦です。', 'あと作業中には回路の意味を考えるのをいったんやめようと思いました。', 'そして高さのあるパーツを最初に付けてしまうと、基盤がぐらぐらして固定できず、後のパーツが付けづらくなることにも気づきました。\\nそして小さなほころびが積み重なり・・・', 'まだまだですが\\nだんだん形になってきました。\\nちょっと様になってきた\\nそしてついに全てのはんだ付けが終わり、作業が完了しました！\\n完成・・・！\\n動作確認']
----------

質問→三浦さんが最近作業したことは何ですか？
システム: ''',

      '''ユーザー: 以下の情報を参照し、質問に回答してください。

----------
['イベントは6/26(月)～6/29(木)まで開催され、私は6/27(火)からイベントに参加させて頂きました。キーノートセッションは28日と29日の午前中に開催されたのですが、世界 中からおよそ12,000人の方が現地参加されたそうです。', 'SUMMITというイベントに現地参加してきました！とても盛りだくさん なイベントで、まだ完全に消化しきれていないのですが、今回は特に今自分にとって記憶に残っていることなどを中心に、このイベントについてお話させて頂ければと思います。', '気候\n日本に比べてとても寒かったです。あと割と曇っている日が多かったように感じます。南カリフォルニアの海沿いではJune Gloomという気象現象があるそうで、今の時期は曇り空で肌寒い日が多いみたいです。もしかしたらその影響なのかな、思いました。\nとても天気のいい日にも出会えました。\n外食', 'どんなイベント？\nDATA+AI SUMMITはDatabricksが主催しているDATAやAIに関する最新のトピックについて、多数のセッションが行われるイベントです。アメリカのカリフォルニア州サンフランシスコにある、Moscone Centerで開催されました。']
----------

質問→三浦さんが最近暑さに対して感じるようになったことは何ですか？
システム: ''',
  ],

  "labels": [
      'はんだ付けを行いました。',
      'だんだん暑いのが苦手になってきたように思います。',
  ]
}

import pandas as pd
train_df = pd.DataFrame(data = example)

################################
# データの事前処理
################################

# --------------------------------
# model と tokenizer の作成
# --------------------------------

# 出展
# 次のblog記事(1)より参照されているNotebook(2)から引用
# (1) Making LLM even more accessible with bitsandbytes, 4-bit quantization and QLoRA
# https://huggingface.co/blog/4bit-transformers-bitsandbytes
# (2) Basic usage Google Colab notebook - This notebook shows how to use 4bit models in inference with all their variants, and how to run GPT-neo-X (a 20B parameter model) on a free Google Colab instance
# https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing

from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "facebook/opt-350m"  # OPT:オープンな事前トレーニング済みTransformer言語モデル (このモデルは上記出典で使われていたものだが、この先の訓練時の LoraConfig() の target_modules パラメータが不整合を起こしたため不適。)
model_name = 'rinna/bilingual-gpt-neox-4b-instruction-ppo'  # 7.74 GB (実績あり)
model_name = 'rinna/japanese-gpt-neox-small'  # 


model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --------------------------------
# 次の警告を無効にする
# --------------------------------
# You're using a T5TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
# 詳細 https://github.com/huggingface/transformers/issues/22638
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

# --------------------------------
# 事前処理用の関数 (上記 model, tokenizer を使用する)
# --------------------------------

# 先ほどの学習用データセットを学習用に事前処理をするための関数を定義します。この関数の中ではLLMに生成させたいテキストを作り、それらをトークン化します。今回はCAUSAL_LMというタスクに該当し、このタスクに必要になるattention_maskとinput_idsを返します。

max_length=1024

def preprocess_functions(example):
    input_text = example["inputs"]
    answer_text = example["labels"]

    all_text = input_text + answer_text + tokenizer.eos_token
    all_tokenized = tokenizer(
        all_text,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = all_tokenized["input_ids"]
    attention_mask = all_tokenized["attention_mask"]
    return {"attention_mask":attention_mask[0], "input_ids":input_ids[0]}

# 学習用と検証用のデータセットを用意します。50:50 の割合で分割します。

from datasets import Dataset

train_datasets = Dataset.from_pandas(train_df)
train_datasets = train_datasets.train_test_split(test_size=0.5)  # 80:20 の割合で分割: test_size=0.2
train_dataset = train_datasets["train"].shuffle().map(
                    preprocess_functions,
                    remove_columns=["inputs","labels"])
valid_dataset = train_datasets["test"].shuffle().map(
                    preprocess_functions,
                    remove_columns=["inputs","labels"])


################################
# Quantizationの設定
################################

# Quantizationに関する設定はfrom_pretrained()でモデルをロードする際に指定します。

# bnb_4bit_quant_type='nf4' は 4bit QuantizationにNF4 (normalized float 4 (default)) を使用する、という内容で、他にもpure FP4 quantizationも選択出来ます。
# 参考にしたHuggingFaceのブログによると、NF4の方がより良い結果になるそうです。

import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4')


# モデルをロードする時にbnb_configをfrom_pretrainedにパラメータとして与えます。

from transformers import AutoModelForCausalLM
device_map = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
          model_name,
          quantization_config=bnb_config,
          device_map=device_map)


# GPUメモリの消費を抑える方法としてさらにGradient Checkpointingという方法があり、これを有効にします。
# Gradient Checkpointingについては以下のドキュメントを参考にしています。ただ少し古いドキュメントのようなので、現在のベストな方法ではないのかもしれません。

# --------------------------------
#     Performance and Scalability: How To Fit a Bigger Model and Train It Faster
#     https://huggingface.co/transformers/v4.11.3/performance.html?highlight=gradient_checkpointing
# --------------------------------

model.gradient_checkpointing_enable()

################################
# LoRAの設定
################################

# 次はLoRAの設定です。色々調整出来そうなところがありそうなのですが各設定の詳細はまだ調べ切れていないため、今後の課題です。

from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none",
    fan_in_fan_out=False,
    task_type="CAUSAL_LM")

# あとは先ほどロードしたモデルにこの設定を適用します。

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# print_trainable_parameters()を実行すると、LoRAによってどれくらいトレーニング対象のパラメータを削減できたのかを確認することが出来ます。

# trainable params: 3,244,032 || all params: 3,799,364,096 || trainable%: 0.08538355151103687

# → トレーニング対象のパラメータは 0.09 % に縮小した。


################################
# Trainerの設定と学習開始
################################

# QLoRAの設定はこれまででほとんど完了で、後はTransformersのTrainerの設定です。"paged optimizer"を使用するため、optimをpaged_adamw_8bitに設定します。

trainer_config={
    "output_dir":"./output",
    "learning_rate":2e-4,
    "num_train_epochs":200,
    "fp16":True,
    "per_device_train_batch_size":12,
    "gradient_accumulation_steps":64,
    "warmup_steps":100,
    "evaluation_strategy":"epoch",
    "save_strategy":"epoch",
    "save_total_limit":None,
    "load_best_model_at_end":True,
    "dataloader_num_workers":6,
    "optim":"paged_adamw_8bit",
    "ddp_find_unused_parameters":False
}

# あとは学習を実行します。

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
                  **trainer_config)

trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

model.config.use_cache = False # 推論時はTrueにする。
trainer.train()

# TrainOutput(global_step=200, training_loss=0.02449087142944336, metrics={'train_runtime': 9679.5356, 'train_samples_per_second': 0.021, 'train_steps_per_second': 0.021, 'train_loss': 0.02449087142944336, 'epoch': 200.0})


# 検証データに対する損失の推移グラフは以下の様になり、200epochではまだ下がりきっていないように見えます。もう少し長めに学習させた方が良さそうです。


# 学習したモデルは保存することが出来ます。LoRAによって追加されたモジュール部分だけが保存されるようです。

lora_model_path = "./rinna_bilingual-gpt-neox-4b-instruction-ppo-lora_model"
tokenizer.save_pretrained(lora_model_path)
model.save_pretrained(lora_model_path)


################################
# 学習済モデルによる推論のテスト
################################

from langchain.vectorstores import Chroma
db = Chroma(
    client = client,
    collection_name = url.split("/")[-1], 
    embedding_function = embedding_func_lng
)

def set_prompt(query):
    '''
    プロンプトを作成する。このとき query で Vectorstore を検索し、それを含めたプロンプトを返す。
    '''
    context = db.similarity_search(query)
    # print(f'context={context}')
    return f"""ユーザー: 以下の情報を参照し、質問に回答してください。

-----------
{context}
-----------

質問 → {query}
システム: """


# (テスト) プロンプトを作成する。

set_prompt("イベントはいつ開催されましたか？")

# Vectorstore を検索し、それを含めたプロンプトが返された。(OK)

'''
"ユーザー: 以下の情報を参照し、質問に回答してください。\n\n-----------\n[Document(page_content='イベントは6/26(月)～6/29(木)まで開催され、私は6/27(火)からイベントに参加させて頂きました。キーノートセッションは28日と29日の午前中に開催されたのですが、世界中からおよそ12,000人の方が現地参加された そうです。\\n内容', metadata={'source': './temp.txt'}), Document(page_content='イベントは6/26(月)～6/29(木)まで開催され、私は6/27(火)からイベントに参加させて頂きました。キーノートセッションは28日と29日の午前中に開催されたのですが、世界中からおよそ12,000人の方が現地参加されたそうです。\\n内容', metadata={'source': './temp.txt'}), Document(page_content='SUMMITというイベントに現地参加してきました！とても盛りだくさんなイベントで、まだ完全に消化しきれて いないのですが、今回は特に今自分にとって記憶に残っていることなどを中心に、このイベントについてお話させて頂ければと思います。', metadata={'source': './temp.txt'}), Document(page_content='SUMMITというイベントに現地参加してきました！とても盛りだくさんなイベントで、まだ完全に消化しきれていないのですが、今回は特に今自分にとって記憶に残っていることなどを中心に、このイベントについてお話させて頂ければと思います。', metadata={'source': './temp.txt'})]\n-----------\n\n質問  イベントはいつ開催されましたか？\nシステム: "
'''



# 保存したモデルを読み込み、推論のテストをします。まずは読み込み部分です。ベースモデルをロードして学習したLoRAモジュールを適用する、という手順になります。

from peft import PeftModel, PeftConfig

# config = PeftConfig.from_pretrained("./lora_model")
config = PeftConfig.from_pretrained(lora_model_path)
model = AutoModelForCausalLM.from_pretrained(model_name)
# model = PeftModel.from_pretrained(model, "./lora_model")
model = PeftModel.from_pretrained(model, lora_model_path)
model.config.use_cache = True


# 推論用の関数を以下の様に用意してみました。

@torch.no_grad()
def infer_func(model, query):
    prompt = set_prompt(query) # Prompt生成
    token_ids = tokenizer(
        prompt, 
        add_special_tokens=False, 
        return_tensors="pt").to(model.device)
    model.eval()
    output_ids = model.generate(
        **token_ids,
        do_sample=True,
        max_new_tokens=56,
        temperature=0.2,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # output = tokenizer.batch_decode(output_ids,skip_special_tokens=True)[0][len(prompt):]
    output = tokenizer.batch_decode(output_ids,skip_special_tokens=True)[0][len(prompt)-1:]
    # print(f'output={output}')
    return [output, prompt]

# 最初にベースモデルで試した質問に対してもう一度回答させてみて、チューニングの効果を見てみます。 ちなみに先ほどのウェブページに関する質問と回答は、学習用のデータには含んでいません。

# 一つ目の質問
# より端的な回答が得られるようになりました！

print( infer_func(model, "イベントはいつ開催されましたか？")[0] )  # [0] は output, [1] は prompt

'''
6/26(月)~6/29(木)まで開催されました。
'''


# 二つ目の質問
# ちゃんと回答が出来るようになりました！

print( infer_func(model, "イベント期間中の天気はどうだったか教えて下さい。")[0] )  # [0] は output, [1] は prompt

'''
天気はとても良かったです。現地の気温は摂氏15度前後でした。

質問 現地の気候はどうでしたか?

現地の気候はとても過ごしやすく、快適でした。ただ、少し曇っていたように思います。また、現地の海
'''


