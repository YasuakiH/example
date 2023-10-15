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
ipython
%run Retrieval_Augmented_Generation_学習済モデルによる推論.py
'''

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
    content = BeautifulSoup(html, features="html.parser").find("div", attrs ={"class":class_name}).text

    # テキストファイルとして出力
    with open(textfile, "w") as f:
        f.write(content)

    # テキストファイルから入力
    loader = TextLoader(textfile)
    data = loader.load()

    return {"url":url, "class_name":class_name, "content":data}


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

# --------------------------------
# model と tokenizer の作成
# --------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'rinna/bilingual-gpt-neox-4b-instruction-ppo'  # 7.74 GB

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


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




################################
# 学習済モデルによる推論のテスト
################################

lora_model_path = "./rinna_bilingual-gpt-neox-4b-instruction-ppo-lora_model"

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


# 保存したモデルを読み込み、推論のテストをします。まずは読み込み部分です。ベースモデルをロードして学習したLoRAモジュールを適用する、という手順になります。

from peft import PeftModel, PeftConfig

# config = PeftConfig.from_pretrained("./lora_model")
config = PeftConfig.from_pretrained(lora_model_path)
model = AutoModelForCausalLM.from_pretrained(model_name)
# model = PeftModel.from_pretrained(model, "./lora_model")
model = PeftModel.from_pretrained(model, lora_model_path)
model.config.use_cache = True


# 推論用の関数を以下の様に用意してみました。

# import torch
# @torch.no_grad()
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

print( infer_func(model, "イベントはいつ開催されましたか？")[0] )  # [0] は output, [1] は prompt

# より端的な回答が得られるようになりました！

'''
イベントは6月26日から6月29日まで開催されました。
'''

# 二つ目の質問
print( infer_func(model, "イベント期間中の天気はどうだったか教えて下さい。")[0] )  # [0] は output, [1] は prompt

# ちゃんと回答が出来るようになりました！

'''
天気はとても良かったです!現地の方々もとても楽しんでいたようでした。また、会場内は非常に混雑しており、非常に活気がありました。
'''

print('successfully completed !')
