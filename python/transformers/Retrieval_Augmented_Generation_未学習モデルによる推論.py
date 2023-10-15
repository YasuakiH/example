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
%run Retrieval_Augmented_Generation_未学習モデルによる推論.py
'''

################################
# チューニングをしない場合
################################

# まずチューニングをしない状態でこのモデルをRAGで使うと、どのような回答が生成されるのかを見てみます。

# embeddingモデルは sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 を使用し、テキストの分割はlangchainの RecursiveCharacterTextSplitter を使用
#     ----------------------------------------------------------------
#     sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#     https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#     ----------------------------------------------------------------


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

################################
# 未学習モデルによる推論のテスト
################################

from langchain.vectorstores import Chroma
db = Chroma(
    client = client,
    collection_name = url.split("/")[-1], 
    embedding_function = embedding_func_lng)

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

print(db.similarity_search("イベントはいつ開催されましたか？"))

# 推論用の関数
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

# 質問(1) イベントはいつ開催されましたか？
# --------------------------------
print( infer_func(model, "イベントはいつ開催されましたか？")[0] )  # [0] は output, [1] は prompt


# 以下の回答が生成されました。(計4回実施)
# 最終的な答え「6/26(月)～6/29(木)」が含まれており、取り上げている情報は正しいです。

'''
 上記の情報を参照して、答えを見つけてください。

 関連情報は「イベントは6/26(月)~6/29(木)」です。したがって、答えは6月26日(月)から6月29日(木)です。

 上記の情報を参照して、答えを見つけてください。

 関連情報は「キーノートセッションは28日と29日の午前中に開催された」ということです。したがって、答えは6月27日(火)からです。
'''



# 質問(2) イベント期間中の天気はどうだったか教えて下さい。
# --------------------------------
print( infer_func(model, "イベント期間中の天気はどうだったか教えて下さい。")[0] )  # [0] は output, [1] は prompt


# 以下の回答が生成されました。(計4回実施)
# 正しい回答は得られませんでした。チューニングをしなくても上手くいきそうなケースはありそうですが、もう少し端的に回答を表現出来、そして上手く回答が出来るケースが増やせると良いと思いました。これをチューニングすることで実現できるか、試してみます。

'''
 答え → とても天気のいい日にも出会えました。

 この情報が関連しています。

 はい、天気に関する情報を提供します。

 この情報が関連しています。
'''

print('successfully completed !')

