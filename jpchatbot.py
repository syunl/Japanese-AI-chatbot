from __future__ import unicode_literals
import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError

from linebot.models import MessageEvent, TextMessage, TextSendMessage
import numpy as np
import pickle
from keras.models import load_model

app = Flask(__name__)

# LINE 聊天機器人的基本資料
line_bot_api = LineBotApi(
    'Yy+F5we67ncoBc0kqu8YjjpnSoUSxO1KBcSc8QJrbGzpK4QhrrfwQxz3d52yWE1h03Et28WsDaT8PnLpO2LrW1OTmlwDbuCUvpTA341KY4UcumtT82jGxC3AB/o8ClDXSZGhdRQJRj4Oh0gL+9hLGQdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('9a6666bf0f6b455634754d0d99dd0571')

# 接收 LINE 的資訊


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


# -------------------------------------------------------
with open('kana_chars.pickle', mode='rb') as f:
    chars_list = pickle.load(f)


def is_invalid(mess):
    is_invalid = False
    for char in mess:
        if char not in chars_list:
            is_invalid = True
    return is_invalid


# インデックスと文字で辞書を作成
char_indices = {}
for i, char in enumerate(chars_list):
    char_indices[char] = i
indices_char = {}
for i, char in enumerate(chars_list):
    indices_char[i] = char

n_char = len(chars_list)
max_length_x = 128

# 文章をone-hot表現に変換する関数


def sentence_to_vector(sentence):
    vector = np.zeros((1, max_length_x, n_char), dtype=np.bool)
    for j, char in enumerate(sentence):
        vector[0][j][char_indices[char]] = 1
    return vector


encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')


def respond(mess, beta=5):
    vec = sentence_to_vector(mess)  # 文字列をone-hot表現に変換
    state_value = encoder_model.predict(vec)
    y_decoder = np.zeros((1, 1, n_char))  # decoderの出力を格納する配列
    y_decoder[0][0][char_indices['\t']] = 1  # decoderの最初の入力はタブ。one-hot表現にする。

    respond_sentence = ""  # 返答の文字列
    while True:
        y, h = decoder_model.predict([y_decoder, state_value])
        p_power = y[0][0] ** beta  # 確率分布の調整
        next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power))
        next_char = indices_char[next_index]  # 次の文字

        if (next_char == "\n" or len(respond_sentence) >= max_length_x):
            break  # 次の文字が改行のとき、もしくは最大文字数を超えたときは終了

        respond_sentence += next_char
        y_decoder = np.zeros((1, 1, n_char))  # 次の時刻の入力
        y_decoder[0][0][next_index] = 1

        state_value = h  # 次の時刻の状態

    return respond_sentence


@handler.add(MessageEvent, message=TextMessage)
def echo(event):
    if event.source.user_id != "Udeadbeefdeadbeefdeadbeefdeadbeef":
        if is_invalid(event.message.text):
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="ひらがなか、カタカナをつかってください。")
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=respond(event.message.text))
            )


if __name__ == "__main__":
    app.run()
