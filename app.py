from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from openai import OpenAI
import os
import base64

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------
# 既存：テキストプラン + 画像生成（任意で残すなら）
# ------------------------------------------------------------
@app.route('/api/makeup', methods=['POST', 'OPTIONS'])
def makeup():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json()
    try:
        info = []
        for key, value in data.items():
            if isinstance(value, list):
                info.append(f"{key}: {', '.join(value)}")
            else:
                info.append(f"{key}: {value}")
        prompt = "\n".join(info)

        system_content = (
            "あなたはプロのネイリストです。"
            "以下のお客様情報をもとに、ネイルプランを生成してください。"
            "出力は次の2つのセクションに分けてください：\n"
            "[お客様向けネイルコンセプト]：感性に響く丁寧な説明、使用カラーやイメージなど。\n"
            "[サロン向け技術メモ]：プリジェル顔料を使ったカラー調合比率、使用カラー名、塗布順、ポイントなど。"
            "特にカラー調合でお客さんに合わせる部分は大事なので今回のネイルプランに合った調合比率や、"
            "明るくしたいもしくは暗くしたい場合の調合方法やカラーや比率も合わせてあると良い。"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        plan = response.choices[0].message.content.strip()

        return jsonify({
            'status': 'ネイルプランの生成が完了しました。',
            'plan': plan
        })

    except Exception as e:
        return jsonify({'status': 'エラーが発生しました', 'error': str(e)}), 500


# ------------------------------------------------------------
# NEW：爪写真を受け取ってデザイン変更（Try-on）
# ------------------------------------------------------------
@app.route('/api/tryon', methods=['POST', 'OPTIONS'])
def tryon():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if "nail_photo" not in request.files:
            return jsonify({"error": "nail_photo が送信されていません"}), 400

        photo_file = request.files["nail_photo"]
        photo_bytes = photo_file.read()

        # 文字情報（任意）
        age = request.form.get("age", "")
        lifestyle = request.form.get("lifestyle", "")
        nail_duration = request.form.get("nail_duration", "")
        avoid_colors = request.form.get("avoid_colors", "")

        purpose = request.form.getlist("purpose")
        vibe = request.form.getlist("vibe")

        # 1) まずネイルプラン（テキスト）を生成
        user_info = []
        user_info.append(f"age: {age}")
        user_info.append(f"lifestyle: {lifestyle}")
        user_info.append(f"nail_duration: {nail_duration}")
        user_info.append(f"purpose: {', '.join(purpose) if purpose else ''}")
        user_info.append(f"vibe: {', '.join(vibe) if vibe else ''}")
        user_info.append(f"avoid: {avoid_colors}")

        plan_system = (
            "あなたはプロのネイリストです。"
            "以下のお客様情報をもとに、シンプルで上品なジェルネイルの提案をしてください。"
            "ラメ・ストーン・派手柄は禁止。"
            "出力は次の2つのセクションに分けてください：\n"
            "[お客様向けネイルコンセプト]：感性に響く丁寧な説明、使用カラーやイメージなど。\n"
            "[サロン向け技術メモ]：プリジェル顔料を使ったカラー調合比率、使用カラー名、塗布順、ポイントなど。"
            "明るくしたい/暗くしたい場合の調合アレンジも添える。"
        )

        plan_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": plan_system},
                {"role": "user", "content": "\n".join(user_info)}
            ],
            temperature=0.7
        )
        plan = plan_resp.choices[0].message.content.strip()

        # 2) 画像編集用のプロンプトを plan から英語で作る（短く・安全に）
        img_prompt_sys = (
            "Create a short English prompt for an image edit model.\n"
            "Goal: edit ONLY the nail design in the provided photo.\n"
            "Keep the hand, skin tone, lighting, and background unchanged and photorealistic.\n"
            "Style must be simple, elegant, no glitter, no rhinestones.\n"
            "Return ONLY the prompt text."
        )

        img_prompt_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": img_prompt_sys},
                {"role": "user", "content": plan}
            ],
            temperature=0.3
        )
        image_prompt = img_prompt_resp.choices[0].message.content.strip()

        # 3) 画像編集（爪だけ変更）
        # --- 注意：ここはSDK/モデル対応で書き方が変わることがあります ---
        # 返却は base64 を推奨（URLだと期限切れ・CORS等で表示失敗が起きやすい）
        #
        # ✅ もしここで AttributeError 等が出る場合、そのエラーを貼ってください。
        #    あなたのSDKに合う呼び方へ即修正します。

        # photo_bytes を base64 にして渡す方式（SDKが file-like を要求する場合は BytesIO に変えてください）
        from io import BytesIO
        image_file_like = BytesIO(photo_bytes)
        image_file_like.name = "nail.png"

        edit_result = client.images.edits.create(
            # 画像編集対応モデル名は環境により異なる場合があります
            # 例: "gpt-image-1" / "dall-e-3" は generate のみ、など差が出ます
            model="gpt-image-1",
            image=image_file_like,
            prompt=image_prompt,
            size="1024x1024"
        )

        # edit_result.data[0].b64_json を想定（モデルにより url の場合もある）
        image_base64 = None
        if hasattr(edit_result.data[0], "b64_json") and edit_result.data[0].b64_json:
            image_base64 = edit_result.data[0].b64_json
        elif hasattr(edit_result.data[0], "url") and edit_result.data[0].url:
            # URLしか返らない場合はフロント表示が不安定なので、まずはURLを返す（暫定）
            # ここは必要ならバック側でダウンロードしてbase64化する実装に変えられます
            return jsonify({
                "plan": plan,
                "image_url": edit_result.data[0].url,
                "note": "image_url returned (base64 not available in this response)"
            })

        return jsonify({
            "plan": plan,
            "image_base64": image_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    return redirect("/")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
