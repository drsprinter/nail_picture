from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import base64
import os

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/api/makeup", methods=["POST"])
def makeup():
    try:
        # ---------- 1. 入力チェック ----------
        if "image" not in request.files:
            return jsonify({"error": "爪の写真が必要です"}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        form_data = request.form.to_dict(flat=False)

        # ---------- 2. ユーザー入力を整形 ----------
        info_lines = []
        for key, values in form_data.items():
            if isinstance(values, list):
                info_lines.append(f"{key}: {', '.join(values)}")
            else:
                info_lines.append(f"{key}: {values}")
        user_text = "\n".join(info_lines)

        # ---------- 3. ネイルプラン生成（日本語） ----------
        plan_prompt = f"""
あなたはプロのネイリストです。
以下のお客様情報をもとに、やりすぎないが洗練されたネイルデザインを【1案】提案してください。

条件：
・選択項目を最優先で尊重する
・奇抜にしすぎない（新しさは10〜20%まで）
・日常に馴染むデザイン
・ベージュ単色に寄りすぎない

出力形式：
【ネイルコンセプト】
【デザイン詳細】

お客様情報：
{user_text}
"""

        plan_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたはトップネイルアーティストです。"},
                {"role": "user", "content": plan_prompt}
            ],
            temperature=0.6
        )

        plan_text = plan_res.choices[0].message.content.strip()

        # ---------- 4. 画像編集プロンプト生成（英語） ----------
        spec_prompt = f"""
You are a professional nail artist.
Create ONE subtle nail design based strictly on the user's preferences.

Rules:
- Keep the same hand, skin tone, lighting, background, and composition
- Edit ONLY the nails
- Follow user's choices first
- Add only 10–20% novelty
- Elegant, salon-realistic, wearable
- Avoid bold patterns, neon, heavy glitter, large stones
- No text, no watermark

User preferences:
{user_text}
"""

        spec_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate image edit prompts."},
                {"role": "user", "content": spec_prompt}
            ],
            temperature=0.4
        )

        edit_prompt_en = spec_res.choices[0].message.content.strip()

        # ---------- 5. 画像編集（爪写真ベース） ----------
        image_res = client.images.generate(
            model="gpt-image-1",
            prompt=edit_prompt_en,
            image=image_b64,
            size="1024x1024"
        )

        image_base64 = image_res.data[0].b64_json
        image_data_url = f"data:image/png;base64,{image_base64}"

        # ---------- 6. レスポンス ----------
        return jsonify({
            "plan": plan_text,
            "image_data_url": image_data_url
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
