from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import base64
import json

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSONが見つかりません")
    return json.loads(text[start:end+1])


@app.route("/api/makeup", methods=["POST"])
def makeup():
    try:
        # ========= 画像取得 =========
        image_file = request.files.get("nail_image")
        if not image_file:
            return jsonify({"error": "爪の画像が必要です"}), 400

        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        # ========= テキスト情報 =========
        text_info = []
        for k in request.form:
            text_info.append(f"{k}: {request.form.getlist(k)}")

        user_text = "\n".join(text_info)

        # ========= ネイル案生成（幅広く） =========
        idea_prompt = f"""
あなたは一流のネイルデザイナーです。

以下のお客様情報と爪の写真をもとに、
【1案のみ】ネイルデザインを提案してください。

制約：
- ベージュ・ワンカラーに寄りすぎないこと
- ニュアンス、透明感、コントラスト、アート要素、素材感（マット・シアー・ミラー等）を自由に使って良い
- ただし「実際のサロンで再現可能」な範囲にすること
- 個性は出してよいが奇抜すぎないこと

出力は必ず次のJSON形式のみ：

{{
  "concept": "お客様向けのネイルコンセプト説明（情緒的・わかりやすく）",
  "design_description": "色・配置・質感・ポイントを具体的に",
  "image_prompt": "英語。実写ネイル写真。手元アップ。背景はシンプル。幅広い色表現を許可するプロンプト"
}}

お客様情報：
{user_text}
"""

        idea_res = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": idea_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.85
        )

        idea_raw = idea_res.choices[0].message.content
        idea = extract_json(idea_raw)

        # ========= 画像生成 =========
        image_res = client.images.generate(
            model="gpt-image-1",
            prompt=idea["image_prompt"],
            size="1024x1024"
        )

        image_url = image_res.data[0].url

        return jsonify({
            "plan": f"""【ネイルコンセプト】
{idea['concept']}

【デザイン詳細】
{idea['design_description']}
""",
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
