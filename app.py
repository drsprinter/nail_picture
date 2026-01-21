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


def safe_extract_json(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1].strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSONが見つかりませんでした")
    return json.loads(t[start:end+1])


@app.route("/api/makeup", methods=["POST", "OPTIONS"])
def makeup():
    if request.method == "OPTIONS":
        return "", 204

    try:
        # ===== 画像（必須）=====
        image_file = request.files.get("nail_image")
        if not image_file:
            return jsonify({"error": "爪の写真（nail_image）が必要です"}), 400

        img_bytes = image_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # ===== フォーム情報 =====
        info_lines = []
        for k in request.form:
            vals = request.form.getlist(k)
            if len(vals) == 1:
                info_lines.append(f"{k}: {vals[0]}")
            else:
                info_lines.append(f"{k}: {', '.join(vals)}")
        user_text = "\n".join(info_lines).strip()

        # ===== 1) プラン + 編集プロンプト（遊び多め）=====
        spec_prompt = f"""
あなたはトップネイルアーティストです。
以下の「お客様情報」と「爪の写真」を参考に、"シンプル寄りに逃げず" 遊び心とアート感を出した【1案】を作ってください。

制約：
- 元の手・指・肌色・写真のライティングはそのまま（写真編集）
- 変えるのは「ネイルデザイン」だけ
- ラメ/ストーンを必須にしない（入れるなら控えめな表現）
- ベージュ単色に寄らない：色相の幅、質感の幅（マット/グロス/ミラー風/オーロラ風）を使う
- 現実のサロンで再現できる範囲のアート

次のJSONのみで出力：

{{
  "concept_jp": "お客様向けの短いコンセプト（日本語）",
  "detail_jp": "色・配置・質感・ポイント（日本語）",
  "edit_prompt_en": "English prompt for editing the uploaded photo. Keep same hand/skin/lighting. Change ONLY nails. playful, artistic, expressive, multi-color, salon-realistic. No text."
}}

お客様情報：
{user_text}
"""

        idea_res = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": spec_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ],
                }
            ],
            temperature=0.9
        )

        idea = safe_extract_json(idea_res.choices[0].message.content)

        plan_text = f"""【ネイルコンセプト】
{idea.get("concept_jp","")}

【デザイン詳細】
{idea.get("detail_jp","")}
""".strip()

        # ===== 2) 画像編集（gpt-image-1 に image を渡す）=====
        image_data_url = None
        image_error = None

        edit_prompt = (idea.get("edit_prompt_en") or "").strip()
        if not edit_prompt:
            edit_prompt = (
                "Edit the uploaded photo. Keep the same hand, skin tone, and lighting. "
                "Change ONLY the nail design. Make it playful, artistic, expressive, multi-color, "
                "salon-realistic. No text."
            )

        try:
            img_res = client.images.generate(
                model="gpt-image-1",
                # ✅ ここが編集：元画像を渡す
                image=[f"data:image/jpeg;base64,{img_b64}"],
                prompt=edit_prompt,
                size="1024x1024",
            )

            b64_json = getattr(img_res.data[0], "b64_json", None)
            url = getattr(img_res.data[0], "url", None)

            if b64_json:
                image_data_url = "data:image/png;base64," + b64_json
            elif url:
                image_data_url = url
            else:
                image_error = "画像データがレスポンスに含まれていません（b64_json/url共に無し）"

        except Exception as e:
            image_error = str(e)

        return jsonify({
            "plan": plan_text,
            "image_data_url": image_data_url,
            "image_error": image_error
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
