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


def extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
    start = t.find("{")
    end = t.rfind("}")
    return json.loads(t[start:end+1])


@app.route("/api/makeup", methods=["POST", "OPTIONS"])
def makeup():
    if request.method == "OPTIONS":
        return "", 204

    try:
        # ===== 画像（必須）=====
        image_file = request.files.get("nail_image")
        if not image_file:
            return jsonify({"error": "爪の写真が必要です"}), 400

        img_bytes = image_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # ===== フォーム情報 =====
        info = []
        for k in request.form:
            vals = request.form.getlist(k)
            info.append(f"{k}: {', '.join(vals)}")
        user_info = "\n".join(info)

        # ===== 1️⃣ ネイルプラン + 編集用プロンプト生成 =====
        design_prompt = f"""
You are a top creative nail artist.

Based on the user's nail photo and answers, create ONE bold nail design.
Do NOT be minimal. Add playful elements, contrast, texture, and artistic flair.

Return ONLY this JSON:

{{
  "concept": "Emotional explanation for the client (Japanese)",
  "design_detail": "Concrete colors, materials, accents, mood (Japanese)",
  "edit_prompt": "English. Edit the uploaded image. Keep the same hand and skin. Change ONLY the nail design. Artistic, playful, expressive, non-minimal."
}}

User info:
{user_info}
"""

        idea_res = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": design_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.9
        )

        idea = extract_json(idea_res.choices[0].message.content)

        plan_text = f"""【ネイルコンセプト】
{idea["concept"]}

【デザイン詳細】
{idea["design_detail"]}
"""

        # ===== 2️⃣ 画像編集（ここが超重要）=====
        edit_prompt = f"""
Edit the uploaded image.

Rules:
- Keep the same hand, fingers, skin tone, lighting, and photo realism
- Change ONLY the nail design
- Be playful, artistic, expressive
- Use multiple colors, textures, or art elements
- Salon-realistic but bold
- No text, no watermark, no logo

Design:
{idea["edit_prompt"]}
"""

        image_res = client.images.edits(
            model="gpt-image-1",
            image=base64.b64decode(img_b64),
            prompt=edit_prompt,
            size="1024x1024"
        )

        b64_img = image_res.data[0].b64_json
        image_data_url = "data:image/png;base64," + b64_img

        return jsonify({
            "plan": plan_text,
            "image_data_url": image_data_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
