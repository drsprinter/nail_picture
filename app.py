from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import re

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# ユーティリティ
# -----------------------------
def safe_list(val):
    return val if isinstance(val, list) else []

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"(nude|sexy|sensual|erotic|teen|young|schoolgirl)", "", s, flags=re.I)
    return s.strip()[:400]

def build_safe_image_prompt(style, palette, design):
    return (
        "Photorealistic close-up of a hand with elegant gel nails. "
        "Clean studio background, soft natural lighting, "
        "natural light beige skin tone. "
        "No text, no watermark, no logo. "
        f"Nail design: {clean_text(design)}. "
        f"Color palette: {clean_text(', '.join(palette))}. "
        f"Style: {clean_text(style)}. "
        "Minimal, refined, professional nail salon photography."
    )

# -----------------------------
# API
# -----------------------------
@app.route("/api/makeup", methods=["POST", "OPTIONS"])
def makeup():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json()

        # --- 入力整理 ---
        info = []
        for k, v in data.items():
            if isinstance(v, list):
                info.append(f"{k}: {', '.join(v)}")
            else:
                info.append(f"{k}: {v}")
        user_info = "\n".join(info)

        # -----------------------------
        # ① 3方向ネイル案（JSON）
        # -----------------------------
        system_ideas = (
            "あなたはプロのネイリストです。"
            "以下の情報から、方向性が明確に異なるネイルデザイン案を3つ提案してください。\n"
            "条件:\n"
            "- シンプル系（ラメ・ストーン無し）\n"
            "- 色味・印象・雰囲気が被らない\n\n"
            "出力は必ずJSONのみ。\n"
            "{"
            '"proposals": ['
            '{"id":1,"name":"","style":"","palette":[],"design":"","why":""}'
            "]}"
        )

        ideas_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_ideas},
                {"role": "user", "content": user_info}
            ],
            temperature=0.8
        )

        proposals = json.loads(ideas_res.choices[0].message.content)["proposals"]

        # -----------------------------
        # ② 各案の詳細プラン
        # -----------------------------
        results = []

        for p in proposals:
            plan_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content":
                        "以下のネイル案について文章を作成してください。\n"
                        "[お客様向けネイルコンセプト]\n"
                        "[サロン向け技術メモ]（プリジェル調合比率含む）"
                    },
                    {
                        "role": "user",
                        "content": f"{user_info}\n\n提案:\n{json.dumps(p, ensure_ascii=False)}"
                    }
                ],
                temperature=0.7
            )

            plan_text = plan_res.choices[0].message.content.strip()

            # -----------------------------
            # ③ 画像生成（Verify済み gpt-image-1）
            # -----------------------------
            image_url = None
            try:
                img_prompt = build_safe_image_prompt(
                    p.get("style", ""),
                    p.get("palette", []),
                    p.get("design", "")
                )

                img = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size="1024x1024"
                )
                image_url = img.data[0].url

            except Exception as img_err:
                image_url = None

            results.append({
                "id": p.get("id"),
                "name": p.get("name"),
                "style": p.get("style"),
                "palette": p.get("palette"),
                "design": p.get("design"),
                "why": p.get("why"),
                "plan": plan_text,
                "image_url": image_url
            })

        return jsonify({
            "status": "ok",
            "proposals": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
