from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import io
import json

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def safe_extract_json(text: str) -> dict:
    """Chatの出力からJSONだけを安全に取り出す"""
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1].strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSONが見つかりませんでした: " + (text[:200] if text else ""))
    return json.loads(t[start:end+1])

@app.route("/api/makeup", methods=["POST", "OPTIONS"])
def makeup():
    if request.method == "OPTIONS":
        return "", 204

    try:
        # ===== 1) 画像必須 =====
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({
                "error": "爪の写真が必要です（image が見つかりません）",
                "debug_files_keys": list(request.files.keys()),
                "debug_form_keys": list(request.form.keys())
            }), 400

        img_bytes = image_file.read()
        if not img_bytes:
            return jsonify({"error": "爪の写真が空でした（ファイルサイズ0の可能性）"}), 400

        img_stream = io.BytesIO(img_bytes)
        img_stream.name = "nail.jpg"  # SDKがファイル名を必要とする場合がある

        # ===== 2) フォーム情報 =====
        # checkbox等は同名が複数来るので getlist で拾う
        info_lines = []
        for k in request.form.keys():
            vals = request.form.getlist(k)
            if len(vals) == 1:
                info_lines.append(f"{k}: {vals[0]}")
            else:
                info_lines.append(f"{k}: {', '.join(vals)}")
        user_text = "\n".join(info_lines).strip()

        # ===== 3) 日本語プラン（控えめアップデート）=====
        plan_prompt = f"""
あなたはプロのネイリストです。
以下のお客様情報をもとに【1案】のネイル提案を作ってください。

絶対条件：
- お客様の選択項目（vibe /purpose /  avoid_colors / nail_duration / age）を最優先で踏襲
- 奇抜すぎない：新しさは50%程度（新しい自分を発見できる）
- 日常に馴染む範囲で、洗練された雰囲気。ただし、マットに寄りすぎず艶も適度に
- ベージュ単色に寄りすぎない
- ストーンやラメ、アートなどは回答に応じて適切に

出力形式（このまま）：
【ネイルコンセプト】
（お客様向けに短く）
【デザイン詳細】
（色・配置・質感・ポイント。選択項目との整合が分かるように）

お客様情報：
{user_text}
""".strip()

        plan_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたはトップネイルアーティストです。"},
                {"role": "user", "content": plan_prompt}
            ],
            temperature=0.55
        )
        plan_text = (plan_res.choices[0].message.content or "").strip()

        # ===== 4) 画像編集プロンプト（英語）をJSONで作る =====
        spec_prompt = f"""
You are a top nail artist.
Create an English image-edit prompt based strictly on the user's preferences.

Hard rules:
- Keep the same hand, skin tone, lighting, background, and composition
- Edit ONLY the nails (do NOT change fingers, skin, jewelry, background)
- Follow user's choices first
- Add only 10–20% novelty (subtle twist)
- Elegant, wearable, salon-realistic
- Avoid bold patterns, avoid neon, avoid heavy glitter, avoid large stones/decals
- No text, no watermark

Return ONLY valid JSON:
{{
  "edit_prompt_en": "..."
}}

User preferences:
{user_text}

Nail plan (for reference):
{plan_text}
""".strip()

        spec_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write prompts for image editing."},
                {"role": "user", "content": spec_prompt}
            ],
            temperature=0.35
        )
        spec = safe_extract_json(spec_res.choices[0].message.content)
        edit_prompt = (spec.get("edit_prompt_en") or "").strip()

        if not edit_prompt:
            edit_prompt = (
                "Edit the uploaded photo. Keep the same hand, skin tone, lighting, background, and composition. "
                "Change ONLY the nail design. Elegant, wearable, salon-realistic with a subtle twist (10-20% novelty). "
                "Avoid bold patterns, neon, heavy glitter, and large stones/decals. No text, no watermark."
            )

        # ===== 5) 画像編集（gpt-image-1）=====
        image_data_url = None
        image_error = None

        try:
            # 公式例と同様に images.edit を使用 :contentReference[oaicite:0]{index=0}
            img_res = client.images.edit(
                model="gpt-image-1",
                image=img_stream,
                prompt=edit_prompt,
                size="1024x1024"
            )

            # b64_json or url
            b64 = getattr(img_res.data[0], "b64_json", None)
            url = getattr(img_res.data[0], "url", None)

            if b64:
                image_data_url = "data:image/png;base64," + b64
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
