from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import io

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
        img_stream = io.BytesIO(img_bytes)
        img_stream.name = "nail.jpg"  # OpenAI SDKによってはファイル名必須

        # ===== フォーム情報 =====
        info_lines = []
        for k in request.form:
            vals = request.form.getlist(k)
            if len(vals) == 1:
                info_lines.append(f"{k}: {vals[0]}")
            else:
                info_lines.append(f"{k}: {', '.join(vals)}")
        user_text = "\n".join(info_lines).strip()

        # ===== 1) 遊び多めな「編集指示」と「プラン」を作る =====
        spec_prompt = f"""
あなたはトップネイルアーティストです。
ユーザーの爪写真をベースに「元の手・指・肌・光」は絶対に保持しつつ、ネイルデザインだけを大胆に変える提案を【1案】作ってください。

方向性（遊び多め）：
- ベージュ単色に逃げない（色相の幅、素材感の幅）
- ただしギラギラしすぎず上品さは残す
- “写真編集”として成立（手の形や肌は変えない）
- ストーンやラメは必須にしない（入れるなら控えめに）
- 現実のサロンで再現可能な範囲

次のJSONのみで出力：

{{
  "concept_jp": "お客様向けの短いコンセプト（日本語）",
  "detail_jp": "色・配置・質感・ポイント（日本語）",
  "edit_prompt_en": "English. Edit the uploaded photo. Keep the same hand/skin/lighting. Change ONLY the nail design. Playful, artistic, expressive, multi-color, salon-realistic. No text."
}}

お客様情報：
{user_text}
"""

        idea_res = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": spec_prompt}],
            temperature=0.9
        )
        idea = safe_extract_json(idea_res.choices[0].message.content)

        plan_text = f"""【ネイルコンセプト】
{idea.get("concept_jp","")}

【デザイン詳細】
{idea.get("detail_jp","")}
""".strip()

        # ===== 2) 画像編集（ここが本命）=====
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
            # ✅ ここが「編集」
            img_res = client.images.edits(
                model="gpt-image-1",
                image=img_stream,
                prompt=edit_prompt,
                size="1024x1024"
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
