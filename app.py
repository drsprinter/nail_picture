from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import io
import openai

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("OPENAI_VERSION:", openai.__version__)


def safe_extract_json(text: str) -> dict:
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
        # ★デバッグ：送られてきたfilesのキー一覧
        incoming_file_keys = list(request.files.keys())
        incoming_form_keys = list(request.form.keys())

        # ===== 必須：爪画像（フロント name="nail_photo" に合わせる）=====
        image_file = request.files.get("nail_photo")
        if not image_file:
            return jsonify({
                "error": "爪の写真が必要です（nail_photo が見つかりません）",
                "debug_files_keys": incoming_file_keys,
                "debug_form_keys": incoming_form_keys
            }), 400

        img_bytes = image_file.read()
        if not img_bytes:
            return jsonify({
                "error": "爪の写真が空でした（ファイルサイズ0の可能性）",
                "debug_files_keys": incoming_file_keys
            }), 400

        img_stream = io.BytesIO(img_bytes)
        img_stream.name = "nail.jpg"

        # ===== フォーム情報 =====
        info_lines = []
        for k in request.form:
            vals = request.form.getlist(k)
            if len(vals) == 1:
                info_lines.append(f"{k}: {vals[0]}")
            else:
                info_lines.append(f"{k}: {', '.join(vals)}")
        user_text = "\n".join(info_lines).strip()

        # ===== 1) プラン + 編集指示生成（遊び多め）=====
        spec_prompt = f"""
あなたはトップネイルアーティストです。
ユーザーの爪写真を「編集」して、ネイルだけを大胆に変える【1案】を提案してください。

方針（遊び多め）：
- ベージュ単色に寄せない（色相の幅を広く）
- 透明感・ニュアンス・素材感（マット/ツヤ/ミラー風/オーロラ風）をMIX
- ストーンやラメは必須にしない（入れても控えめ）
- 現実のサロンで再現可能な範囲
- 画像は「元の手・肌色・光・背景・構図」を保持し、変えるのは爪だけ

次のJSONだけで出力：

{{
  "concept_jp": "お客様向けの短いコンセプト（日本語）",
  "detail_jp": "色・配置・質感・ポイント（日本語）",
  "edit_prompt_en": "English. Edit the uploaded photo. Keep same hand/skin/lighting/background/composition. Change ONLY nails. Playful, artistic, expressive, multi-color, salon-realistic. No text."
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

        edit_prompt = (idea.get("edit_prompt_en") or "").strip()
        if not edit_prompt:
            edit_prompt = (
                "Edit the uploaded photo. Keep the same hand, skin tone, lighting, background, and composition. "
                "Change ONLY the nail design. Make it playful, artistic, expressive, multi-color, salon-realistic. "
                "No text, no watermark."
            )

        # ===== 2) 画像編集（SDK 2.x は images.edit）=====
        image_data_url = None
        image_error = None

        try:
            img_res = client.images.edit(
                model="gpt-image-1",
                image=img_stream,
                prompt=edit_prompt,
                size="1024x1024",
            )

            # SDKはurl or b64_json のどちらかが来る
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
            "image_error": image_error,
            "debug_files_keys": incoming_file_keys,
            "debug_form_keys": incoming_form_keys
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
