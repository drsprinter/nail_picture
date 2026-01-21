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
    """モデルが ```json ``` で返してもOKにしつつ JSON を抜く"""
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
            return jsonify({"error": "爪の画像（nail_image）が必要です"}), 400

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

        # ===== 1) プラン & 画像プロンプト（幅広く）=====
        idea_prompt = f"""
あなたは一流のネイルデザイナーです。
以下のお客様情報と爪の写真を参考に、【1案】のネイルデザインを提案してください。

重要：
- ベージュのワンカラーに寄りすぎない（幅広い配色・質感・アートの可能性を残す）
- ただし現実のネイルサロンで再現可能な範囲
- 手や指先のアップ写真として自然に成立するデザイン
- 装飾は過剰にせず、上品な範囲で「透明感・ニュアンス・素材感・コントラスト」を表現

必ず次のJSONだけで出力：

{{
  "concept": "お客様向け（情緒的でわかりやすく）",
  "design_description": "色・配置・質感・ポイントを具体的に",
  "image_prompt": "英語。Realistic photo. Close-up of a hand. Nails only. No text. Neutral background. Wide color expression allowed."
}}

お客様情報：
{user_text}
"""

        idea_res = client.chat.completions.create(
            # 速さ優先なら mini でもOK：gpt-4.1-mini
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": idea_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            temperature=0.85
        )

        idea_raw = idea_res.choices[0].message.content
        idea = safe_extract_json(idea_raw)

        plan_text = f"""【ネイルコンセプト】
{idea.get("concept","")}

【デザイン詳細】
{idea.get("design_description","")}
""".strip()

        # ===== 2) 画像生成（gpt-image-1）=====
        # 画像プロンプトは「手のアップ」「ネイルのみ」「露出系に見えない」方向に安全寄せ
        img_prompt = (idea.get("image_prompt") or "").strip()
        if not img_prompt:
            # 最低限の保険
            img_prompt = (
                "A realistic close-up photo of a hand with stylish gel nails, "
                "modern nuanced design, neutral background, soft natural lighting, no text."
            )

        image_data_url = None
        image_error = None

        try:
            img_res = client.images.generate(
                model="gpt-image-1",
                prompt=img_prompt,
                size="1024x1024"
            )

            # ✅ URLではなく b64_json を返すことが多い
            b64_json = getattr(img_res.data[0], "b64_json", None)
            url = getattr(img_res.data[0], "url", None)

            if b64_json:
                image_data_url = "data:image/png;base64," + b64_json
            elif url:
                # 環境によっては url が返る場合もあるので対応
                image_data_url = url
            else:
                image_error = "画像データがレスポンスに含まれていません（b64_json/url共に無し）"

        except Exception as e:
            # 画像だけ失敗しても plan は返す
            image_error = str(e)

        return jsonify({
            "plan": plan_text,
            # フロントは image_data_url を優先して読む
            "image_data_url": image_data_url,
            # デバッグ用（必要ならフロントで表示）
            "image_error": image_error,
            # 互換キー（古いフロント向け）
            "image_url": image_data_url if (image_data_url and image_data_url.startswith("http")) else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
