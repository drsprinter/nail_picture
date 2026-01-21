from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import re
import base64

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_text(s: str) -> str:
    if not s:
        return ""
    # 念のため：誤判定されやすい語を避ける
    s = re.sub(r"(nude|sexy|sensual|erotic|teen|young|schoolgirl)", "", s, flags=re.I)
    return s.strip()[:600]

def build_safe_image_prompt(style, palette, design):
    # “手”はOKだけど、より安全に「manicure photo」「fingers」寄りで表現
    return (
        "Photorealistic manicure photo, close-up of fingers with gel nails. "
        "Soft natural lighting, clean minimal background, professional nail salon photography. "
        "No text, no watermark, no logo. "
        "Natural skin tone (light beige). "
        f"Style: {clean_text(style)}. "
        f"Colors: {clean_text(', '.join(palette))}. "
        f"Design: {clean_text(design)}. "
        "Simple and elegant, no glitter, no rhinestones."
    )

def parse_multipart_form(req):
    data = {}
    for key in req.form.keys():
        vals = req.form.getlist(key)
        if len(vals) == 1:
            data[key] = vals[0]
        else:
            data[key] = vals
    return data

@app.route("/api/makeup", methods=["POST", "OPTIONS"])
def makeup():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = parse_multipart_form(request)

        nail_photo = request.files.get("nail_photo")
        if nail_photo is None:
            return jsonify({"status": "error", "error": "nail_photo が送られていません"}), 400

        # --- 入力整形 ---
        info_lines = []
        for k, v in data.items():
            if isinstance(v, list):
                info_lines.append(f"{k}: {', '.join(v)}")
            else:
                info_lines.append(f"{k}: {v}")
        user_info = "\n".join(info_lines)

        # ① 3方向の提案（JSON）
        system_ideas = (
            "あなたはプロのネイリストです。"
            "以下の情報から、方向性が明確に異なるネイルデザイン案を3つ提案してください。\n"
            "条件:\n"
            "- シンプル系（ラメ・ストーン無し）\n"
            "- 色味・印象・雰囲気が被らない\n"
            "- 30代に似合う\n\n"
            "出力は必ずJSONのみ。\n"
            "{"
            '"proposals": ['
            '{"id":1,"name":"","style":"","palette":[],"design":"","why":""},'
            '{"id":2,"name":"","style":"","palette":[],"design":"","why":""},'
            '{"id":3,"name":"","style":"","palette":[],"design":"","why":""}'
            "]}"
        )

        ideas_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_ideas},
                {"role": "user", "content": user_info}
            ],
            temperature=0.85
        )

        proposals = json.loads(ideas_res.choices[0].message.content)["proposals"]

        results = []
        for p in proposals:
            # ② 詳細プラン
            plan_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content":
                        "以下のネイル案について文章を作成してください。\n"
                        "[お客様向けネイルコンセプト]\n"
                        "[サロン向け技術メモ]（プリジェル調合比率・明るく/暗くの調整案も）"
                    },
                    {"role": "user", "content": f"{user_info}\n\n提案:\n{json.dumps(p, ensure_ascii=False)}"}
                ],
                temperature=0.7
            )
            plan_text = plan_res.choices[0].message.content.strip()

            image_data_url = None
            image_error = None

            # ③ 画像生成（b64 を data URL に変換）
            try:
                img_prompt = build_safe_image_prompt(
                    p.get("style", ""),
                    p.get("palette", []),
                    p.get("design", "")
                )

                img_res = client.images.generate(
                    model="gpt-image-1",
                    prompt=img_prompt,
                    size="1024x1024"
                    # response_formatはSDKや環境で不要な場合あり
                )

                # ✅ gpt-image-1 は b64_json のことが多い
                b64 = getattr(img_res.data[0], "b64_json", None)
                url = getattr(img_res.data[0], "url", None)

                if b64:
                    image_data_url = f"data:image/png;base64,{b64}"
                elif url:
                    image_data_url = url  # urlが来た環境でもOK
                else:
                    image_error = "画像レスポンスに url / b64_json のどちらもありませんでした"

            except Exception as e:
                image_error = str(e)

            results.append({
                "id": p.get("id"),
                "name": p.get("name"),
                "style": p.get("style"),
                "palette": p.get("palette"),
                "design": p.get("design"),
                "why": p.get("why"),
                "plan": plan_text,
                # ✅ フロントではこれを src に突っ込むだけで表示できる
                "image_data_url": image_data_url,
                # ✅ 失敗理由も見える（デバッグ用）
                "image_error": image_error
            })

        return jsonify({"status": "ok", "proposals": results})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
