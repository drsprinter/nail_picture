from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from io import BytesIO

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/api/tryon", methods=["POST", "OPTIONS"])
def tryon():
    if request.method == "OPTIONS":
        return "", 204

    try:
        if "nail_photo" not in request.files:
            return jsonify({"error": "nail_photo が送信されていません"}), 400

        photo_file = request.files["nail_photo"]
        photo_bytes = photo_file.read()

        age = request.form.get("age", "")
        lifestyle = request.form.get("lifestyle", "")
        nail_duration = request.form.get("nail_duration", "")
        avoid_colors = request.form.get("avoid_colors", "")
        purpose = request.form.getlist("purpose")
        vibe = request.form.getlist("vibe")

        # 1) ネイルプラン生成（テキスト）
        plan_system = (
            "あなたはプロのネイリストです。"
            "以下のお客様情報をもとに、シンプルで上品なジェルネイルの提案をしてください。"
            "ラメ・ストーン・派手柄は禁止。"
            "出力は次の2つのセクションに分けてください：\n"
            "[お客様向けネイルコンセプト]：感性に響く丁寧な説明、使用カラーやイメージなど。\n"
            "[サロン向け技術メモ]：プリジェル顔料を使ったカラー調合比率、使用カラー名、塗布順、ポイントなど。"
            "明るくしたい/暗くしたい場合の調合アレンジも添える。"
        )

        user_info = "\n".join([
            f"age: {age}",
            f"lifestyle: {lifestyle}",
            f"nail_duration: {nail_duration}",
            f"purpose: {', '.join(purpose) if purpose else ''}",
            f"vibe: {', '.join(vibe) if vibe else ''}",
            f"avoid: {avoid_colors}"
        ])

        plan_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": plan_system},
                {"role": "user", "content": user_info}
            ],
            temperature=0.7
        )
        plan = plan_resp.choices[0].message.content.strip()

        # 2) 画像編集用プロンプト生成（短く英語）
        img_prompt_sys = (
            "Write a short English prompt for an image editing model.\n"
            "Goal: modify ONLY the nail design in the provided photo.\n"
            "Keep the hand, skin tone, lighting, and background unchanged and photorealistic.\n"
            "Style: simple, elegant, glossy gel finish, no glitter, no rhinestones.\n"
            "Return ONLY the prompt text."
        )
        img_prompt_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": img_prompt_sys},
                {"role": "user", "content": plan}
            ],
            temperature=0.3
        )
        image_prompt = img_prompt_resp.choices[0].message.content.strip()

        # 3) 画像編集（ここが重要）
        #    - 古いSDKだと client.images.edits が存在しない → openai をアップデート必須
        if not hasattr(client, "images"):
            return jsonify({"error": "OpenAI SDK が古く images API が使えません。openai を更新してください。"}), 500

        # 一部SDKは `client.images.edit` / `client.images.edits` の差があるので分岐
        image_file_like = BytesIO(photo_bytes)
        image_file_like.name = "nail.png"

        if hasattr(client.images, "edits") and hasattr(client.images.edits, "create"):
            # ✅ edits.create がある場合
            edit_result = client.images.edits.create(
                model="gpt-image-1",
                image=image_file_like,
                prompt=image_prompt,
                size="1024x1024"
            )
        elif hasattr(client.images, "edit") and callable(getattr(client.images, "edit")):
            # ✅ edit があるSDK用（念のため）
            edit_result = client.images.edit(
                model="gpt-image-1",
                image=image_file_like,
                prompt=image_prompt,
                size="1024x1024"
            )
        else:
            return jsonify({
                "error": (
                    "この環境の OpenAI SDK では画像編集APIが使えません。"
                    "requirements.txt で openai>=1.40.0 に更新してデプロイしてください。"
                )
            }), 500

        # 返却（base64優先）
        image_base64 = None
        first = edit_result.data[0]
        if hasattr(first, "b64_json") and first.b64_json:
            image_base64 = first.b64_json
        elif hasattr(first, "url") and first.url:
            # URLしか返らない場合（環境次第）
            return jsonify({"plan": plan, "image_url": first.url})

        return jsonify({"plan": plan, "image_base64": image_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
