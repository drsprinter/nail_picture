from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_list_from_form(key: str):
    # checkbox用
    return request.form.getlist(key) if request.form.getlist(key) else []

def safe_join(lst):
    return ", ".join([x for x in lst if x])

@app.route("/api/proposals", methods=["POST", "OPTIONS"])
def proposals():
    if request.method == "OPTIONS":
        return "", 204

    try:
        # 1) 画像必須チェック
        if "nail_photo" not in request.files:
            return jsonify({"error": "nail_photo が送信されていません（必須）"}), 400

        photo = request.files["nail_photo"]
        photo_bytes = photo.read()

        if not photo_bytes:
            return jsonify({"error": "nail_photo が空です（撮影/選択をやり直してください）"}), 400

        # 2) フォーム項目（multipartは request.form）
        age = request.form.get("age", "")
        lifestyle = request.form.get("lifestyle", "")
        nail_duration = request.form.get("nail_duration", "")
        avoid_colors = request.form.get("avoid_colors", "")

        purpose = normalize_list_from_form("purpose")
        vibe = normalize_list_from_form("vibe")

        user_info = "\n".join([
            f"年齢層: {age}",
            f"職業/ライフスタイル: {lifestyle}",
            f"他サロンでの持ち: {nail_duration}",
            f"目的: {safe_join(purpose)}",
            f"雰囲気: {safe_join(vibe)}",
            f"避けたい: {avoid_colors}",
            "条件: シンプル系、ラメ無し、ストーン無し、派手柄無し"
        ])

        # 3) まず「方向性の違う3案」をJSONで作る
        system_ideas = (
            "あなたはプロのネイリストです。"
            "以下のお客様情報から、方向性が明確に異なるネイル提案を3種類作ってください。\n"
            "必須条件：\n"
            "- シンプル系（ラメ・ストーン無し、派手柄無し）\n"
            "- 3案は『色味・雰囲気・印象』が被らない\n"
            "- 各案に、短い提案名、方向性（王道/トレンド/アクセント）、色、デザイン、似合う理由\n\n"
            "出力は必ずJSONのみ。説明文は禁止。\n"
            "JSONスキーマ：\n"
            "{\n"
            '  "proposals": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "name": "提案名（短く）",\n'
            '      "style_axis": "王道/トレンド/アクセント のどれか",\n'
            '      "palette": ["色1","色2","色3"],\n'
            '      "keywords": ["雰囲気","質感","印象"],\n'
            '      "design": "デザイン説明（例：極細フレンチ、ワンカラー、シアー、グラデ等）",\n'
            '      "why_fit": "なぜこのお客様に合うか（1-2文）"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        ideas_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_ideas},
                {"role": "user", "content": user_info}
            ],
            temperature=0.85
        )

        raw = ideas_resp.choices[0].message.content.strip()

        try:
            ideas_json = json.loads(raw)
        except Exception:
            # JSON崩れ救済
            fix_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "必ずJSONだけを返してください。余計な文字は禁止。"},
                    {"role": "user", "content": raw}
                ],
                temperature=0.0
            )
            ideas_json = json.loads(fix_resp.choices[0].message.content.strip())

        proposals_list = (ideas_json.get("proposals") or [])[:3]

        # 4) 各案ごとに「プラン文章」＋「画像プロンプト」＋「サンプル画像生成」
        system_plan = (
            "あなたはプロのネイリストです。"
            "以下の『お客様情報』と『デザイン案』を元に、文章を作ってください。\n"
            "必須条件：ラメ無し、ストーン無し、派手柄無し。\n"
            "出力は次の2セクション：\n"
            "[お客様向けネイルコンセプト]：丁寧でワクワクする説明。色・質感・印象。\n"
            "[サロン向け技術メモ]：プリジェル顔料の調合比率（例：顔料A:顔料B:クリア=1:1:8など）"
            "＋塗布順（ベース/カラー/トップ）＋明るく/暗くする調整案も書く。"
        )

        system_img_prompt = (
            "Write a short English prompt for generating a photorealistic nail sample image.\n"
            "Constraints:\n"
            "- Simple and elegant gel nails\n"
            "- No glitter, no rhinestones, no heavy patterns\n"
            "- Close-up of a hand, clean minimal background\n"
            "- Japanese average natural beige skin tone\n"
            "- No text in the image\n"
            "Return ONLY the prompt."
        )

        out = []
        for item in proposals_list:
            design_block = "\n".join([
                f"提案名: {item.get('name','')}",
                f"方向性: {item.get('style_axis','')}",
                f"カラーパレット: {safe_join(item.get('palette',[]))}",
                f"キーワード: {safe_join(item.get('keywords',[]))}",
                f"デザイン: {item.get('design','')}",
                f"似合う理由: {item.get('why_fit','')}",
            ])

            plan_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_plan},
                    {"role": "user", "content": "【お客様情報】\n" + user_info + "\n\n【デザイン案】\n" + design_block}
                ],
                temperature=0.7
            )
            plan_text = plan_resp.choices[0].message.content.strip()

            img_prompt_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_img_prompt},
                    {"role": "user", "content": design_block}
                ],
                temperature=0.4
            )
            img_prompt = img_prompt_resp.choices[0].message.content.strip()

            # サンプル画像生成（Verify不要）
            img = client.images.generate(
                model="dall-e-3",
                prompt=img_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = img.data[0].url

            out.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "style_axis": item.get("style_axis"),
                "palette": item.get("palette", []),
                "keywords": item.get("keywords", []),
                "design": item.get("design"),
                "why_fit": item.get("why_fit"),
                "plan": plan_text,
                "image_prompt": img_prompt,
                "image_url": image_url,
            })

        # NOTE:
        # photo_bytes は将来「写真試着（edit）」に使います（Verify後）
        # 現状は必須入力としてUXを担保しつつ、提案精度に活かす段階です。

        return jsonify({
            "status": "ok",
            "proposals": out
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
