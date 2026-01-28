from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import io
import json
import base64

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def safe_extract_json(text: str) -> dict:
    """Chatの出力からJSONだけを安全に取り出す（コードブロックや前後テキストが混ざってもOK）"""
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

def form_to_dict(req_form) -> dict:
    data = {}
    for k in req_form.keys():
        vals = req_form.getlist(k)
        data[k] = vals if len(vals) != 1 else vals[0]
    return data

def build_user_text(form_data: dict) -> str:
    lines = []
    for k, v in form_data.items():
        if isinstance(v, list):
            lines.append(f"{k}: {', '.join([str(x) for x in v if str(x).strip()])}".strip())
        else:
            lines.append(f"{k}: {v}".strip())
    return "\n".join([x for x in lines if x.strip()]).strip()

def minimax_pick(candidates: list, eval_payload: dict) -> dict:
    """各候補のスコアから minimax で1つ選ぶ"""
    # 期待する形式:
    # eval_payload = { "results": [ { "id": "A", "scores": {..}, "notes": "" }, ... ] }
    results = eval_payload.get("results") or []
    by_id = {r.get("id"): r for r in results if r.get("id")}

    best = None
    for c in candidates:
        cid = c.get("id")
        r = by_id.get(cid, {})
        scores = (r.get("scores") or {})
        # minimax: 重要な軸の「最低点」を最大化
        key_axes = [
            "adherence_to_selections",
            "wearability_daily_fit",
            "novelty_target_80",
            "colorfulness_not_beige_only",
            "accent_fit_one_point"
        ]
        vals = [float(scores.get(k, 0)) for k in key_axes]
        worst = min(vals) if vals else 0.0
        total = sum(vals)
        # tie-breaker: worst -> total -> adherence
        adherence = float(scores.get("adherence_to_selections", 0) or 0)
        score_tuple = (worst, total, adherence)

        if (best is None) or (score_tuple > best["score_tuple"]):
            best = {"candidate": c, "score_tuple": score_tuple, "eval": r, "worst": worst, "total": total}

    return best or {"candidate": candidates[0] if candidates else {}, "eval": {}, "worst": 0, "total": 0, "score_tuple": (0,0,0)}

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

        # gpt-image-1へ渡す base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # ===== 2) フォーム情報 =====
        form_data = form_to_dict(request.form)
        user_text = build_user_text(form_data)

        # 重要キー（強い制約として扱う）
        selected_summary = {
            "age": form_data.get("age", ""),
            "nail_duration": form_data.get("nail_duration", ""),
            "purpose": form_data.get("purpose", []),
            "vibe": form_data.get("vibe", []),
            "avoid_colors": form_data.get("avoid_colors", ""),
            # 追加項目
            "challenge_level": form_data.get("challenge_level", ""),
            "outfit_style": form_data.get("outfit_style", ""),
            "top_priority": form_data.get("top_priority", ""),
            "accent_preference": form_data.get("accent_preference", "")
        }

        # ===== 3) 候補を3案生成（内部で幅出し）=====
        candidates_prompt = f"""
あなたはプロのネイリストです。
以下の「お客様の選択項目」を最優先に踏襲しつつ、ネイル提案を【3案】作ってください。
この3案は、方向性が被らないように“自然な幅”をつけてください（奇抜はNG）。

絶対条件（Hard constraints）：
- お客様の選択項目（vibe / purpose / avoid_colors / nail_duration / age）を最優先に踏襲
- 奇抜すぎない：新しさは80%程度（新しい自分を発見できるが上品で現実的）
- 日常へ馴染むかどうかは選択項目に応じて適切に調整（仕事寄りなら控えめ、イベント寄りなら少し遊ぶ）
- ベージュ単色に寄りすぎない（程よく鮮やかさ・血色・透明感を入れる）
- ストーン/ラメ/アートは回答に応じて適切に。華やかさは“ワンポイント”で上品に

3案の役割（必ず守る）：
- A：一番外さない（上品・日常適合が高い）
- B：一番“今っぽい”/トレンド寄り（ただし上品・現実的）
- C：アクセントの置き方が新鮮（1〜2本 or 先端など、ワンポイントで攻める）

出力は【JSONのみ】（必ず有効なJSON。説明文なし）：
{{
  "candidates": [
    {{
      "id": "A",
      "plan_ja": "【ネイルコンセプト】...\n【デザイン詳細】...",
      "style_hint": "一言メモ（内部用）"
    }},
    {{
      "id": "B",
      "plan_ja": "...",
      "style_hint": "..."
    }},
    {{
      "id": "C",
      "plan_ja": "...",
      "style_hint": "..."
    }}
  ]
}}

お客様情報（そのまま）：
{user_text}

お客様の選択項目サマリ（最優先）：
{json.dumps(selected_summary, ensure_ascii=False)}
""".strip()

        cand_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたはトップネイルアーティストです。"},
                {"role": "user", "content": candidates_prompt}
            ],
            temperature=0.65
        )
        cand_payload = safe_extract_json(cand_res.choices[0].message.content)
        candidates = cand_payload.get("candidates") or []

        # フォールバック（万一）
        if not candidates:
            candidates = [{
                "id": "A",
                "plan_ja": "【ネイルコンセプト】\n（プラン生成に失敗しました）\n【デザイン詳細】\n（もう一度お試しください）",
                "style_hint": "fallback"
            }]

        # ===== 4) minimax評価（最悪ケースを避けて1案選抜）=====
        eval_prompt = f"""
あなたはネイル提案の品質評価者です。
下の「お客様の選択項目」と「候補3案」を読み、各案を0〜100点で採点してください。
目的は minimax（最悪ケース回避）です。どの軸でも致命的に外さない案が高評価です。

採点軸（0〜100）：
- adherence_to_selections: 選択項目（vibe/purpose/avoid_colors/nail_duration/age）への忠実さ
- wearability_daily_fit: 目的に応じた日常適合（仕事なら浮かない、イベントなら程よく映える等）
- novelty_target_80: “80%新しさ”のちょうど良さ（奇抜すぎない・でも新しい）
- colorfulness_not_beige_only: ベージュ単色に寄りすぎず、程よい鮮やかさ/血色/透明感がある
- accent_fit_one_point: ストーン/ラメ/アートの使い方が回答に合い、ワンポイントで上品

重要ルール：
- avoid_colors（自由入力）は“NG/苦手”が含まれる可能性があります。明確なNGを踏んでいる場合は該当軸を大きく減点してください。
- 追加項目（challenge_level / outfit_style / top_priority / accent_preference）があれば、整合している案を加点してください（ただし選択項目より優先しない）。

出力は【JSONのみ】：
{{
  "results": [
    {{
      "id": "A",
      "scores": {{
        "adherence_to_selections": 0,
        "wearability_daily_fit": 0,
        "novelty_target_80": 0,
        "colorfulness_not_beige_only": 0,
        "accent_fit_one_point": 0
      }},
      "notes": "短い根拠（内部用）"
    }}
  ]
}}

お客様の選択項目サマリ：
{json.dumps(selected_summary, ensure_ascii=False)}

候補：
{json.dumps(candidates, ensure_ascii=False)}
""".strip()

        eval_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Return JSON only."},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.2
        )
        eval_payload = safe_extract_json(eval_res.choices[0].message.content)

        picked = minimax_pick(candidates, eval_payload)
        plan_text = (picked.get("candidate") or {}).get("plan_ja") or ""
        if not plan_text:
            # 最低限のフォールバック
            plan_text = candidates[0].get("plan_ja", "")

        # ===== 5) 画像編集プロンプト（英語）をJSONで作る =====
        spec_prompt = f"""
You are a top nail artist who writes image-edit prompts.

Create an English prompt to edit the uploaded photo.

Hard rules (must follow):
- Keep the same hand, skin tone, lighting, background, and composition.
- Edit ONLY the nails. Do NOT change fingers, skin, jewelry, or background.
- Strictly follow the customer's selected options first (vibe / purpose / avoid_colors / nail_duration / age).
- Not overly eccentric: aim for around 80% freshness—help the customer discover a new side of themselves while keeping it wearable.
- Adjust how well it blends into daily life depending on the selected options (work/daily = subtle; event = a bit playful but still elegant).
- Do not lean too heavily toward a plain beige monochrome; make it moderately vivid and lively.
- Use stones, glitter, and nail art appropriately based on the customer's answers, with only ONE tasteful accent point.
- No text, no watermark, no logos.

Return ONLY valid JSON:
{{
  "edit_prompt_en": "..."
}}

Customer preferences:
{user_text}

Chosen nail plan (Japanese, for reference only):
{plan_text}
""".strip()

        spec_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write prompts for image editing. Return JSON only."},
                {"role": "user", "content": spec_prompt}
            ],
            temperature=0.25
        )
        spec = safe_extract_json(spec_res.choices[0].message.content)
        edit_prompt = (spec.get("edit_prompt_en") or "").strip()

        if not edit_prompt:
            edit_prompt = (
                "Keep the same hand, skin tone, lighting, background, and composition. "
                "Edit ONLY the nails (do not change fingers/skin/jewelry/background). "
                "Follow the customer's selected options first. Aim for ~80% freshness while staying wearable and elegant. "
                "Avoid a plain beige-only look; keep it moderately vivid. "
                "If adding sparkle/art, keep it as ONE tasteful accent point. "
                "No text, no watermark."
            )

        # ===== 6) 画像編集（gpt-image-1 / images.generate）=====
        image_data_url = None
        image_error = None

        try:
            img_res = client.images.generate(
                model="gpt-image-1",
                prompt=edit_prompt,
                image=img_b64,
                size="1024x1024"
            )

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
