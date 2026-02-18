from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import io
import json
import time
import math
import secrets

app = Flask(__name__)
CORS(app, origins=[
    "https://drsprinter.github.io",
    "https://drsprinter.github.io/nail_sample",
    "http://localhost:5500"
])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================================================
# 0) Utilities
# =========================================================

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
            vv = ", ".join([str(x) for x in v if str(x).strip()])
            if vv.strip():
                lines.append(f"{k}: {vv}")
        else:
            if str(v).strip():
                lines.append(f"{k}: {v}")
    return "\n".join(lines).strip()

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def entropy(p: list) -> float:
    e = 0.0
    for x in p:
        if x > 1e-12:
            e -= x * math.log(x)
    return e

def normalize(weights: list) -> list:
    s = sum(weights)
    if s <= 0:
        n = len(weights)
        return [1.0 / n for _ in range(n)]
    return [w / s for w in weights]

# =========================================================
# 1) Nailist Personas (A/B/C)
#    ★ここをヒヤリング結果で埋める（最重要）
# =========================================================
# A: 安全・上品・日常適合を最優先（保守派）
# B: 上品なトレンド編集（今っぽさ重視）
# C: ワンポイントで新鮮（アートディレクター）
NAILIST_PERSONAS = {
    "A": {
        "persona_id": "conservative_elegant",
        "display_name": "保守上品派",
        "tagline": "失敗しない上品さ。肌なじみと清潔感で手元を格上げ。",
        "voice": {
            "tone": "丁寧・安心感・控えめに背中を押す",
            "style_keywords": ["上品", "清潔感", "肌なじみ", "透明感", "ツヤ"],
            "avoid_phrases": ["ゴテゴテ", "盛る", "派手すぎ"]
        },
        "design_policy": {
            "signature_moves": [
                "透け感ベース＋微細ラメで清潔感",
                "先端にだけほんのりニュアンス",
                "ストーンは1粒レベルで上品"
            ],
            "taboo": [
                "avoid_colorsに触れる色",
                "強すぎる原色やネオン",
                "3Dや過度な盛り"
            ],
            "accent_rules": {
                "max_accent_nails": 1,
                "accent_types_preferred": ["micro_glitter", "subtle_stone", "thin_line", "french_tip"],
                "accent_intensity": 0.25
            }
        },
        "params": {
            "risk": 0.25,
            "daily_fit": 0.90,
            "trend": 0.40,
            "minimal": 0.80,
            "sparkle": 0.25,
            "vivid": 0.45,
            "safety": 0.90
        }
    },
    "B": {
        "persona_id": "trend_editor",
        "display_name": "上品トレンド派",
        "tagline": "盛らずに垢抜け。上品のまま今っぽく。",
        "voice": {
            "tone": "テンポよく・前向き・おしゃれ編集者",
            "style_keywords": ["トレンド", "うる艶", "抜け感", "ニュアンス", "洗練"],
            "avoid_phrases": ["古い", "無難すぎ", "手抜き"]
        },
        "design_policy": {
            "signature_moves": [
                "マグネット/うる艶系を控えめに",
                "今っぽいシアーカラーで透明感",
                "微細ラメを“線”や“先端”で効かせる"
            ],
            "taboo": [
                "avoid_colorsに触れる色",
                "色数が多すぎて散らかる",
                "トレンドの押し付け"
            ],
            "accent_rules": {
                "max_accent_nails": 2,
                "accent_types_preferred": ["magnet", "micro_glitter", "sheer_gloss", "one_nail_accent"],
                "accent_intensity": 0.40
            }
        },
        "params": {
            "risk": 0.55,
            "daily_fit": 0.70,
            "trend": 0.95,
            "minimal": 0.55,
            "sparkle": 0.40,
            "vivid": 0.60,
            "safety": 0.70
        }
    },
    "C": {
        "persona_id": "art_director",
        "display_name": "ワンポイントアート派",
        "tagline": "ワンポイントで新しい自分。上品な遊び心で差をつける。",
        "voice": {
            "tone": "クリエイティブ・具体的・提案が明快",
            "style_keywords": ["アクセント", "余白", "ライン", "配置", "個性"],
            "avoid_phrases": ["なんとなく", "適当", "盛り盛り"]
        },
        "design_policy": {
            "signature_moves": [
                "1〜2本だけ意図のあるアクセント配置",
                "先端/ライン/小さなアートで新鮮さ",
                "色は増やしすぎず“配置”で攻める"
            ],
            "taboo": [
                "avoid_colorsに触れる色",
                "全面アートで重くなる",
                "統一感のない柄の乱用"
            ],
            "accent_rules": {
                "max_accent_nails": 2,
                "accent_types_preferred": ["thin_line", "abstract_art", "french_tip", "one_nail_art", "subtle_stone"],
                "accent_intensity": 0.45
            }
        },
        "params": {
            "risk": 0.60,
            "daily_fit": 0.60,
            "trend": 0.65,
            "minimal": 0.55,
            "sparkle": 0.35,
            "vivid": 0.65,
            "safety": 0.65
        }
    }
}

def build_persona_candidate_prompt(candidate_id: str, persona: dict, user_text: str, selected_summary: dict) -> str:
    """
    ペルソナ注入版：候補1つだけ生成（A/B/Cを別々に呼ぶ）
    """
    return f"""
あなたはプロのネイリストです。以下の「ネイリストのペルソナ」に厳密に従って、
お客様に合うネイル提案を【1案】だけ作ってください。

【ネイリストのペルソナ（必ず従う）】
{json.dumps(persona, ensure_ascii=False)}

【絶対条件（Hard constraints）】
- お客様の選択項目（vibe / purpose / avoid_colors / nail_duration / age）を最優先に踏襲（これが最上位）
- 奇抜すぎない：新しさは80%程度（新しい自分を発見できるが上品で現実的）
- 日常へ馴染むかどうかは選択項目に応じて適切に調整（仕事寄りなら控えめ、イベント寄りなら少し遊ぶ）
- ベージュ単色に寄りすぎない（程よく鮮やかさ・血色・透明感を入れる）
- ストーン/ラメ/アートは回答に応じて適切に。華やかさは“ワンポイント”で上品に
- avoid_colors（自由入力）はNG/苦手が含まれる可能性があるため、踏まないこと

【この案の役割】
- id="{candidate_id}" の案として、ペルソナらしさが最も出る方向で提案する
- ただし上の絶対条件は必ず守る

【出力（JSONのみ）】
{{
  "id": "{candidate_id}",
  "plan_ja": "【ネイルコンセプト】...\\n【デザイン詳細】...",
  "style_hint": "内部用の短いメモ（ペルソナらしさ/狙い）"
}}

【お客様情報（そのまま）】
{user_text}

【お客様の選択項目サマリ（最優先）】
{json.dumps(selected_summary, ensure_ascii=False)}
""".strip()

# =========================================================
# 2) Type space (user's private preference type θ)
# =========================================================

TYPE_SPACE = [
    {"id": "T1", "name": "work_minimal",    "t": {"risk":0.15,"sparkle":0.15,"minimal":0.85,"trend":0.35,"daily":0.95,"vivid":0.30}},
    {"id": "T2", "name": "elegant_trend",   "t": {"risk":0.45,"sparkle":0.30,"minimal":0.65,"trend":0.80,"daily":0.70,"vivid":0.55}},
    {"id": "T3", "name": "feminine_glow",   "t": {"risk":0.35,"sparkle":0.45,"minimal":0.55,"trend":0.60,"daily":0.65,"vivid":0.60}},
    {"id": "T4", "name": "cool_mode",       "t": {"risk":0.55,"sparkle":0.25,"minimal":0.60,"trend":0.70,"daily":0.55,"vivid":0.55}},
    {"id": "T5", "name": "playful_event",   "t": {"risk":0.75,"sparkle":0.60,"minimal":0.30,"trend":0.75,"daily":0.35,"vivid":0.75}},
    {"id": "T6", "name": "classic_groomed", "t": {"risk":0.25,"sparkle":0.20,"minimal":0.75,"trend":0.40,"daily":0.85,"vivid":0.45}},
    {"id": "T7", "name": "art_accent",      "t": {"risk":0.65,"sparkle":0.45,"minimal":0.45,"trend":0.55,"daily":0.50,"vivid":0.70}},
    {"id": "T8", "name": "cute_pop",        "t": {"risk":0.55,"sparkle":0.55,"minimal":0.35,"trend":0.65,"daily":0.50,"vivid":0.80}},
]
TYPE_INDEX = {t["id"]: i for i, t in enumerate(TYPE_SPACE)}

def prior_from_selections(form: dict) -> list:
    w = [1.0 for _ in TYPE_SPACE]

    purpose = form.get("purpose", [])
    if isinstance(purpose, str): purpose = [purpose]
    vibe = form.get("vibe", [])
    if isinstance(vibe, str): vibe = [vibe]
    age = str(form.get("age", "") or "")

    def boost(type_id, amount):
        w[TYPE_INDEX[type_id]] *= amount

    if "仕事用" in purpose:
        boost("T1", 1.7); boost("T6", 1.5); boost("T2", 1.2); boost("T5", 0.7)
    if "イベント" in purpose:
        boost("T5", 1.6); boost("T7", 1.3); boost("T8", 1.3)
    if "気分転換" in purpose:
        boost("T3", 1.2); boost("T2", 1.2); boost("T7", 1.2)

    if "上品" in vibe:
        boost("T6", 1.4); boost("T2", 1.2); boost("T1", 1.2)
    if "可愛い" in vibe:
        boost("T8", 1.4); boost("T3", 1.2)
    if "クール" in vibe:
        boost("T4", 1.5); boost("T2", 1.1)
    if "トレンド感" in vibe:
        boost("T2", 1.6); boost("T4", 1.2); boost("T5", 1.1)

    if "10代" in age or "20代" in age:
        boost("T5", 1.1); boost("T8", 1.1)
    if "40代" in age or "50代" in age:
        boost("T6", 1.15); boost("T1", 1.1)

    return normalize(w)

QUESTIONS = {
    "challenge_level": {
        "text": "今日はどれくらい挑戦したい？",
        "options": [
            {"value":"10(ほぼいつも通り)","label":"ほぼいつも通り"},
            {"value":"30(ちょい変化)","label":"ちょい変化"},
            {"value":"60(いつもより攻めたい)","label":"いつもより攻めたい"},
            {"value":"80(新しい自分に寄せたい)","label":"新しい自分に寄せたい"},
        ]
    },
    "top_priority": {
        "text": "ネイルで一番大事なのは？",
        "options": [
            {"value":"仕事/生活で浮かない","label":"仕事/生活で浮かない"},
            {"value":"手元がきれいに見える","label":"手元がきれいに見える"},
            {"value":"気分が上がる","label":"気分が上がる"},
            {"value":"周りに褒められたい","label":"周りに褒められたい"},
            {"value":"写真映え","label":"写真映え"},
        ]
    },
    "accent_preference": {
        "text": "華やかさ（ワンポイント）はどんな感じが好き？",
        "options": [
            {"value":"近くで見ると分かるくらい（繊細）","label":"近くで見ると分かるくらい（繊細）"},
            {"value":"ぱっと見で可愛い（分かりやすい）","label":"ぱっと見で可愛い（分かりやすい）"},
            {"value":"指先だけ強め（フレンチ/先端）","label":"指先だけ強め（フレンチ/先端）"},
            {"value":"1〜2本だけ遊びたい（アクセント爪）","label":"1〜2本だけ遊びたい（アクセント爪）"},
        ]
    },
    "outfit_style": {
        "text": "普段の服のテイスト",
        "options": [
            {"value":"ベーシック/きれいめ","label":"ベーシック/きれいめ"},
            {"value":"カジュアル","label":"カジュアル"},
            {"value":"フェミニン","label":"フェミニン"},
            {"value":"モード/クール","label":"モード/クール"},
            {"value":"トレンド/韓国っぽ","label":"トレンド/韓国っぽ"},
            {"value":"日によってバラバラ","label":"日によってバラバラ"},
        ]
    }
}

def likelihood(question_id: str, answer_value: str, theta: dict, form: dict) -> float:
    t = theta["t"]
    if question_id == "challenge_level":
        target = 0.3
        if answer_value.startswith("10"): target = 0.15
        elif answer_value.startswith("30"): target = 0.35
        elif answer_value.startswith("60"): target = 0.65
        elif answer_value.startswith("80"): target = 0.80
        d = abs(t["risk"] - target)
        return 0.10 + 0.90 * math.exp(-4.0 * d)

    if question_id == "top_priority":
        if answer_value == "仕事/生活で浮かない":
            s = 0.15 + 0.85 * (0.7*t["daily"] + 0.3*t["minimal"])
        elif answer_value == "手元がきれいに見える":
            s = 0.15 + 0.85 * (0.6*t["minimal"] + 0.4*(1.0-abs(t["vivid"]-0.55)))
        elif answer_value == "気分が上がる":
            s = 0.15 + 0.85 * (0.5*t["vivid"] + 0.5*t["risk"])
        elif answer_value == "周りに褒められたい":
            s = 0.15 + 0.85 * (0.6*t["trend"] + 0.4*t["sparkle"])
        elif answer_value == "写真映え":
            s = 0.15 + 0.85 * (0.6*t["trend"] + 0.4*t["vivid"])
        else:
            s = 0.2
        return clamp01(s)

    if question_id == "accent_preference":
        if "繊細" in answer_value:
            s = 0.15 + 0.85 * (0.6*t["minimal"] + 0.4*(1.0-t["sparkle"]))
        elif "分かりやすい" in answer_value:
            s = 0.15 + 0.85 * (0.55*t["sparkle"] + 0.45*t["vivid"]))
        elif "先端" in answer_value:
            s = 0.15 + 0.85 * (0.5*t["minimal"] + 0.5*t["trend"]))
        elif "アクセント爪" in answer_value:
            s = 0.15 + 0.85 * (0.55*t["risk"] + 0.45*t["trend"]))
        else:
            s = 0.2
        return clamp01(s)

    if question_id == "outfit_style":
        if "きれいめ" in answer_value:
            s = 0.15 + 0.85 * (0.55*t["minimal"] + 0.45*t["daily"]))
        elif "カジュアル" in answer_value:
            s = 0.15 + 0.85 * (0.45*t["daily"] + 0.55*(1.0-abs(t["vivid"]-0.6))))
        elif "フェミニン" in answer_value:
            s = 0.15 + 0.85 * (0.55*t["vivid"] + 0.45*t["sparkle"]))
        elif "モード" in answer_value:
            s = 0.15 + 0.85 * (0.65*t["trend"] + 0.35*(1.0-t["sparkle"])))
        elif "韓国" in answer_value:
            s = 0.15 + 0.85 * (0.75*t["trend"] + 0.25*t["vivid"]))
        elif "バラバラ" in answer_value:
            s = 0.25 + 0.75 * (0.5*t["risk"] + 0.5*t["trend"]))
        else:
            s = 0.2
        return clamp01(s)

    return 0.2

def bayes_update(posterior: list, form: dict, question_id: str, answer_value: str) -> list:
    new_w = []
    for i, th in enumerate(TYPE_SPACE):
        new_w.append(posterior[i] * likelihood(question_id, answer_value, th, form))
    return normalize(new_w)

def choose_next_question(posterior: list, form: dict):
    unanswered = []
    for qid in QUESTIONS.keys():
        v = form.get(qid, "")
        if not str(v).strip():
            unanswered.append(qid)
    if not unanswered:
        return None

    H0 = entropy(posterior)
    best = None
    for qid in unanswered:
        opts = QUESTIONS[qid]["options"]

        p_ans = []
        post_by_ans = []
        for opt in opts:
            un = []
            for i, th in enumerate(TYPE_SPACE):
                un.append(posterior[i] * likelihood(qid, opt["value"], th, form))
            p = sum(un)
            p_ans.append(p)
            post_by_ans.append(normalize(un) if p <= 0 else [u/p for u in un])

        p_ans = normalize(p_ans)
        expected_H = sum(pa * entropy(post) for pa, post in zip(p_ans, post_by_ans))
        ig = H0 - expected_H
        if (best is None) or (ig > best["ig"]):
            best = {"qid": qid, "ig": ig}

    if best is None:
        return None

    q = QUESTIONS[best["qid"]]
    return {"id": best["qid"], "text": q["text"], "options": q["options"], "required": True}

# =========================================================
# 3) Candidate evaluation / selection (same as before)
# =========================================================

AXES = [
    "adherence_to_selections",
    "wearability_daily_fit",
    "novelty_target_80",
    "colorfulness_not_beige_only",
    "accent_fit_one_point"
]
TYPE_WEIGHTS = {
    "T1": {"adherence_to_selections":0.30,"wearability_daily_fit":0.30,"novelty_target_80":0.14,"colorfulness_not_beige_only":0.14,"accent_fit_one_point":0.12},
    "T2": {"adherence_to_selections":0.24,"wearability_daily_fit":0.20,"novelty_target_80":0.22,"colorfulness_not_beige_only":0.18,"accent_fit_one_point":0.16},
    "T3": {"adherence_to_selections":0.22,"wearability_daily_fit":0.18,"novelty_target_80":0.20,"colorfulness_not_beige_only":0.20,"accent_fit_one_point":0.20},
    "T4": {"adherence_to_selections":0.22,"wearability_daily_fit":0.18,"novelty_target_80":0.22,"colorfulness_not_beige_only":0.20,"accent_fit_one_point":0.18},
    "T5": {"adherence_to_selections":0.18,"wearability_daily_fit":0.12,"novelty_target_80":0.26,"colorfulness_not_beige_only":0.22,"accent_fit_one_point":0.22},
    "T6": {"adherence_to_selections":0.28,"wearability_daily_fit":0.28,"novelty_target_80":0.16,"colorfulness_not_beige_only":0.16,"accent_fit_one_point":0.12},
    "T7": {"adherence_to_selections":0.20,"wearability_daily_fit":0.18,"novelty_target_80":0.24,"colorfulness_not_beige_only":0.18,"accent_fit_one_point":0.20},
    "T8": {"adherence_to_selections":0.18,"wearability_daily_fit":0.16,"novelty_target_80":0.22,"colorfulness_not_beige_only":0.22,"accent_fit_one_point":0.22},
}

def expected_utility(scores: dict, posterior: list) -> float:
    u = 0.0
    for p, th in zip(posterior, TYPE_SPACE):
        w = TYPE_WEIGHTS[th["id"]]
        su = 0.0
        for ax in AXES:
            su += w.get(ax, 0.0) * (float(scores.get(ax, 0.0)) / 100.0)
        u += p * su
    return u

def pick_by_expected_utility(candidates: list, eval_payload: dict, posterior: list) -> dict:
    results = eval_payload.get("results") or []
    by_id = {r.get("id"): r for r in results if r.get("id")}
    best = None
    for c in candidates:
        cid = c.get("id")
        r = by_id.get(cid, {})
        scores = (r.get("scores") or {})
        eu = expected_utility(scores, posterior)
        adherence = float(scores.get("adherence_to_selections", 0) or 0)
        tup = (eu, adherence)
        if (best is None) or (tup > best["tup"]):
            best = {"candidate": c, "eval": r, "eu": eu, "tup": tup}
    return best or {"candidate": candidates[0] if candidates else {}, "eval": {}, "eu": 0.0, "tup": (0.0, 0.0)}

# =========================================================
# 4) Sessions
# =========================================================

SESSIONS = {}
SESSION_TTL_SEC = 10 * 60

def cleanup_sessions():
    now = time.time()
    dead = [k for k, v in SESSIONS.items() if now - v.get("created", now) > SESSION_TTL_SEC]
    for k in dead:
        SESSIONS.pop(k, None)

# =========================================================
# 5) Routes
# =========================================================

@app.route("/api/game/start", methods=["POST", "OPTIONS"])
def game_start():
    if request.method == "OPTIONS":
        return "", 204

    try:
        cleanup_sessions()

        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "爪の写真が必要です（image が見つかりません）"}), 400
        img_bytes = image_file.read()
        if not img_bytes:
            return jsonify({"error": "爪の写真が空でした（ファイルサイズ0の可能性）"}), 400

        form = form_to_dict(request.form)

        post = prior_from_selections(form)
        for qid in QUESTIONS.keys():
            ans = str(form.get(qid, "") or "").strip()
            if ans:
                post = bayes_update(post, form, qid, ans)

        next_q = choose_next_question(post, form)

        # ln(8)=2.079 / threshold for "still uncertain"
        if next_q is not None and entropy(post) > 1.15:
            token = secrets.token_urlsafe(16)
            SESSIONS[token] = {"created": time.time(), "img_bytes": img_bytes, "form": form, "posterior": post}
            return jsonify({"status":"need_more","token":token,"question":next_q})

        return finalize_with_posterior(img_bytes, form, post)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/game/answer", methods=["POST", "OPTIONS"])
def game_answer():
    if request.method == "OPTIONS":
        return "", 204

    try:
        cleanup_sessions()

        token = str(request.form.get("token", "") or "").strip()
        qid = str(request.form.get("question_id", "") or "").strip()
        ans = str(request.form.get("answer", "") or "").strip()

        if not token or token not in SESSIONS:
            return jsonify({"error": "セッションが見つかりません。最初からやり直してください。"}), 400
        if qid not in QUESTIONS:
            return jsonify({"error": "不明な質問です。"}), 400
        if not ans:
            return jsonify({"error": "回答が空です。"}), 400

        sess = SESSIONS.pop(token)
        img_bytes = sess["img_bytes"]
        form = sess["form"]
        post = sess["posterior"]

        form[qid] = ans
        post2 = bayes_update(post, form, qid, ans)

        return finalize_with_posterior(img_bytes, form, post2)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# 6) Main finalize
# =========================================================

def finalize_with_posterior(img_bytes: bytes, form: dict, posterior: list):
    user_text = build_user_text(form)

    selected_summary = {
        "age": form.get("age", ""),
        "nail_duration": form.get("nail_duration", ""),
        "purpose": form.get("purpose", []),
        "vibe": form.get("vibe", []),
        "avoid_colors": form.get("avoid_colors", ""),
        "challenge_level": form.get("challenge_level", ""),
        "outfit_style": form.get("outfit_style", ""),
        "top_priority": form.get("top_priority", ""),
        "accent_preference": form.get("accent_preference", "")
    }

    # -----------------------------------------------------
    # (1) Generate 3 candidates separately using personas
    # -----------------------------------------------------
    candidates = []
    for cid in ["A", "B", "C"]:
        persona = NAILIST_PERSONAS.get(cid, {})
        prompt = build_persona_candidate_prompt(cid, persona, user_text, selected_summary)

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたはトップネイルアーティストです。必ずJSONのみを返してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.65
        )

        try:
            payload = safe_extract_json(res.choices[0].message.content)
            # minimal validation
            if payload.get("id") != cid:
                payload["id"] = cid
            if not payload.get("plan_ja"):
                payload["plan_ja"] = "【ネイルコンセプト】\n（生成に失敗しました）\n【デザイン詳細】\n（もう一度お試しください）"
            if "style_hint" not in payload:
                payload["style_hint"] = f"persona:{persona.get('persona_id','unknown')}"
            candidates.append(payload)
        except Exception:
            candidates.append({
                "id": cid,
                "plan_ja": "【ネイルコンセプト】\n（プラン生成に失敗しました）\n【デザイン詳細】\n（もう一度お試しください）",
                "style_hint": f"fallback persona:{persona.get('persona_id','unknown')}"
            })

    # -----------------------------------------------------
    # (2) Evaluate candidates (same as before)
    # -----------------------------------------------------
    eval_prompt = f"""
あなたはネイル提案の品質評価者です。
下の「お客様の選択項目」と「候補3案」を読み、各案を0〜100点で採点してください。

採点軸（0〜100）：
- adherence_to_selections
- wearability_daily_fit
- novelty_target_80
- colorfulness_not_beige_only
- accent_fit_one_point

重要ルール：
- avoid_colors（自由入力）は“NG/苦手”が含まれる可能性があります。明確なNGを踏んでいる場合は該当軸を大きく減点。
- 追加項目（challenge_level / outfit_style / top_priority / accent_preference）があれば、整合している案を加点（ただし選択項目より優先しない）。

出力は【JSONのみ】：
{{
  "results": [
    {{"id":"A","scores":{{"adherence_to_selections":0,"wearability_daily_fit":0,"novelty_target_80":0,"colorfulness_not_beige_only":0,"accent_fit_one_point":0}},"notes":"..."}},
    {{"id":"B","scores":{{"adherence_to_selections":0,"wearability_daily_fit":0,"novelty_target_80":0,"colorfulness_not_beige_only":0,"accent_fit_one_point":0}},"notes":"..."}},
    {{"id":"C","scores":{{"adherence_to_selections":0,"wearability_daily_fit":0,"novelty_target_80":0,"colorfulness_not_beige_only":0,"accent_fit_one_point":0}},"notes":"..."}}
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

    picked = pick_by_expected_utility(candidates, eval_payload, posterior)
    plan_text = (picked.get("candidate") or {}).get("plan_ja") or candidates[0].get("plan_ja", "")

    # -----------------------------------------------------
    # (3) Build English image-edit prompt (same as before)
    # -----------------------------------------------------
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
{{"edit_prompt_en":"..."}}

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

    # -----------------------------------------------------
    # (4) Image edit (same as before)
    # -----------------------------------------------------
    image_data_url = None
    image_error = None
    try:
        img_stream = io.BytesIO(img_bytes)
        img_stream.name = "nail.jpg"
        img_res = client.images.edit(
            model="gpt-image-1",
            image=img_stream,
            prompt=edit_prompt,
            size="1024x1024",
            n=1
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

    top = sorted(
        [{"type": th["id"], "name": th["name"], "p": posterior[i]} for i, th in enumerate(TYPE_SPACE)],
        key=lambda x: x["p"],
        reverse=True
    )[:3]

    return jsonify({
        "plan": plan_text,
        "image_data_url": image_data_url,
        "image_error": image_error,
        "debug": {
            "posterior_top3": top,
            "picked_expected_utility": picked.get("eu"),
            "picked_id": (picked.get("candidate") or {}).get("id"),
            "candidates_debug": candidates
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
