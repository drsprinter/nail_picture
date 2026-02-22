from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import io
import json
import time
import math
import secrets
import re

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

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

# =========================================================
# 1) Persona registry (readable by persona_id)
# =========================================================

DEFAULT_PERSONA_ID = "nailist_01"

PERSONA_REGISTRY = {
    "nailist_01": {
        "persona_id": "nailist_01",
        "display_name": "そのちゃん",
        "tagline": "お友達",
        "voice": {
            "tone": "柔らかい。男前感もあるさっぱりさ。",
            "style_keywords": ["上品", "透明感", "うる艶", "抜け感"],
            "avoid_phrases": ["ギャルすぎ", "盛る", "ゴテゴテ"]
        },
        "design_policy": {
            "must_keep": [
                "Keep customer's selected options first: vibe, purpose, avoid_colors, nail_duration, age",
                "Not overly eccentric; aim for ~80% freshness (discover a new side but wearable)",
                "Adjust daily-fit depending on purpose (work/daily=subtle, event=slightly playful but elegant)",
                "Do not lean too heavily toward plain beige monochrome; keep it moderately vivid and lively",
                "Use stones/glitter/art appropriately with ONE tasteful accent point"
            ],
            "signature_moves": [
                "Sheer / milky base with 'churun' glossy finish to make hands look clean and beautiful",
                "Gradient or jelly-like translucency (soft ombre) to add depth without looking heavy",
                "Modern French variations: thin French, diagonal French, glitter French, color French (kept elegant)",
                "Micro-glitter / aurora / subtle magnetic shine used softly (never too flashy)",
                "Fine mirror/metallic line accents to sharpen the look while keeping it minimal",
                "One-nail (or max two nails) accent: bijou/stone, small charm, or delicate art as a focal point",
                "When using motifs/characters, limit color count and keep the overall palette cohesive and clean"
            ],
            "taboo": [
                "Overloaded decoration on all nails (too many stones, heavy glitter, 3D everywhere)",
                "Too many colors with no cohesion (scattered look)",
                "Neon / overly saturated primary colors used as full coverage without balance",
                "Flat heavy matte finish across all nails (prefers glossy / sheer look)",
                "Solid black full set without design intent (black is fine as an accent/art element)"
            ],
            "accent_rules": {
                "max_accent_nails": 2,
                "accent_types_preferred": [
                    "micro_glitter",
                    "aurora",
                    "magnet",
                    "mirror_line",
                    "thin_french",
                    "subtle_stone",
                    "bijou_one_nail",
                    "small_charm",
                    "minimal_art",
                    "hand drawing art"
                ],
                "accent_intensity": 0.40,
                "placement_notes": "Either 'one focal point' (bijou on 1 nail) OR 'soft distributed shine' (micro glitter/aurora on multiple nails), but avoid half-hearted scattered accents."
            }
        },
        "params": {
            "risk": 0.6,
            "daily_fit": 0.70,
            "trend": 0.70,
            "minimal": 0.7,
            "sparkle": 0.40,
            "vivid": 0.4,
            "safety": 1.0
        },
        "color_direction": {
            "base_palette_notes": "Milky white, sheer beige, greige, dusty pink, soft mauve, light gray, pale blue. Prioritize translucency and hand-brightening tones.",
            "allowed_accents_notes": "Silver/gold mirror lines, aurora/iridescent shine, subtle magnetic shimmer. Navy/blue as a clean 'tightening' color. Burgundy/red for seasonal/event moods (kept elegant).",
            "avoid_palette_notes": "Neon full coverage, overly saturated primary colors without balance, muddy/dark heavy full-coverage sets that lose translucency."
        },
        "format_constraints": {
            "output_language": "ja",
            "plan_format": [
                "【ネイルコンセプト】...",
                "【デザイン詳細】..."
            ],
            "length_hint": "全体で250〜450文字目安"
        }
    }
}

def get_persona_from_form(form: dict) -> dict:
    pid = str(form.get("persona_id", "") or "").strip()
    if not pid:
        pid = DEFAULT_PERSONA_ID
    return PERSONA_REGISTRY.get(pid, PERSONA_REGISTRY[DEFAULT_PERSONA_ID])

# =========================================================
# 1.5) Free input -> spec (Lv2)
# =========================================================

def quick_specificity_heuristic(free_text: str) -> int:
    """
    LLMが落ちた時の雑な具体度推定（0-100）
    - 文字数 + 記号/色/技法キーワードで加点
    """
    t = (free_text or "").strip()
    if not t:
        return 0
    score = 0
    # length
    score += min(60, len(t))
    # keywords (rough)
    keywords = ["フレンチ", "グラデ", "マグネット", "ミラー", "オーロラ", "ちゅるん", "シアー",
                "ストーン", "ラメ", "ニュアンス", "チェック", "ハート", "リボン", "キャラ",
                "ブルー", "ネイビー", "ピンク", "赤", "黒", "白", "グレー", "ベージュ", "ブラウン", "グリーン"]
    score += 5 * sum(1 for k in keywords if k in t)
    # "NG" patterns
    if "NG" in t or "苦手" in t or "避け" in t or "なし" in t:
        score += 10
    return max(0, min(100, score))

def extract_free_spec(free_text: str, selected_summary: dict) -> dict:
    """
    avoid_colors(自由入力)を 'spec' に変換して、強制力を持たせる
    spec = {specificity:0-100, must:[], must_not:[], soft:[], keywords:[], summary:"..."}
    """
    free_text = (free_text or "").strip()
    if not free_text:
        return {
            "specificity": 0,
            "must": [],
            "must_not": [],
            "soft": [],
            "keywords": [],
            "summary": ""
        }

    prompt = f"""
You are an expert nail concierge.

Convert the customer's free-text request into a structured spec.
This free-text field may contain:
- Desired design ideas (must)
- Things they want to avoid (must_not)
- Mood/finish preferences (soft)
- Colors / techniques / motifs keywords (keywords)

Important:
- This spec MUST be honored strongly when specificity is high.
- Respect avoid_colors-as-free-text nature: it may include "NG" or "avoid" items.
- Do not invent new preferences not implied by the text.
- Return ONLY valid JSON.

Return JSON format:
{{
  "specificity": 0-100,
  "must": ["..."],
  "must_not": ["..."],
  "soft": ["..."],
  "keywords": ["..."],
  "summary": "one short Japanese summary"
}}

Customer free-text:
{free_text}

Customer selection summary (for context only):
{json.dumps(selected_summary, ensure_ascii=False)}
""".strip()

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        spec = safe_extract_json(res.choices[0].message.content)

        # sanitize
        spec_out = {
            "specificity": max(0, min(100, safe_int(spec.get("specificity", 0), 0))),
            "must": [str(x).strip() for x in (spec.get("must") or []) if str(x).strip()],
            "must_not": [str(x).strip() for x in (spec.get("must_not") or []) if str(x).strip()],
            "soft": [str(x).strip() for x in (spec.get("soft") or []) if str(x).strip()],
            "keywords": [str(x).strip() for x in (spec.get("keywords") or []) if str(x).strip()],
            "summary": str(spec.get("summary", "") or "").strip()
        }

        # fallback if somehow empty
        if spec_out["specificity"] == 0:
            spec_out["specificity"] = quick_specificity_heuristic(free_text)

        return spec_out

    except Exception:
        # fallback heuristic
        return {
            "specificity": quick_specificity_heuristic(free_text),
            "must": [],
            "must_not": [],
            "soft": [],
            "keywords": [],
            "summary": ""
        }

def build_persona_candidate_prompt(candidate_id: str, persona: dict, user_text: str, selected_summary: dict, free_spec: dict) -> str:
    """
    同一ペルソナでA/B/Cを作るが、自由入力specの具体度に応じて優先順位を変える（Lv2）
    """
    role_map = {
        "A": "A：一番外さない（上品・日常適合が高い。清潔感と手元が綺麗に見える方向）",
        "B": "B：一番“今っぽい”（トレンド寄り。ただし上品で現実的。マグネット/オーロラ/ミラーはやりすぎない）",
        "C": "C：アクセントが新鮮（1〜2本 or 先端など、ワンポイントで攻める。奇抜NG、でも新しい自分）"
    }
    role_text = role_map.get(candidate_id, "方向性が被らないように提案する")

    specificity = safe_int(free_spec.get("specificity", 0), 0)
    free_mode = "LOW"
    if specificity >= 70:
        free_mode = "HIGH"
    elif specificity >= 35:
        free_mode = "MID"

    # In HIGH mode, spec.must/must_not are treated as top priority constraints (within safety + avoid_colors).
    priority_rules = ""
    if free_mode == "HIGH":
        priority_rules = """
【最重要ルール（自由入力が具体的な場合）】
- 以下の free_spec.must / free_spec.must_not は「最優先の設計条件」です（選択項目よりも上に扱う）
- ただし、お客様の avoid_colors（NG）や安全性（奇抜すぎない/上品/現実的）は常に守る
- must がある場合は必ず含め、must_not は絶対に踏まない
""".strip()
    elif free_mode == "MID":
        priority_rules = """
【重要ルール（自由入力がある程度具体的な場合）】
- free_spec.must / free_spec.must_not を強く反映する（可能な限り満たす）
- ただし選択項目との整合も保つ
""".strip()
    else:
        priority_rules = """
【自由入力が少ない/曖昧な場合】
- free_spec.soft/keywords はヒントとして扱い、選択項目を中心に提案する
""".strip()

    return f"""
あなたはプロのネイリストです。以下の「ネイリストのペルソナ」に厳密に従って、
お客様に合うネイル提案を【1案】だけ作ってください。

【ネイリストのペルソナ（必ず従う）】
{json.dumps(persona, ensure_ascii=False)}

【自由入力の解釈（free_spec）】
{json.dumps(free_spec, ensure_ascii=False)}

{priority_rules}

【絶対条件（Hard constraints）】
- お客様の選択項目（vibe / purpose / nail_duration / age）を踏襲
- avoid_colors（自由入力）はNG/苦手が含まれる可能性があるため、踏まないこと（最優先の安全制約）
- 奇抜すぎない：新しさは80%程度（新しい自分を発見できるが上品で現実的）
- 日常へ馴染むかどうかは選択項目に応じて適切に調整（仕事寄りなら控えめ、イベント寄りなら少し遊ぶ）
- ベージュ単色に寄りすぎない（程よく鮮やかさ・血色・透明感を入れる）
- ストーン/ラメ/アートは回答に応じて適切に。華やかさは“ワンポイント”で上品に

【この案の役割（必ず守る）】
{role_text}

【出力（JSONのみ）】
{{
  "id": "{candidate_id}",
  "plan_ja": "【ネイルコンセプト】...\\n【デザイン詳細】...",
  "style_hint": "内部用の短いメモ（ペルソナらしさ/狙い）"
}}

【お客様情報（そのまま）】
{user_text}

【お客様の選択項目サマリ】
{json.dumps(selected_summary, ensure_ascii=False)}
""".strip()

# =========================================================
# 2) Bayesian type model (unchanged)
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
    if isinstance(purpose, str):
        purpose = [purpose]
    vibe = form.get("vibe", [])
    if isinstance(vibe, str):
        vibe = [vibe]
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
        if answer_value.startswith("10"):
            target = 0.15
        elif answer_value.startswith("30"):
            target = 0.35
        elif answer_value.startswith("60"):
            target = 0.65
        elif answer_value.startswith("80"):
            target = 0.80
        d = abs(t["risk"] - target)
        return 0.10 + 0.90 * math.exp(-4.0 * d)

    if question_id == "top_priority":
        if answer_value == "仕事/生活で浮かない":
            s = 0.15 + 0.85 * (0.7 * t["daily"] + 0.3 * t["minimal"])
        elif answer_value == "手元がきれいに見える":
            s = 0.15 + 0.85 * (0.6 * t["minimal"] + 0.4 * (1.0 - abs(t["vivid"] - 0.55)))
        elif answer_value == "気分が上がる":
            s = 0.15 + 0.85 * (0.5 * t["vivid"] + 0.5 * t["risk"])
        elif answer_value == "周りに褒められたい":
            s = 0.15 + 0.85 * (0.6 * t["trend"] + 0.4 * t["sparkle"])
        elif answer_value == "写真映え":
            s = 0.15 + 0.85 * (0.6 * t["trend"] + 0.4 * t["vivid"])
        else:
            s = 0.2
        return clamp01(s)

    if question_id == "accent_preference":
        if "繊細" in answer_value:
            s = 0.15 + 0.85 * (0.6 * t["minimal"] + 0.4 * (1.0 - t["sparkle"]))
        elif "分かりやすい" in answer_value:
            s = 0.15 + 0.85 * (0.55 * t["sparkle"] + 0.45 * t["vivid"])
        elif "先端" in answer_value:
            s = 0.15 + 0.85 * (0.5 * t["minimal"] + 0.5 * t["trend"])
        elif "アクセント爪" in answer_value:
            s = 0.15 + 0.85 * (0.55 * t["risk"] + 0.45 * t["trend"])
        else:
            s = 0.2
        return clamp01(s)

    if question_id == "outfit_style":
        if "きれいめ" in answer_value:
            s = 0.15 + 0.85 * (0.55 * t["minimal"] + 0.45 * t["daily"])
        elif "カジュアル" in answer_value:
            s = 0.15 + 0.85 * (0.45 * t["daily"] + 0.55 * (1.0 - abs(t["vivid"] - 0.6)))
        elif "フェミニン" in answer_value:
            s = 0.15 + 0.85 * (0.55 * t["vivid"] + 0.45 * t["sparkle"])
        elif "モード" in answer_value:
            s = 0.15 + 0.85 * (0.65 * t["trend"] + 0.35 * (1.0 - t["sparkle"]))
        elif "韓国" in answer_value:
            s = 0.15 + 0.85 * (0.75 * t["trend"] + 0.25 * t["vivid"])
        elif "バラバラ" in answer_value:
            s = 0.25 + 0.75 * (0.5 * t["risk"] + 0.5 * t["trend"])
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
# 3) Candidate evaluation / selection (Lv2: free_input_alignment)
# =========================================================

AXES_BASE = [
    "adherence_to_selections",
    "wearability_daily_fit",
    "novelty_target_80",
    "colorfulness_not_beige_only",
    "accent_fit_one_point"
]
FREE_AXIS = "free_input_alignment"

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

def free_weight_from_specificity(spec: dict) -> float:
    """
    自由入力の影響度（0.0〜0.35）
    """
    s = safe_int(spec.get("specificity", 0), 0)
    if s >= 85:
        return 0.35
    if s >= 70:
        return 0.30
    if s >= 50:
        return 0.22
    if s >= 35:
        return 0.14
    if s >= 20:
        return 0.08
    return 0.0

def expected_utility(scores: dict, posterior: list, free_spec: dict) -> float:
    """
    既存軸の期待効用 + 自由入力整合の加点（具体度に応じて）
    """
    base_u = 0.0
    for p, th in zip(posterior, TYPE_SPACE):
        w = TYPE_WEIGHTS[th["id"]]
        su = 0.0
        for ax in AXES_BASE:
            su += w.get(ax, 0.0) * (float(scores.get(ax, 0.0)) / 100.0)
        base_u += p * su

    fw = free_weight_from_specificity(free_spec)
    free_score = float(scores.get(FREE_AXIS, 0.0) or 0.0) / 100.0
    return base_u + fw * free_score

def pick_by_expected_utility(candidates: list, eval_payload: dict, posterior: list, free_spec: dict) -> dict:
    results = eval_payload.get("results") or []
    by_id = {r.get("id"): r for r in results if r.get("id")}

    specificity = safe_int(free_spec.get("specificity", 0), 0)
    hard_gate = (specificity >= 70)

    best = None
    for c in candidates:
        cid = c.get("id")
        r = by_id.get(cid, {})
        scores = (r.get("scores") or {})

        # Hard gate when free input is very specific:
        if hard_gate:
            align = float(scores.get(FREE_AXIS, 0) or 0)
            if align < 70:
                continue  # disqualify

        eu = expected_utility(scores, posterior, free_spec)
        adherence = float(scores.get("adherence_to_selections", 0) or 0)
        free_align = float(scores.get(FREE_AXIS, 0) or 0)

        # tie-breaker: EU -> free_align -> adherence
        tup = (eu, free_align, adherence)
        if (best is None) or (tup > best["tup"]):
            best = {"candidate": c, "eval": r, "eu": eu, "tup": tup}

    # If everything got disqualified, fall back to highest free_align then adherence
    if best is None and candidates:
        for c in candidates:
            cid = c.get("id")
            r = by_id.get(cid, {})
            scores = (r.get("scores") or {})
            free_align = float(scores.get(FREE_AXIS, 0) or 0)
            adherence = float(scores.get("adherence_to_selections", 0) or 0)
            tup = (free_align, adherence)
            if (best is None) or (tup > best["tup"]):
                best = {"candidate": c, "eval": r, "eu": 0.0, "tup": (0.0, free_align, adherence)}

    return best or {"candidate": candidates[0] if candidates else {}, "eval": {}, "eu": 0.0, "tup": (0.0, 0.0, 0.0)}

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
# 6) Main finalize (Lv2)
# =========================================================

def finalize_with_posterior(img_bytes: bytes, form: dict, posterior: list):
    user_text = build_user_text(form)

    selected_summary = {
        "age": form.get("age", ""),
        "nail_duration": form.get("nail_duration", ""),
        "purpose": form.get("purpose", []),
        "vibe": form.get("vibe", []),
        "avoid_colors": form.get("avoid_colors", ""),  # free input field
        "challenge_level": form.get("challenge_level", ""),
        "outfit_style": form.get("outfit_style", ""),
        "top_priority": form.get("top_priority", ""),
        "accent_preference": form.get("accent_preference", "")
    }

    persona = get_persona_from_form(form)

    # --- Lv2: Build free_spec from free input ---
    free_text = str(form.get("avoid_colors", "") or "").strip()
    free_spec = extract_free_spec(free_text, selected_summary)

    # -----------------------------------------------------
    # (1) Generate 3 candidates (A/B/C) with free_spec injected
    # -----------------------------------------------------
    candidates = []
    for cid in ["A", "B", "C"]:
        prompt = build_persona_candidate_prompt(cid, persona, user_text, selected_summary, free_spec)

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
    # (2) Evaluate candidates (add free_input_alignment)
    # -----------------------------------------------------
    eval_prompt = f"""
あなたはネイル提案の品質評価者です。
下の「お客様の選択項目」「free_spec（自由入力の解釈）」「候補3案」を読み、各案を0〜100点で採点してください。

採点軸（0〜100）：
- adherence_to_selections: 選択項目（vibe/purpose/nail_duration/age）への忠実さ
- wearability_daily_fit: 目的に応じた日常適合（仕事なら浮かない、イベントなら程よく映える等）
- novelty_target_80: “80%新しさ”のちょうど良さ（奇抜すぎない・でも新しい）
- colorfulness_not_beige_only: ベージュ単色に寄りすぎず、程よい鮮やかさ/血色/透明感がある
- accent_fit_one_point: ストーン/ラメ/アートの使い方が回答に合い、ワンポイントで上品
- free_input_alignment: free_spec.must / must_not / soft をどれだけ満たしているか（自由入力の反映度）

重要ルール（超重要）：
- avoid_colors（自由入力）は“NG/苦手”が含まれる可能性があります。明確なNGを踏んでいる場合は大幅減点。
- free_spec.specificity が高い（70以上）場合、free_spec.must / must_not を満たせていない案は free_input_alignment を低くし、全体評価も厳しくしてください。
- 追加項目（challenge_level / outfit_style / top_priority / accent_preference）があれば整合している案を加点（ただしNG違反は絶対にNG）。

出力は【JSONのみ】：
{{
  "results": [
    {{
      "id":"A",
      "scores": {{
        "adherence_to_selections": 0,
        "wearability_daily_fit": 0,
        "novelty_target_80": 0,
        "colorfulness_not_beige_only": 0,
        "accent_fit_one_point": 0,
        "free_input_alignment": 0
      }},
      "notes":"短い根拠（内部用）"
    }}
  ]
}}

free_spec（自由入力の解釈）：
{json.dumps(free_spec, ensure_ascii=False)}

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

    # Pick with Lv2: hard-gate + bonus from free alignment
    picked = pick_by_expected_utility(candidates, eval_payload, posterior, free_spec)
    plan_text = (picked.get("candidate") or {}).get("plan_ja") or candidates[0].get("plan_ja", "")

    # -----------------------------------------------------
    # (3) Build English image-edit prompt (prioritize free_spec when specific)
    # -----------------------------------------------------
    spec_prompt = f"""
You are a top nail artist who writes image-edit prompts.

Create an English prompt to edit the uploaded photo.

Hard rules (must follow):
- Keep the same hand, skin tone, lighting, background, and composition.
- Edit ONLY the nails. Do NOT change fingers, skin, jewelry, or background.
- Do NOT add text, watermark, or logos.

Priority rules:
- If free_spec.specificity is high (>=70), treat free_spec.must and free_spec.must_not as top-priority constraints.
- Always respect customer's avoid items and must_not.

Design constraints:
- Follow the customer's selected options (vibe / purpose / nail_duration / age).
- Not overly eccentric: aim for around 80% freshness—wearable and elegant.
- Avoid plain beige-only; keep it moderately vivid.
- If adding sparkle/art, keep it as ONE tasteful accent point.

Return ONLY valid JSON:
{{"edit_prompt_en":"..."}}

free_spec:
{json.dumps(free_spec, ensure_ascii=False)}

Customer selections:
{json.dumps(selected_summary, ensure_ascii=False)}

Customer full text (for context):
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
        # fallback: basic prompt + free_spec highlights
        must = free_spec.get("must") or []
        must_not = free_spec.get("must_not") or []
        edit_prompt = (
            "Keep the same hand, skin tone, lighting, background, and composition. "
            "Edit ONLY the nails (do not change fingers/skin/jewelry/background). "
            "No text, no watermark. "
            "Follow customer selections first. "
            "Aim for ~80% freshness while staying wearable and elegant. "
            "Avoid a plain beige-only look; keep it moderately vivid. "
            "Use only ONE tasteful accent point if needed. "
        )
        if must:
            edit_prompt += "Must include: " + "; ".join(must) + ". "
        if must_not:
            edit_prompt += "Must NOT include: " + "; ".join(must_not) + ". "

    # -----------------------------------------------------
    # (4) Image edit
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
            "persona_id_used": persona.get("persona_id"),
            "persona_name": persona.get("display_name"),
            "free_spec": free_spec,
            "posterior_top3": top,
            "picked_expected_utility": picked.get("eu"),
            "picked_id": (picked.get("candidate") or {}).get("id"),
            "candidates_debug": candidates,
            "eval_debug": eval_payload
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
