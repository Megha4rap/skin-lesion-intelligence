"""
Multimodal Skin Lesion Intelligence  -  Streamlit dashboard
ResNet50 + metadata  and  EfficientNet-B0 + metadata  (cross-modal attention + gated fusion), ensembled.

Run:  streamlit run app.py
Folder layout expected:
    app.py, multimodal_lib.py, requirements.txt
    tokenizer/
    models/fusion_resnet50.pt, models/fusion_efficientnet_b0.pt, models/ensemble.json,
    models/results.csv, models/confusion_matrix.png
"""
import base64
import io
import json
import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import DistilBertTokenizerFast

from multimodal_lib import (CLASSES, CLASS_NAMES, RISK, LOCALIZATIONS, SEXES, MAX_LEN,
                            build_text, eval_tf, load_fusion, grad_cam, overlay)

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEL = CLASSES.index('mel')
NAMES = {'resnet50': 'ResNet50 + Metadata', 'efficientnet_b0': 'EfficientNet-B0 + Metadata'}
MODEL_COLORS = ['#6366f1', '#14b8a6', '#f97316']
RISK_STYLE = {  # css class, colour, emoji
    'Benign': ('v-ben', '#10b981', '🟢'),
    'Pre-cancerous': ('v-pre', '#f59e0b', '🟡'),
    'Malignant': ('v-mal', '#ef4444', '🔴'),
}
SCREEN_MODES = {  # (refer threshold, monitor threshold) on P(malignant or pre-cancerous)
    'Balanced': (0.50, 0.25),
    'High sensitivity (flag more)': (0.35, 0.15),
    'High specificity (flag fewer)': (0.65, 0.35),
}

st.set_page_config(page_title='Skin Lesion Intelligence', page_icon='🩺', layout='wide')

# =============================================================================
# THEME / CSS
# =============================================================================
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1300px;}
.hero {background: linear-gradient(135deg,#4f46e5 0%,#0891b2 100%); padding: 1.6rem 2rem;
       border-radius: 20px; color: #fff; margin-bottom: 1.1rem;
       box-shadow: 0 10px 30px rgba(79,70,229,.25);}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; letter-spacing: -.5px;}
.hero p {margin: .35rem 0 .8rem; opacity: .92; font-size: 1.02rem;}
.chip {display: inline-block; padding: 3px 12px; border-radius: 999px; background: rgba(255,255,255,.2);
       font-size: .78rem; margin: 0 6px 4px 0; backdrop-filter: blur(4px);}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 1rem 1.2rem;
       background: rgba(128,128,128,.06); height: 100%;}
.card h4 {margin: 0 0 .4rem 0;}
.verdict {border-radius: 18px; padding: 1.3rem 1.6rem; color: #fff; margin-bottom: 1rem;
          box-shadow: 0 8px 24px rgba(0,0,0,.18);}
.v-ben {background: linear-gradient(135deg,#047857,#10b981);}
.v-pre {background: linear-gradient(135deg,#b45309,#f59e0b);}
.v-mal {background: linear-gradient(135deg,#b91c1c,#ef4444);}
.verdict .small {opacity: .9; font-size: .85rem; text-transform: uppercase; letter-spacing: 1px;}
.verdict .big {font-size: 1.9rem; font-weight: 700; line-height: 1.2; margin: .15rem 0 .5rem;}
.vchip {display: inline-block; background: rgba(255,255,255,.22); padding: 3px 12px; border-radius: 999px;
        font-size: .82rem; margin: 0 6px 4px 0;}
.bar {height: 14px; border-radius: 8px; overflow: hidden; display: flex; background: rgba(128,128,128,.3);}
.bar .i {background: #6366f1;} .bar .m {background: #14b8a6;}
.legend {display: flex; justify-content: space-between; font-size: .8rem; opacity: .85; margin-top: 3px;}
.row {display: grid; grid-template-columns: 1.7fr 2fr .55fr; gap: 10px; align-items: center; margin: 7px 0;}
.track {height: 10px; border-radius: 6px; background: rgba(128,128,128,.25); overflow: hidden;}
.fill {height: 100%; border-radius: 6px;}
.tag {color: #fff; font-size: .68rem; padding: 1px 8px; border-radius: 999px; margin-left: 6px;}
.rv {text-align: right; font-weight: 600; font-size: .9rem;}
.step {text-align: center; padding: .6rem;}
.step .n {font-size: 2rem;}
.small-note {font-size: .8rem; opacity: .75;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.2);
                             padding: .7rem 1rem; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING + INFERENCE
# =============================================================================
@st.cache_resource(show_spinner='Loading models...')
def get_assets():
    tok = DistilBertTokenizerFast.from_pretrained(os.path.join(BASE, 'tokenizer'))
    models = {b: load_fusion(b, os.path.join(BASE, 'models', f'fusion_{b}.pt'), DEVICE)
              for b in NAMES}
    w = json.load(open(os.path.join(BASE, 'models', 'ensemble.json')))['w_resnet50']
    return tok, models, w


try:
    tok, models, W_DEFAULT = get_assets()
except Exception as e:  # noqa
    st.error('Could not load the trained models. Make sure `models/` and `tokenizer/` sit next to '
             '`app.py` (run `python train.py` first).')
    st.code(str(e))
    st.stop()


def tokenize(texts):
    enc = tok(texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    return enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)


def forward_models(x, texts, tta=False):
    """x:(1,3,H,W)  texts:list[str]  ->  per-model probs (n,7) and image-gate (n,). Optional 4-view TTA."""
    ids, mask = tokenize(texts)
    n = len(texts)
    views = [x]
    if tta:
        views += [torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])]
    probs = {b: np.zeros((n, len(CLASSES))) for b in models}
    gates = {b: np.zeros(n) for b in models}
    with torch.no_grad():
        for v in views:
            xv = v.expand(n, -1, -1, -1)
            for b, m in models.items():
                probs[b] += m(xv, ids, mask).softmax(1).cpu().numpy() / len(views)
                gates[b] += m.last_gate.cpu().numpy() / len(views)
    return probs, gates


def to_tensor(img_bytes):
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return pil, eval_tf(pil).unsqueeze(0).to(DEVICE)


@st.cache_data(show_spinner=False)
def cached_predict(img_bytes, note, tta):
    _, x = to_tensor(img_bytes)
    probs, gates = forward_models(x, [note], tta)
    return {b: p[0] for b, p in probs.items()}, {b: float(g[0]) for b, g in gates.items()}


@st.cache_data(show_spinner=False)
def cached_sweep(img_bytes, notes, tta=False):
    """Batch many clinical notes for ONE image. notes is a tuple of strings."""
    _, x = to_tensor(img_bytes)
    pr, pe = [], []
    for i in range(0, len(notes), 24):
        probs, _ = forward_models(x, list(notes[i:i + 24]), tta)
        pr.append(probs['resnet50']); pe.append(probs['efficientnet_b0'])
    return np.concatenate(pr), np.concatenate(pe)


@st.cache_data(show_spinner=False)
def cached_cam(img_bytes, note, backbone, cls):
    _, x = to_tensor(img_bytes)
    ids, mask = tokenize([note])
    m = models[backbone]
    cam, _ = grad_cam(m, x, ids, mask, m.enc.cam_layer, cls=cls)
    return cam


def blend(d, w):
    return w * d['resnet50'] + (1 - w) * d['efficientnet_b0']


def _to_img(img):
    """Return a displayable uint8 PIL image from PIL / float / uint8 arrays (fixes blank or washed-out heat-maps)."""
    if isinstance(img, np.ndarray):
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = a.astype(np.float32)
            a = np.nan_to_num(a)
            a = a * 255.0 if a.max() <= 1.0 + 1e-6 else a
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.shape[-1] == 4:
            a = a[..., :3]
        return Image.fromarray(a)
    return img


def show_image(target, img, caption=None, max_side=900):
    """Render an image as an inline base64 <img>. This bypasses Streamlit's media-file server, which is what
    causes the broken-image icon (alt text '0') when files expire/reset between reruns."""
    img = _to_img(img).convert('RGB')
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    cap = (f'<div style="text-align:center;font-size:.85rem;opacity:.75;margin-top:4px">{caption}</div>'
           if caption else '')
    target.markdown(f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:10px;display:block">'
                    f'{cap}', unsafe_allow_html=True)


# =============================================================================
# SMALL HELPERS
# =============================================================================
def p_malignant(p):
    return float(sum(p[i] for i, c in enumerate(CLASSES) if RISK[c] != 'Benign'))


def certainty(p):
    ent = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
    return float(1 - ent)


def triage(pm, mode):
    hi, mid = SCREEN_MODES[mode]
    if pm >= hi:
        return 'Refer', '🚨 Screening flag: prompt dermatologist evaluation advised'
    if pm >= mid:
        return 'Monitor', '👀 Borderline: monitor the lesion and consider a clinical check'
    return 'Routine', '✅ Low model concern: routine skin self-monitoring'


def gate_bar(g):
    return (f'<div class="bar"><div class="i" style="width:{g*100:.0f}%"></div>'
            f'<div class="m" style="width:{(1-g)*100:.0f}%"></div></div>'
            f'<div class="legend"><span>🖼️ Image {g*100:.0f}%</span>'
            f'<span>📝 Metadata {(1-g)*100:.0f}%</span></div>')


def top_rows(p, k=3):
    html = ''
    for i in np.argsort(-p)[:k]:
        c = CLASSES[i]
        col = RISK_STYLE[RISK[c]][1]
        html += (f'<div class="row"><div>{CLASS_NAMES[c]}<span class="tag" style="background:{col}">'
                 f'{RISK[c]}</span></div><div class="track"><div class="fill" '
                 f'style="width:{p[i]*100:.1f}%;background:{col}"></div></div>'
                 f'<div class="rv">{p[i]*100:.1f}%</div></div>')
    return html


def long_df(pr, pe, pens):
    rows = []
    for label, p in [(NAMES['resnet50'], pr), (NAMES['efficientnet_b0'], pe), ('Ensemble', pens)]:
        for i, c in enumerate(CLASSES):
            rows.append({'Model': label, 'Class': c, 'Name': CLASS_NAMES[c], 'Probability': float(p[i])})
    return pd.DataFrame(rows)


def prob_chart(d):
    return alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X('Class:N', sort=CLASSES, axis=alt.Axis(labelAngle=0, title=None)),
        xOffset='Model:N',
        y=alt.Y('Probability:Q', axis=alt.Axis(format='%', title=None)),
        color=alt.Color('Model:N', scale=alt.Scale(range=MODEL_COLORS), legend=alt.Legend(orient='top')),
        tooltip=['Model', 'Name', alt.Tooltip('Probability:Q', format='.2%')]).properties(height=320)


def style_map(styler, fn, subset):
    return (styler.map if hasattr(styler, 'map') else styler.applymap)(fn, subset=subset)


def risk_color(v):
    col = {'Benign': '#10b981', 'Pre-cancerous': '#f59e0b', 'Malignant': '#ef4444'}.get(v)
    return f'background-color:{col};color:white' if col else ''


def b64_png(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    img = img.copy(); img.thumbnail((420, 420))
    buf = io.BytesIO(); img.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


def arch_dot():
    return """
    digraph G { rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor="#111"];
      edge [color="#888"];
      img  [label="Dermoscopic\\nimage", fillcolor="#fde68a"];
      meta [label="Age / Sex / Site\\n→ clinical note", fillcolor="#fde68a"];
      cnn  [label="ResNet50 or\\nEfficientNet-B0", fillcolor="#c7d2fe"];
      bert [label="DistilBERT", fillcolor="#99f6e4"];
      tok  [label="49 region tokens\\n(7x7 feature map)", fillcolor="#e0e7ff"];
      wt   [label="word tokens", fillcolor="#ccfbf1"];
      attn [label="Cross-modal attention\\n(text↔image) + learned gate", fillcolor="#fbcfe8"];
      mlp  [label="MLP head", fillcolor="#fed7aa"];
      ens  [label="Weighted ensemble\\n(ResNet + EfficientNet)", fillcolor="#bbf7d0"];
      out  [label="7-class prediction\\n+ risk + Grad-CAM", fillcolor="#fecaca"];
      img -> cnn -> tok -> attn; meta -> bert -> wt -> attn; attn -> mlp -> ens -> out; }
    """


def build_report(pil, cams, note, pr, pe, pens, gates, verdict, tri_text, mode, tta, w):
    css = ('body{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#1f2937}'
           'h1{color:#4338ca}.v{padding:16px;border-radius:12px;color:#fff;margin:12px 0}'
           'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px 10px;text-align:left}'
           'th{background:#eef2ff}.imgs{display:flex;gap:14px}.imgs div{text-align:center;font-size:12px}'
           '.warn{background:#fef3c7;padding:10px;border-radius:8px;font-size:13px;margin-top:20px}')
    risk, code = verdict
    col = RISK_STYLE[risk][1]
    rows = ''.join(
        f'<tr><td>{CLASS_NAMES[c]} ({c})</td><td>{pr[i]*100:.1f}%</td><td>{pe[i]*100:.1f}%</td>'
        f'<td><b>{pens[i]*100:.1f}%</b></td></tr>' for i, c in enumerate(CLASSES))
    imgs = (f'<div><img src="data:image/png;base64,{b64_png(pil)}"><br>Input</div>' +
            ''.join(f'<div><img src="data:image/png;base64,{b64_png(c)}"><br>{n}</div>' for n, c in cams))
    return (f'<html><head><meta charset="utf-8"><title>Skin lesion AI report</title><style>{css}</style></head><body>'
            f'<h1>Multimodal Skin Lesion Analysis Report</h1>'
            f'<p>Generated {datetime.now():%Y-%m-%d %H:%M} | Clinical note: <i>{note}</i></p>'
            f'<div class="v" style="background:{col}"><b>{risk}</b>: {CLASS_NAMES[code]} ({code}) '
            f'– confidence {pens[CLASSES.index(code)]*100:.1f}%<br>{tri_text}</div>'
            f'<div class="imgs">{imgs}</div><h3>Class probabilities</h3>'
            f'<table><tr><th>Class</th><th>{NAMES["resnet50"]}</th><th>{NAMES["efficientnet_b0"]}</th>'
            f'<th>Ensemble</th></tr>{rows}</table>'
            f'<p>Modality gate (image weight): ResNet {gates["resnet50"]*100:.0f}% | '
            f'EfficientNet {gates["efficientnet_b0"]*100:.0f}%<br>Settings: screening mode “{mode}”, '
            f'TTA {"on" if tta else "off"}, ensemble weight ResNet {w:.2f}</p>'
            f'<div class="warn">Research / educational prototype. NOT a medical device and NOT a substitute '
            f'for diagnosis by a qualified dermatologist.</div></body></html>')


# =============================================================================
# SIDEBAR
# =============================================================================
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.markdown('## 🧾 Patient metadata')
age = st.sidebar.slider('Age', 1, 100, 50)
sex = st.sidebar.selectbox('Sex', SEXES)
loc = st.sidebar.selectbox('Lesion location', LOCALIZATIONS, index=LOCALIZATIONS.index('back'))
note = build_text(age, sex, loc)
st.sidebar.caption('Clinical note fed to the DistilBERT branch:')
st.sidebar.info(note)

st.sidebar.markdown('## ⚙️ Inference settings')
w_res = st.sidebar.slider('Ensemble weight on ResNet50', 0.0, 1.0, float(W_DEFAULT), 0.05,
                          help='Trained value was tuned on the validation set. EfficientNet gets 1 − w.')
tta = st.sidebar.toggle('Test-time augmentation (4 views)', value=False,
                        help='Averages predictions over flips: slower but more stable.')
mode = st.sidebar.selectbox('Screening mode', list(SCREEN_MODES),
                            help='Sets how readily the app raises a referral flag.')
alpha = st.sidebar.slider('Heat-map opacity', 0.1, 0.9, 0.45, 0.05)

# =============================================================================
# HERO
# =============================================================================
n_params = {b: sum(p.numel() for p in m.parameters()) / 1e6 for b, m in models.items()}
st.markdown(f"""
<div class="hero">
  <h1>🩺 Skin Lesion Intelligence</h1>
  <p>Dermoscopic image × clinical metadata → cross-modal attention fusion with two CNN backbones, ensembled.</p>
  <span class="chip">⚡ {'GPU' if DEVICE.type == 'cuda' else 'CPU'}</span>
  <span class="chip">🧠 2 fusion models loaded</span>
  <span class="chip">⚖️ ResNet {w_res:.2f} / EffNet {1-w_res:.2f}</span>
  <span class="chip">🔁 TTA {'on' if tta else 'off'}</span>
  <span class="chip">🎚️ {mode.split(' (')[0]}</span>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader('📤 Upload a dermoscopic image (jpg / png)', type=['jpg', 'jpeg', 'png'])

# =============================================================================
# ANALYSIS (shared by tabs)
# =============================================================================
A = None
if file is not None:
    img_bytes = file.getvalue()
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    with st.spinner('Analysing lesion...'):
        probs, gates = cached_predict(img_bytes, note, tta)
    pr, pe = probs['resnet50'], probs['efficientnet_b0']
    pens = blend(probs, w_res)
    top = int(pens.argmax())
    code = CLASSES[top]
    pm = p_malignant(pens)
    level, tri_text = triage(pm, mode)
    A = dict(pil=pil, bytes=img_bytes, pr=pr, pe=pe, pens=pens, gates=gates, top=top, code=code,
             risk=RISK[code], pm=pm, level=level, tri=tri_text,
             agree=bool(pr.argmax() == pe.argmax()))
    key = (file.name, note, tta, round(w_res, 2))
    if st.session_state.get('last_key') != key:
        st.session_state.last_key = key
        st.session_state.history.append({
            'time': datetime.now().strftime('%H:%M:%S'), 'file': file.name, 'age': age, 'sex': sex,
            'location': loc, 'prediction': CLASS_NAMES[code], 'risk': A['risk'],
            'confidence': round(float(pens[top]), 4), 'P(melanoma)': round(float(pens[MEL]), 4),
            'P(malignant/pre)': round(pm, 4), 'models agree': A['agree'], 'triage': level})

tabs = st.tabs(['🔍 Prediction', '🧠 Explainability', '🧪 What-if Lab', '📁 Batch',
                '📊 Performance', '🕘 History', 'ℹ️ About'])

# =============================================================================
# TAB 1 - PREDICTION
# =============================================================================
with tabs[0]:
    if A is None:
        st.info('Upload a dermoscopic image above and set the patient metadata in the sidebar.')
        c = st.columns(4)
        for col, (n, t, d) in zip(c, [('1️⃣', 'Upload', 'a dermoscopic lesion image'),
                                      ('2️⃣', 'Describe', 'age, sex and lesion site (sidebar)'),
                                      ('3️⃣', 'Fuse', 'CNN image features + DistilBERT note via attention'),
                                      ('4️⃣', 'Review', 'risk, Grad-CAM, what-if analysis, report')]):
            col.markdown(f'<div class="card step"><div class="n">{n}</div><b>{t}</b><br>{d}</div>',
                         unsafe_allow_html=True)
        st.markdown('#### How the model works')
        st.graphviz_chart(arch_dot(), use_container_width=True)
    else:
        css_cls, col_hex, emoji = RISK_STYLE[A['risk']]
        st.markdown(
            f'<div class="verdict {css_cls}"><div class="small">{emoji} Ensemble verdict</div>'
            f'<div class="big">{A["risk"]}: {CLASS_NAMES[A["code"]]}</div>'
            f'<span class="vchip">confidence {A["pens"][A["top"]]*100:.1f}%</span>'
            f'<span class="vchip">certainty {certainty(A["pens"])*100:.0f}%</span>'
            f'<span class="vchip">{"models agree ✅" if A["agree"] else "models disagree ⚠️"}</span>'
            f'<span class="vchip">{A["tri"]}</span></div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Melanoma probability', f'{A["pens"][MEL]*100:.1f}%',
                  delta=f'{(A["pens"][MEL]-A["pr"][MEL])*100:+.1f}% vs ResNet', delta_color='off')
        m2.metric('P(malignant or pre-cancerous)', f'{A["pm"]*100:.1f}%')
        m3.metric('Referral level', A['level'])
        m4.metric('Certainty (1 − entropy)', f'{certainty(A["pens"])*100:.0f}%')

        c0, c1, c2 = st.columns([1, 1, 1])
        with c0:
            st.markdown('#### Input')
            show_image(st, A['pil'])
            st.caption(f'📝 {note}')
        cams = []
        for col, b, p in zip((c1, c2), NAMES, (A['pr'], A['pe'])):
            with col:
                st.markdown(f'#### {NAMES[b]}')
                cam = cached_cam(A['bytes'], note, b, A['top'])
                ov = overlay(A['pil'], cam, alpha)
                cams.append((NAMES[b], ov))
                show_image(st, ov, caption=f'Grad-CAM for “{A["code"]}”')
                pt = int(p.argmax())
                st.markdown(f'Prediction: **{CLASS_NAMES[CLASSES[pt]]}** ({p[pt]*100:.1f}%)')
                st.markdown(gate_bar(A['gates'][b]), unsafe_allow_html=True)

        left, right = st.columns([1.6, 1])
        with left:
            st.markdown('#### Class probabilities')
            st.altair_chart(prob_chart(long_df(A['pr'], A['pe'], A['pens'])), use_container_width=True)
            st.caption('akiec = actinic keratosis · bcc = basal cell carcinoma · bkl = benign keratosis · '
                       'df = dermatofibroma · mel = melanoma · nv = nevus · vasc = vascular')
        with right:
            st.markdown('#### Top-3 (ensemble)')
            st.markdown(f'<div class="card">{top_rows(A["pens"])}</div>', unsafe_allow_html=True)
            risk_df = pd.DataFrame({'Risk group': ['Benign', 'Pre-cancerous', 'Malignant'],
                                    'Probability': [sum(A['pens'][i] for i, c in enumerate(CLASSES) if RISK[c] == r)
                                                    for r in ['Benign', 'Pre-cancerous', 'Malignant']]})
            st.altair_chart(alt.Chart(risk_df).mark_arc(innerRadius=45).encode(
                theta='Probability:Q',
                color=alt.Color('Risk group:N', scale=alt.Scale(
                    domain=['Benign', 'Pre-cancerous', 'Malignant'],
                    range=['#10b981', '#f59e0b', '#ef4444']), legend=alt.Legend(orient='bottom')),
                tooltip=['Risk group', alt.Tooltip('Probability:Q', format='.1%')]).properties(height=200),
                use_container_width=True)

        with st.expander('Full probability table'):
            tbl = pd.DataFrame({NAMES['resnet50']: A['pr'], NAMES['efficientnet_b0']: A['pe'],
                                'Ensemble': A['pens']}, index=[CLASS_NAMES[c] for c in CLASSES])
            st.dataframe(tbl.style.format('{:.3%}').highlight_max(axis=0), use_container_width=True)

        report = build_report(A['pil'], cams, note, A['pr'], A['pe'], A['pens'], A['gates'],
                              (A['risk'], A['code']), A['tri'], mode, tta, w_res)
        st.download_button('⬇️ Download analysis report (HTML)', report,
                           file_name=f'skin_report_{datetime.now():%Y%m%d_%H%M%S}.html', mime='text/html')
    st.warning('Research / educational prototype only. NOT a medical device and NOT a substitute for '
               'diagnosis by a qualified dermatologist.')

# =============================================================================
# TAB 2 - EXPLAINABILITY
# =============================================================================
with tabs[1]:
    if A is None:
        st.info('Upload an image to explore explanations.')
    else:
        st.markdown('### Where is each model looking?')
        cls = st.selectbox('Explain which class?', list(range(len(CLASSES))), index=A['top'],
                           format_func=lambda i: f'{CLASS_NAMES[CLASSES[i]]} ({CLASSES[i]})')
        cam_r = cached_cam(A['bytes'], note, 'resnet50', cls)
        cam_e = cached_cam(A['bytes'], note, 'efficientnet_b0', cls)
        import matplotlib
        raw = lambda cam: matplotlib.colormaps['jet'](cam)[..., :3]
        c = st.columns(4)
        show_image(c[0], A['pil'], caption='Original')
        show_image(c[1], overlay(A['pil'], cam_r, alpha), caption='ResNet50 + meta')
        show_image(c[2], overlay(A['pil'], cam_e, alpha), caption='EfficientNet-B0 + meta')
        show_image(c[3], raw((cam_r + cam_e) / 2), caption='Combined raw heat-map')
        corr = np.corrcoef(cam_r.ravel(), cam_e.ravel())[0, 1]
        e1, e2, e3 = st.columns(3)
        e1.metric('Attention agreement (corr.)', 'n/a' if np.isnan(corr) else f'{corr:.2f}',
                  help='How similar the two models’ heat-maps are. High = both focus on the same region.')
        e2.metric('Image weight · ResNet', f'{A["gates"]["resnet50"]*100:.0f}%')
        e3.metric('Image weight · EfficientNet', f'{A["gates"]["efficientnet_b0"]*100:.0f}%')

        st.markdown('### How much did the metadata matter for this patient?')
        neutral = build_text(None, 'unknown', 'unknown')
        pn_d, _ = cached_predict(A['bytes'], neutral, tta)
        pn = blend(pn_d, w_res)
        tv = 0.5 * np.abs(A['pens'] - pn).sum()
        cmp_df = pd.DataFrame(
            [{'Input': 'With real metadata', 'Class': c, 'Probability': float(A['pens'][i])}
             for i, c in enumerate(CLASSES)] +
            [{'Input': 'Metadata removed (neutral note)', 'Class': c, 'Probability': float(pn[i])}
             for i, c in enumerate(CLASSES)])
        k1, k2 = st.columns([1, 2])
        k1.metric('Metadata influence', f'{tv*100:.1f}%',
                  help='Total-variation distance between predictions with and without metadata.')
        k1.caption(f'Neutral note used: “{neutral}”')
        k1.write(f'Prediction without metadata: **{CLASS_NAMES[CLASSES[int(pn.argmax())]]}** '
                 f'({pn.max()*100:.1f}%)')
        k2.altair_chart(alt.Chart(cmp_df).mark_bar().encode(
            x=alt.X('Class:N', sort=CLASSES, axis=alt.Axis(labelAngle=0, title=None)), xOffset='Input:N',
            y=alt.Y('Probability:Q', axis=alt.Axis(format='%', title=None)),
            color=alt.Color('Input:N', scale=alt.Scale(range=['#6366f1', '#9ca3af']),
                            legend=alt.Legend(orient='top', title=None)),
            tooltip=['Input', 'Class', alt.Tooltip('Probability:Q', format='.2%')]).properties(height=260),
            use_container_width=True)

# =============================================================================
# TAB 3 - WHAT-IF LAB
# =============================================================================
with tabs[2]:
    if A is None:
        st.info('Upload an image to use the what-if lab.')
    else:
        st.markdown('Keep the **same image** and change only the clinical metadata to see how the risk responds.')
        cur = alt.value('#ef4444')

        if st.toggle('📍 Lesion-location sweep', value=True):
            notes = tuple(build_text(age, sex, l) for l in LOCALIZATIONS)
            r, e = cached_sweep(A['bytes'], notes, tta)
            P = w_res * r + (1 - w_res) * e
            sdf = pd.DataFrame({'location': LOCALIZATIONS, 'P(melanoma)': P[:, MEL],
                                'P(malignant/pre)': [p_malignant(p) for p in P],
                                'predicted': [CLASSES[int(p.argmax())] for p in P]})
            ch = alt.Chart(sdf).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y('location:N', sort='-x', title=None),
                x=alt.X('P(melanoma):Q', axis=alt.Axis(format='%')),
                color=alt.condition(alt.datum.location == loc, cur, alt.value('#6366f1')),
                tooltip=['location', alt.Tooltip('P(melanoma):Q', format='.2%'),
                         alt.Tooltip('P(malignant/pre):Q', format='.2%'), 'predicted']
            ).properties(height=380, title='P(melanoma) by lesion location (red = current)')
            a, b = st.columns([1.5, 1])
            a.altair_chart(ch, use_container_width=True)
            b.dataframe(sdf.set_index('location').style.format(
                {'P(melanoma)': '{:.2%}', 'P(malignant/pre)': '{:.2%}'}), use_container_width=True, height=380)

        if st.toggle('🎂 Age & sex sweep', value=False):
            ages = list(range(5, 96, 10))
            notes = tuple(build_text(a_, sex, loc) for a_ in ages)
            r, e = cached_sweep(A['bytes'], notes, tta)
            P = w_res * r + (1 - w_res) * e
            adf = pd.DataFrame({'Age': ages * 2,
                                'Metric': ['P(melanoma)'] * len(ages) + ['P(malignant/pre)'] * len(ages),
                                'Probability': list(P[:, MEL]) + [p_malignant(p) for p in P]})
            sx_notes = tuple(build_text(age, s_, loc) for s_ in SEXES)
            sr, se = cached_sweep(A['bytes'], sx_notes, tta)
            SP = w_res * sr + (1 - w_res) * se
            a, b = st.columns([1.5, 1])
            a.altair_chart(alt.Chart(adf).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Age:Q'), y=alt.Y('Probability:Q', axis=alt.Axis(format='%')),
                color=alt.Color('Metric:N', scale=alt.Scale(range=['#ef4444', '#6366f1']),
                                legend=alt.Legend(orient='top')),
                tooltip=['Age', 'Metric', alt.Tooltip('Probability:Q', format='.2%')]
            ).properties(height=300, title='Risk vs patient age'), use_container_width=True)
            b.markdown('**Effect of sex in the note**')
            b.dataframe(pd.DataFrame({'sex': SEXES, 'P(melanoma)': SP[:, MEL],
                                      'P(malignant/pre)': [p_malignant(p) for p in SP]}
                                     ).set_index('sex').style.format('{:.2%}'), use_container_width=True)

        if st.toggle('🗺️ Age × location heat-map (slower)', value=False):
            ages = [20, 40, 60, 80]
            grid = [(a_, l) for a_ in ages for l in LOCALIZATIONS]
            notes = tuple(build_text(a_, sex, l) for a_, l in grid)
            with st.spinner(f'Running {len(notes)} what-if scenarios...'):
                r, e = cached_sweep(A['bytes'], notes, False)
            P = w_res * r + (1 - w_res) * e
            hdf = pd.DataFrame({'Age': [str(a_) for a_, _ in grid], 'Location': [l for _, l in grid],
                                'P(melanoma)': P[:, MEL]})
            st.altair_chart(alt.Chart(hdf).mark_rect().encode(
                x=alt.X('Location:N', axis=alt.Axis(labelAngle=-40)), y=alt.Y('Age:O', sort=None),
                color=alt.Color('P(melanoma):Q', scale=alt.Scale(scheme='orangered'),
                                legend=alt.Legend(format='%')),
                tooltip=['Age', 'Location', alt.Tooltip('P(melanoma):Q', format='.2%')]
            ).properties(height=240, title='P(melanoma) over age × lesion location'),
                use_container_width=True)
        st.caption('These are model-sensitivity experiments, not clinical risk estimates.')

# =============================================================================
# TAB 4 - BATCH
# =============================================================================
# NOTE: widget keys (batch_files / batch_csv) must NOT collide with the key used to store the
# results DataFrame (batch_df). Streamlit stores a widget's value in session_state[key], so the
# old key='batch' made st.session_state.batch the uploaded-file *list*, not the results table.
with tabs[3]:
    st.markdown('Analyse many images at once. Optionally upload a CSV with columns '
                '`filename, age, sex, localization` for per-image metadata (otherwise the sidebar values are used).')
    files = st.file_uploader('Images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True,
                             key='batch_files')
    meta_csv = st.file_uploader('Metadata CSV (optional)', type=['csv'], key='batch_csv')
    if files and st.button('🚀 Run batch analysis', type='primary'):
        meta_map = {}
        if meta_csv is not None:
            try:
                mdf = pd.read_csv(meta_csv)
                meta_map = {str(r['filename']): r for _, r in mdf.iterrows()}
            except Exception as ex:  # noqa
                st.warning(f'Could not read metadata CSV ({ex}); using sidebar values.')
        rows, bar = [], st.progress(0.0)
        for k, f in enumerate(files):
            m = meta_map.get(f.name)
            a_ = age; s_ = sex; l_ = loc
            if m is not None:
                try: a_ = int(m['age'])
                except Exception: pass  # noqa
                s_ = m['sex'] if m.get('sex') in SEXES else sex
                l_ = m['localization'] if m.get('localization') in LOCALIZATIONS else loc
            d, g = cached_predict(f.getvalue(), build_text(a_, s_, l_), tta)
            pe_ = blend(d, w_res); t = int(pe_.argmax())
            rows.append({'file': f.name, 'age': a_, 'sex': s_, 'location': l_,
                         'prediction': CLASS_NAMES[CLASSES[t]], 'risk': RISK[CLASSES[t]],
                         'confidence': float(pe_[t]), 'P(melanoma)': float(pe_[MEL]),
                         'P(malignant/pre)': p_malignant(pe_),
                         'models agree': bool(d['resnet50'].argmax() == d['efficientnet_b0'].argmax()),
                         'triage': triage(p_malignant(pe_), mode)[0]})
            bar.progress((k + 1) / len(files))
        st.session_state['batch_df'] = pd.DataFrame(rows)
    if isinstance(st.session_state.get('batch_df'), pd.DataFrame):
        bdf = st.session_state['batch_df']
        s1, s2, s3, s4 = st.columns(4)
        s1.metric('Images', len(bdf))
        s2.metric('Malignant', int((bdf['risk'] == 'Malignant').sum()))
        s3.metric('Pre-cancerous', int((bdf['risk'] == 'Pre-cancerous').sum()))
        s4.metric('Referral flags', int((bdf['triage'] == 'Refer').sum()))
        st.dataframe(style_map(bdf.style.format({'confidence': '{:.1%}', 'P(melanoma)': '{:.1%}',
                                                 'P(malignant/pre)': '{:.1%}'}), risk_color, ['risk']),
                     use_container_width=True)
        cnt = bdf['prediction'].value_counts().reset_index()
        cnt.columns = ['prediction', 'count']
        st.altair_chart(alt.Chart(cnt).mark_bar(color='#6366f1').encode(
            x=alt.X('count:Q'), y=alt.Y('prediction:N', sort='-x', title=None)).properties(height=220),
            use_container_width=True)
        st.download_button('⬇️ Download results (CSV)', bdf.to_csv(index=False), 'batch_results.csv', 'text/csv')

# =============================================================================
# TAB 5 - PERFORMANCE
# =============================================================================
with tabs[4]:
    res_path = os.path.join(BASE, 'models', 'results.csv')
    cm_path = os.path.join(BASE, 'models', 'confusion_matrix.png')
    if os.path.exists(res_path):
        res = pd.read_csv(res_path, index_col=0)
        st.markdown('### Test-set comparison of all trained models')
        st.dataframe(res.style.format('{:.3f}').highlight_max(axis=0), use_container_width=True)
        metric = st.selectbox('Chart metric', list(res.columns), index=list(res.columns).index('BalancedAcc')
                              if 'BalancedAcc' in res.columns else 0)
        cd = res[metric].reset_index()
        cd.columns = ['Model', metric]
        st.altair_chart(alt.Chart(cd).mark_bar(cornerRadiusEnd=5).encode(
            y=alt.Y('Model:N', sort='-x', title=None), x=alt.X(f'{metric}:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.Model == 'ENSEMBLE', alt.value('#10b981'), alt.value('#6366f1')),
            tooltip=['Model', alt.Tooltip(f'{metric}:Q', format='.3f')]).properties(height=60 + 38 * len(cd)),
            use_container_width=True)
        st.caption('Balanced accuracy, macro-F1 and melanoma recall matter more than accuracy here: ~67% of the '
                   'data are benign nevi, so plain accuracy is misleading.')
    else:
        st.info('results.csv not found – run train.py first.')
    if os.path.exists(cm_path):
        st.markdown('### Confusion matrix – ensemble')
        st.markdown(f'<img src="data:image/png;base64,'
                    f'{base64.b64encode(open(cm_path, "rb").read()).decode()}" style="width:560px;max-width:100%">',
                    unsafe_allow_html=True)

# =============================================================================
# TAB 6 - HISTORY
# =============================================================================
with tabs[5]:
    hist = pd.DataFrame(st.session_state.history)
    if hist.empty:
        st.info('Predictions made in this session will be listed here.')
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric('Analyses this session', len(hist))
        h2.metric('Malignant predictions', int((hist['risk'] == 'Malignant').sum()))
        h3.metric('Mean confidence', f'{hist.confidence.mean()*100:.0f}%')
        st.dataframe(style_map(hist.style.format({'confidence': '{:.1%}', 'P(melanoma)': '{:.1%}',
                                                  'P(malignant/pre)': '{:.1%}'}), risk_color, ['risk']),
                     use_container_width=True)
        d1, d2 = st.columns(2)
        d1.download_button('⬇️ Download history (CSV)', hist.to_csv(index=False), 'history.csv', 'text/csv')
        if d2.button('🗑️ Clear history'):
            st.session_state.history, st.session_state.last_key = [], None
            st.rerun()

# =============================================================================
# TAB 7 - ABOUT
# =============================================================================
with tabs[6]:
    st.markdown('### Model cards')
    st.dataframe(pd.DataFrame({
        'Model': [NAMES[b] for b in NAMES],
        'Image backbone': ['ResNet50', 'EfficientNet-B0'],
        'Text branch': ['DistilBERT'] * 2,
        'Fusion': ['Cross-modal attention + gate'] * 2,
        'Parameters (M)': [round(n_params[b], 1) for b in NAMES],
        'Ensemble weight': [round(w_res, 2), round(1 - w_res, 2)]}), use_container_width=True, hide_index=True)
    st.graphviz_chart(arch_dot(), use_container_width=True)
    st.markdown("""
**Dataset:** HAM10000 – 10,015 dermoscopic images, 7 diagnostic classes, with age, sex and lesion site.

**Fusion:** text tokens attend to 7×7 image regions and vice-versa; a learned gate weighs image vs metadata
per patient; an interaction term (image ⊙ text) and the global CNN vector are added before the classifier.

**Training pipeline:** lesion-level split (no leakage) → augmentation → class-weighted loss → mixed precision →
validation-tuned ensemble weight → balanced accuracy / macro-F1 / AUC / melanoma recall → Grad-CAM.

**App extras:** test-time augmentation, screening-mode triage, certainty (entropy), metadata-influence analysis,
what-if sweeps (location, age, sex, age×location), batch analysis with optional metadata CSV, session history
and a downloadable HTML report.

**Limitations:** only 3 metadata fields, a single public dataset, and images unlike HAM10000 (phone photos,
other devices) may be predicted unreliably. Not validated clinically.
""")
    st.error('Not a medical device. Never use this tool to make or delay a medical decision – consult a '
             'qualified dermatologist.')