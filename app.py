import streamlit as st
import os, uuid, json, dataclasses
from PIL import Image
from database import (
    init_db, register_user, login_user, validate_token, logout_user,
    save_query, get_user_queries, delete_query,
    save_rag_search, get_user_rag_searches, delete_rag_search,
)
from model_inference import QUESTION_TYPES, GRADE_LABELS

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(page_title="DeepEye · DME Analysis", page_icon="👁️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0d0f18;color:#dde1f0}
section[data-testid="stSidebar"]{background:#11131f!important;border-right:1px solid #1e2235}
.card{background:#161929;border:1px solid #1e2235;border-radius:14px;padding:1.3rem 1.5rem;margin-bottom:.9rem}
.answer-box{background:#0e1e33;border-left:4px solid #4a90f5;border-radius:10px;padding:1rem 1.4rem;margin:.6rem 0;line-height:1.75}
.answer-yes{background:#0e2a1a;border-left:4px solid #4fcf7a;border-radius:10px;padding:1rem 1.4rem;margin:.6rem 0}
.answer-no{background:#2a0e0e;border-left:4px solid #cf4f4f;border-radius:10px;padding:1rem 1.4rem;margin:.6rem 0}
.answer-grade{background:#1a1a0e;border-left:4px solid #cfb84f;border-radius:10px;padding:1rem 1.4rem;margin:.6rem 0}
.section-header{background:#131626;border:1px solid #1e2235;border-radius:10px;padding:.6rem 1rem;margin:.5rem 0;font-weight:700;color:#4a90f5}
.qtype-card{background:#131626;border:2px solid #1e2235;border-radius:12px;padding:1rem;margin:.4rem 0;cursor:pointer;transition:border .2s}
.qtype-card:hover{border-color:#4a90f5}
.qtype-card.selected{border-color:#4a90f5;background:#0e1e33}
.tag{display:inline-block;background:#1a2e1a;color:#6fcf6f;border:1px solid #2e4a2e;padding:2px 9px;border-radius:20px;font-size:.76rem;margin:2px 3px}
.grade-0{color:#6fcf6f;font-weight:700;font-size:1.1rem}
.grade-1{color:#cfb84f;font-weight:700;font-size:1.1rem}
.grade-2{color:#cf4f4f;font-weight:700;font-size:1.1rem}
.score-badge{display:inline-block;background:linear-gradient(135deg,#4a90f5,#7c5cbf);color:#fff;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:700}
.disclaimer-box{background:#1a1410;border:2px solid #c0882a;border-radius:12px;padding:1rem 1.4rem;margin:1rem 0}
.conf-high{color:#6fcf6f;font-weight:700}
.conf-mid{color:#f0c060;font-weight:700}
.conf-low{color:#f07070;font-weight:700}
.mask-info{background:#0e1a2e;border:1px solid #1e3a5a;border-radius:10px;padding:.7rem 1rem;margin:.5rem 0;font-size:.85rem;color:#7ab0d4}
.ts{color:#5a6080;font-size:.78rem}
h1,h2,h3,h4{color:#dde1f0}
#MainMenu,footer,header{visibility:hidden}
.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:#161929!important;border:1px solid #1e2235!important;color:#dde1f0!important;border-radius:8px!important}
.stButton>button{background:linear-gradient(135deg,#4a90f5,#7c5cbf);color:#fff;border:none;border-radius:8px;font-weight:600;width:100%}
.stButton>button:hover{opacity:.88;transform:translateY(-1px)}
.stTabs [data-baseweb="tab"]{color:#8090b8;font-weight:600}
.stTabs [aria-selected="true"]{color:#4a90f5!important;border-bottom:2px solid #4a90f5}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Init
# ─────────────────────────────────────────
init_db()
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

for k, v in [("token",None),("user_id",None),("username",None),
             ("vqa_result",None),("rag_result",None),("selected_qtype","grade")]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.token:
    valid, uid, uname = validate_token(st.session_state.token)
    if valid:
        st.session_state.user_id  = uid
        st.session_state.username = uname
    else:
        for k in ["token","user_id","username"]:
            st.session_state[k] = None

def save_upload(f, uid):
    ext  = f.name.split(".")[-1].lower()
    name = f"u{uid}_{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path,"wb") as fp: fp.write(f.getbuffer())
    return path

def grade_class(g):
    return {0:"grade-0",1:"grade-1",2:"grade-2"}.get(int(g) if g.isdigit() else 0, "grade-0")

def conf_class(level):
    l = level.lower()
    if "high" in l:  return "conf-high"
    if "low" in l:   return "conf-low"
    return "conf-mid"

# ─────────────────────────────────────────
# Auth
# ─────────────────────────────────────────
def show_auth():
    _, col, _ = st.columns([1,1.4,1])
    with col:
        st.markdown("<br><br>",unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align:center;margin-bottom:2rem'>
                <span style='font-size:3.5rem'>👁️</span>
                <h1 style='margin:.3rem 0 0;color:#4a90f5;letter-spacing:2px'>DeepEye</h1>
                <p style='color:#5a6080;margin:0'>Diabetic Macular Edema · VQA + RAG</p>
            </div>""", unsafe_allow_html=True)
        tl, tr = st.tabs(["🔑 Login","📝 Register"])
        with tl:
            st.markdown("<br>",unsafe_allow_html=True)
            u = st.text_input("Username", key="lu", placeholder="Username")
            p = st.text_input("Password", type="password", key="lp", placeholder="Password")
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("Login", key="btn_l"):
                if not u or not p: st.error("Fill all fields.")
                else:
                    ok,res,uid = login_user(u,p)
                    if ok: st.session_state.update(token=res,user_id=uid,username=u); st.rerun()
                    else:  st.error(res)
        with tr:
            st.markdown("<br>",unsafe_allow_html=True)
            ru  = st.text_input("Username", key="ru", placeholder="Min 3 chars")
            rp  = st.text_input("Password", type="password", key="rp", placeholder="Min 6 chars")
            rp2 = st.text_input("Confirm",  type="password", key="rp2")
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("Create Account", key="btn_r"):
                if not ru or not rp or not rp2: st.error("Fill all fields.")
                elif len(ru)<3: st.error("Username ≥ 3 chars.")
                elif len(rp)<6: st.error("Password ≥ 6 chars.")
                elif rp!=rp2:   st.error("Passwords don't match.")
                else:
                    ok,msg = register_user(ru,rp)
                    st.success(msg) if ok else st.error(msg)

# ─────────────────────────────────────────
# Main app
# ─────────────────────────────────────────
def show_app():
    uid  = st.session_state.user_id
    user = st.session_state.username

    with st.sidebar:
        st.markdown(f"""
            <div style='text-align:center;padding:1.2rem 0 .5rem'>
                <div style='font-size:2.8rem'>👁️</div>
                <h2 style='color:#4a90f5;margin:.2rem 0'>DeepEye</h2>
                <p style='color:#5a6080;font-size:.8rem;margin:0'>DME Clinical AI</p>
                <hr style='border-color:#1e2235;margin:.8rem 0'>
                <p style='color:#aab0cc'>👤 <b>{user}</b></p>
            </div>""", unsafe_allow_html=True)

        nav = st.radio("", [
            "🔬 VQA — DME Analysis",
            "🗂️ Multimodal RAG Search",
            "📋 VQA History",
            "🔍 RAG History",
        ], label_visibility="collapsed")

        st.markdown("<br>",unsafe_allow_html=True)
        from rag_engine import index_status
        status = index_status()
        if status["exists"]:
            st.success(f"✅ RAG Index: {status['n_total']:,} images")
        else:
            st.warning("⚠️ RAG index not built")
            if st.button("🔨 Build RAG Index"):
                with st.spinner("Building index (uses unique images from trainqa.json)..."):
                    try:
                        from rag_engine import build_index
                        build_index(force=True)
                        st.success("Done!"); st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.markdown("<br><br>",unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            logout_user(st.session_state.token)
            for k in ["token","user_id","username","vqa_result","rag_result"]:
                st.session_state[k] = None
            st.rerun()

    if nav == "🔬 VQA — DME Analysis":     vqa_page(uid)
    elif nav == "🗂️ Multimodal RAG Search": rag_page(uid)
    elif nav == "📋 VQA History":           vqa_history(uid)
    else:                                   rag_history(uid)


# ─────────────────────────────────────────
# VQA Page — DME structured questions
# ─────────────────────────────────────────
def vqa_page(uid):
    st.markdown("## 🔬 DME Visual Question Answering")
    st.markdown(
        "Upload a retinal fundus image and select a clinical question. "
        "The model answers using the 5 question types it was trained on."
    )
    st.markdown("---")

    # ── Question type selector ────────────
    st.markdown("### 1️⃣ Select Question Type")

    # Display as clickable cards in a grid
    qt_cols = st.columns(3)
    qt_keys = list(QUESTION_TYPES.keys())
    qt_labels = {
        "whole"           : ("🖼️", "Whole Image", "Hard exudates in image?"),
        "fovea"           : ("🎯", "Fovea Region", "Hard exudates in fovea?"),
        "grade"           : ("📊", "DME Grade",    "Severity: 0 / 1 / 2"),
        "inside_exudates" : ("🔍", "Region: Exudates", "Hard exudates in region?"),
        "inside_optic"    : ("⭕", "Region: Optic Disc", "Optic disc in region?"),
    }
    for i, key in enumerate(qt_keys):
        icon, title, desc = qt_labels[key]
        selected = st.session_state.selected_qtype == key
        border   = "#4a90f5" if selected else "#1e2235"
        bg       = "#0e1e33" if selected else "#131626"
        with qt_cols[i % 3]:
            st.markdown(f"""
                <div style='background:{bg};border:2px solid {border};border-radius:12px;
                            padding:.9rem 1rem;margin:.3rem 0;text-align:center'>
                    <div style='font-size:1.5rem'>{icon}</div>
                    <div style='font-weight:700;color:#dde1f0;font-size:.9rem'>{title}</div>
                    <div style='color:#7a8099;font-size:.75rem;margin-top:3px'>{desc}</div>
                </div>""", unsafe_allow_html=True)
            if st.button(f"Select", key=f"sel_{key}",
                         help=QUESTION_TYPES[key]["description"]):
                st.session_state.selected_qtype = key
                st.session_state.vqa_result     = None
                st.rerun()

    qt_key  = st.session_state.selected_qtype
    qt_info = QUESTION_TYPES[qt_key]

    # Show selected question
    st.markdown(f"""
        <div class='mask-info'>
            ✅ <b>Selected question:</b> &nbsp; <i>"{qt_info['question']}"</i>
            <br>
            {qt_info['description']}
            {'<br>⚠️ <b>This question type requires a region mask.</b>' if qt_info["needs_mask"] else ''}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("### 2️⃣ Upload Image" + (" & Mask" if qt_info["needs_mask"] else ""))

    img_col, mask_col = st.columns([1, 1]) if qt_info["needs_mask"] else (st.columns([1, 1]))

    with img_col:
        st.markdown("**Fundus Image**")
        up = st.file_uploader("Fundus image", type=["jpg","jpeg","png","bmp","tiff"],
                              label_visibility="collapsed", key="vqa_up")
        if up:
            st.image(Image.open(up), caption="Uploaded fundus image", use_container_width=True)

    mask_file = None
    if qt_info["needs_mask"]:
        with mask_col:
            st.markdown("**Region Mask** (binary .tif or .png)")
            mask_file = st.file_uploader("Region mask", type=["tif","tiff","png","jpg"],
                                         label_visibility="collapsed", key="vqa_mask")
            if mask_file:
                mask_img = Image.open(mask_file).convert("L")
                st.image(mask_img, caption="Uploaded mask", use_container_width=True)
            else:
                st.markdown("""
                    <div class='mask-info'>
                        Upload a binary mask (.tif) where white pixels mark the region to analyse.
                        Without a mask the full image will be used as fallback.
                    </div>""", unsafe_allow_html=True)
    else:
        with mask_col:
            st.markdown("**Expected Answers**")
            if qt_key == "grade":
                st.markdown("""
                    <div class='card' style='padding:.8rem 1rem'>
                        <p class='grade-0'>● Grade 0 — No DME</p>
                        <p class='grade-1'>● Grade 1 — Mild DME</p>
                        <p class='grade-2'>● Grade 2 — Moderate–Severe DME</p>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='card' style='padding:.8rem 1rem'>
                        <p style='color:#6fcf6f;font-weight:700'>✅ Yes — finding present</p>
                        <p style='color:#cf4f4f;font-weight:700'>❌ No — finding absent</p>
                    </div>""", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    go = st.button("🚀 Run Analysis", key="vqa_go")

    if go:
        if not up:
            st.error("Please upload a fundus image.")
        else:
            from guardrails import validate_image
            img = Image.open(up).convert("RGB")
            iv  = validate_image(img, up.size)
            if not iv.passed:
                st.error(f"🚫 {iv.message}")
            else:
                mask_img = Image.open(mask_file).convert("L") if mask_file else None
                with st.spinner(f"🧠 Analysing: \"{qt_info['question']}\""):
                    try:
                        from model_inference import run_inference
                        result = run_inference(img, qt_key, mask=mask_img)
                        ipath  = save_upload(up, uid)
                        # Save to DB: question + answer
                        save_query(uid, ipath, result["question"], result["answer_label"])
                        st.session_state.vqa_result = (img, result, mask_img)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        import traceback; st.code(traceback.format_exc())

    # ── Result panel ─────────────────────
    if st.session_state.vqa_result:
        img, result, mask_img = st.session_state.vqa_result
        st.markdown("---")
        st.markdown("### 💡 Analysis Result")

        r1, r2 = st.columns([1, 1.5], gap="large")

        with r1:
            # Show processed image (with mask applied if relevant)
            proc_img = result.get("processed_img", img)
            caption  = "Image fed to model (mask applied)" if result["mask_applied"] else "Analysed image"
            st.image(proc_img, caption=caption, use_container_width=True)
            if mask_img and result["mask_applied"]:
                st.image(mask_img, caption="Region mask used", use_container_width=True)

        with r2:
            raw = result["raw_answer"]
            label = result["answer_label"]

            # Style answer box based on answer
            if result["question_type"] == "grade":
                box_class = "answer-grade"
                gc = {"0":"grade-0","1":"grade-1","2":"grade-2"}.get(raw,"grade-0")
                answer_html = f"<p class='{gc}' style='font-size:1.3rem;margin:0'>{label}</p>"
            elif raw == "yes":
                box_class   = "answer-yes"
                answer_html = f"<p style='color:#6fcf6f;font-size:1.3rem;font-weight:700;margin:0'>{label}</p>"
            else:
                box_class   = "answer-no"
                answer_html = f"<p style='color:#cf4f4f;font-size:1.3rem;font-weight:700;margin:0'>{label}</p>"

            st.markdown(f"""
                <div class='card'>
                    <p style='color:#5a6080;margin:0 0 4px;font-size:.8rem'>Question</p>
                    <p style='font-weight:600;margin:0'>{result['question']}</p>
                </div>
                <div class='{box_class}'>
                    <p style='color:#8090b8;font-weight:700;margin:0 0 6px;font-size:.8rem'>
                        🤖 MODEL ANSWER
                    </p>
                    {answer_html}
                </div>
                <div style='margin:.5rem 0;color:#5a6080;font-size:.78rem'>
                    Question type: <b>{result['question_type']}</b> &nbsp;·&nbsp;
                    Mask applied: <b>{'Yes' if result['mask_applied'] else 'No'}</b>
                </div>""", unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
                <div class='disclaimer-box'>
                    <p style='color:#c0882a;font-weight:700;margin:0 0 4px'>⚠️ Medical Disclaimer</p>
                    <p style='color:#d4a85a;margin:0;font-size:.85rem'>
                        This is an <b>AI-generated</b> result for informational purposes only.
                        It does <b>not</b> constitute a medical diagnosis.
                        For clinical confirmation and treatment decisions, please consult a
                        qualified <b>ophthalmologist</b>.
                    </p>
                </div>""", unsafe_allow_html=True)
            st.success("✅ Saved to VQA history")


# ─────────────────────────────────────────
# RAG Page
# ─────────────────────────────────────────
def rag_page(uid):
    st.markdown("## 🗂️ Multimodal RAG — Similar Case Search")
    st.markdown(
        "Upload a retinal image to find the most similar DME cases in the dataset. "
        "Select a question type to filter results and generate a RAG-augmented clinical answer."
    )
    st.markdown("---")

    from rag_engine import index_status
    if not index_status()["exists"]:
        st.warning("⚠️ RAG index not built. Click **Build RAG Index** in the sidebar first.")
        return

    c1, c2 = st.columns([1,1], gap="large")

    with c1:
        st.markdown("### 📷 Query Image")
        rup = st.file_uploader("Image", type=["jpg","jpeg","png","bmp","tiff"],
                               label_visibility="collapsed", key="rag_up")
        if rup:
            st.image(Image.open(rup), use_container_width=True)

        # Mask for inside question types
        st.markdown("### 🗺️ Region Mask (optional, for region questions)")
        rmask = st.file_uploader("Mask", type=["tif","tiff","png","jpg"],
                                 label_visibility="collapsed", key="rag_mask")
        if rmask:
            st.image(Image.open(rmask).convert("L"), caption="Region mask", use_container_width=True)

    with c2:
        st.markdown("### ⚙️ Settings")

        # Question type selector
        qt_options = {
            "grade"           : "📊 DME Grade (0/1/2)",
            "whole"           : "🖼️ Hard exudates in image?",
            "fovea"           : "🎯 Hard exudates in fovea?",
            "inside_exudates" : "🔍 Hard exudates in region?",
            "inside_optic"    : "⭕ Optic disc in region?",
        }
        rag_qt = st.selectbox("Question type", options=list(qt_options.keys()),
                              format_func=lambda k: qt_options[k], key="rag_qt")

        topk  = st.slider("Top-K similar cases", 1, 10, 5, key="rag_k")
        alpha = st.slider("Image ↔ Text weight", 0.0, 1.0, 0.7, 0.05,
                          help="1.0 = image only · 0.0 = question text only", key="rag_alpha")
        gen   = st.checkbox("Generate RAG-augmented answer via LLaVA", value=True, key="rag_gen")

        st.markdown("<br>",unsafe_allow_html=True)
        go = st.button("🔍 Search & Analyse", key="rag_go")

    if go:
        if not rup:
            st.error("Upload an image first.")
        else:
            with st.spinner("🔎 Running RAG pipeline..."):
                try:
                    from guardrails import validate_image
                    img      = Image.open(rup).convert("RGB")
                    mask_img = Image.open(rmask).convert("L") if rmask else None

                    iv = validate_image(img, rup.size)
                    if not iv.passed:
                        st.error(f"🚫 {iv.message}"); return

                    if gen:
                        from rag_engine import rag_pipeline
                        out, retrieved, err = rag_pipeline(
                            user_id=uid, image=img, question_type=rag_qt,
                            file_size_b=rup.size, top_k=topk, alpha=alpha, mask=mask_img,
                        )
                        if err:
                            st.error(f"🚫 {err}"); return
                    else:
                        from rag_engine import search
                        filter_map = {"whole":"whole","fovea":"fovea","grade":"grade",
                                      "inside_exudates":"inside","inside_optic":"inside"}
                        retrieved = search(img, QUESTION_TYPES[rag_qt]["question"],
                                          top_k=topk, alpha=alpha,
                                          filter_type=filter_map.get(rag_qt))
                        out = None

                    ipath = save_upload(rup, uid)
                    save_rag_search(
                        uid, ipath, QUESTION_TYPES[rag_qt]["question"],
                        json.dumps([{"image_name":r["image_name"],"grade":r.get("grade"),
                                     "has_exudates":r.get("has_exudates"),"fovea":r.get("fovea"),
                                     "diseases":r.get("diseases",""),"score":r["score"]}
                                    for r in retrieved]),
                        json.dumps(dataclasses.asdict(out)) if out else ""
                    )
                    st.session_state.rag_result = (img, retrieved, out, rag_qt, mask_img)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback; st.code(traceback.format_exc())

    if st.session_state.rag_result:
        img, retrieved, out, rag_qt, mask_img = st.session_state.rag_result
        qt_info = QUESTION_TYPES[rag_qt]
        st.markdown("---")

        if out:
            st.markdown("## 🧠 RAG Analysis Report")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Similar Cases",  out.similar_cases_count)
            m2.metric("Retrieval",      out.retrieval_quality)
            m3.metric("Confidence",     out.confidence_level)
            m4.metric("Time",           f"{out.processing_time_s}s")

            st.markdown("<br>",unsafe_allow_html=True)

            st.markdown("<div class='section-header'>🔍 Primary Finding</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='answer-box'><p style='margin:0'>{out.primary_finding}</p></div>",
                        unsafe_allow_html=True)

            if out.possible_conditions:
                st.markdown("<div class='section-header'>🏷️ Conditions in Similar Cases</div>",
                            unsafe_allow_html=True)
                tags = "".join(f"<span class='tag'>{c}</span>" for c in out.possible_conditions)
                st.markdown(f"<div class='card' style='padding:.7rem 1rem'>{tags}</div>",
                            unsafe_allow_html=True)

            if out.clinical_description:
                st.markdown("<div class='section-header'>📋 Clinical Description</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='answer-box'><p style='margin:0'>{out.clinical_description}</p></div>",
                            unsafe_allow_html=True)

            ca, cb = st.columns(2)
            with ca:
                st.markdown("<div class='section-header'>📊 Confidence</div>",
                            unsafe_allow_html=True)
                cc = conf_class(out.confidence_level)
                st.markdown(f"""
                    <div class='card' style='padding:.8rem 1rem'>
                        <p class='{cc}' style='margin:0 0 6px;font-size:1.1rem'>{out.confidence_level}</p>
                        <p style='margin:0;color:#aab0cc;font-size:.85rem'>{out.confidence_note}</p>
                    </div>""", unsafe_allow_html=True)
            with cb:
                st.markdown("<div class='section-header'>📚 Retrieval</div>",
                            unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='card' style='padding:.8rem 1rem'>
                        <p style='color:#4a90f5;font-weight:700;margin:0 0 6px;font-size:1.1rem'>
                            {out.retrieval_quality}
                        </p>
                        <p style='margin:0;color:#aab0cc;font-size:.85rem'>
                            {out.similar_cases_count} similar DME cases retrieved
                        </p>
                    </div>""", unsafe_allow_html=True)

            if out.safety_flags:
                tags = "".join(f"<span style='background:#2e1a1a;color:#f07070;border:1px solid #4a2e2e;"
                               f"padding:2px 9px;border-radius:20px;font-size:.76rem;margin:2px'>"
                               f"⚠ {f}</span>" for f in out.safety_flags)
                st.markdown(f"<div class='section-header'>🛡️ Safety Flags</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='card' style='padding:.7rem 1rem'>{tags}</div>",
                            unsafe_allow_html=True)

            st.markdown(f"""
                <div class='disclaimer-box'>
                    <p style='color:#c0882a;font-weight:700;font-size:1rem;margin:0 0 6px'>
                        ⚠️ AI-Generated Analysis — Medical Disclaimer
                    </p>
                    <p style='color:#d4a85a;margin:0;font-size:.88rem;line-height:1.6'>
                        This report is <b>AI-generated</b> for <b>informational and research purposes only</b>.
                        It does <b>not</b> constitute a medical diagnosis, clinical opinion, or treatment
                        recommendation. The model may produce inaccurate findings.<br><br>
                        🩺 <b>For clinical confirmation, please consult a qualified
                        <u>ophthalmologist</u> or retinal specialist.</b>
                    </p>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

        # Retrieved cases
        st.markdown(f"### 📚 Top {len(retrieved)} Similar DME Cases")
        st.markdown(f"*Question type: **{qt_info['question']}***")

        for i, rec in enumerate(retrieved):
            rc1, rc2 = st.columns([0.7,2.5], gap="medium")
            with rc1:
                ip = rec.get("image_path","")
                if ip and os.path.exists(ip):
                    st.image(Image.open(ip), use_container_width=True)
                else:
                    st.markdown("<div style='background:#1a1d2e;height:90px;border-radius:8px;"
                                "display:flex;align-items:center;justify-content:center;"
                                "color:#5a6080'>No image</div>", unsafe_allow_html=True)
            with rc2:
                score_pct = f"{rec.get('score',0)*100:.1f}%"
                grade     = rec.get("grade")
                exudates  = rec.get("has_exudates")
                fovea     = rec.get("fovea")
                img_name  = rec.get("image_name","—")

                grade_html = ""
                if grade is not None:
                    gc = {"0":"grade-0","1":"grade-1","2":"grade-2"}.get(str(grade),"grade-0")
                    gl = GRADE_LABELS.get(str(grade), f"Grade {grade}")
                    grade_html = f"<p class='{gc}' style='margin:4px 0;font-size:.9rem'>📊 {gl}</p>"

                findings = []
                if exudates: findings.append(f"Hard exudates: <b>{exudates}</b>")
                if fovea:    findings.append(f"Fovea: <b>{fovea}</b>")
                findings_html = " &nbsp;·&nbsp; ".join(findings) if findings else ""

                # Show matching QA pairs for the selected question type
                filter_map = {"whole":"whole","fovea":"fovea","grade":"grade",
                              "inside_exudates":"inside","inside_optic":"inside"}
                ft = filter_map.get(rag_qt,"")
                matching_qa = [p for p in rec.get("qa_pairs",[]) if p["question_type"]==ft][:2]
                qa_html = ""
                for pair in matching_qa:
                    ans_color = "#6fcf6f" if str(pair["answer"])=="yes" else "#cf4f4f" if str(pair["answer"])=="no" else "#cfb84f"
                    qa_html += (f"<p style='margin:2px 0;font-size:.82rem;color:#aab0cc'>"
                                f"Q: {pair['question']} "
                                f"<span style='color:{ans_color};font-weight:700'>→ {pair['answer']}</span></p>")

                st.markdown(f"""
                    <div style='background:#131626;border:1px solid #1e2235;border-radius:10px;
                                padding:.85rem 1rem;margin-bottom:.6rem'>
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                            <span style='font-weight:700;color:#dde1f0'>#{i+1} · {img_name}</span>
                            <span class='score-badge'>Similarity: {score_pct}</span>
                        </div>
                        {grade_html}
                        <p style='color:#7a8099;font-size:.8rem;margin:3px 0'>{findings_html}</p>
                        {qa_html}
                    </div>""", unsafe_allow_html=True)
            st.markdown("<hr style='border-color:#1e2235;margin:.3rem 0'>",
                        unsafe_allow_html=True)


# ─────────────────────────────────────────
# VQA History
# ─────────────────────────────────────────
def vqa_history(uid):
    st.markdown("## 📋 VQA History")
    st.markdown("---")
    rows = get_user_queries(uid)
    if not rows:
        st.markdown("<div style='text-align:center;color:#5a6080;padding:3rem'>"
                    "<div style='font-size:3rem'>📭</div><p>No VQA queries yet.</p></div>",
                    unsafe_allow_html=True)
        return
    st.markdown(f"**{len(rows)} queries**")
    for q in rows:
        h1,h2,h3 = st.columns([0.7,2.8,.25])
        with h1:
            if os.path.exists(q["image_path"]):
                st.image(Image.open(q["image_path"]), use_container_width=True)
        with h2:
            ts  = q["created_at"][:16].replace("T"," ")
            ans = q["answer"]
            if "Grade 0" in ans or "No DME" in ans:    ans_color="#6fcf6f"
            elif "Grade 1" in ans or "Mild" in ans:    ans_color="#cfb84f"
            elif "Grade 2" in ans or "Severe" in ans:  ans_color="#cf4f4f"
            elif "Yes" in ans:                          ans_color="#6fcf6f"
            else:                                       ans_color="#cf4f4f"
            st.markdown(f"""
                <div class='card' style='padding:.8rem 1rem'>
                    <span class='ts'>🕒 {ts}</span>
                    <p style='font-weight:600;color:#dde1f0;margin:4px 0 2px'>❓ {q['question']}</p>
                    <p style='color:{ans_color};font-weight:700;margin:0'>→ {q['answer']}</p>
                </div>""", unsafe_allow_html=True)
        with h3:
            if st.button("🗑️", key=f"dv_{q['id']}"):
                delete_query(q["id"], uid); st.rerun()
        st.markdown("<hr style='border-color:#1e2235;margin:.3rem 0'>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────
# RAG History
# ─────────────────────────────────────────
def rag_history(uid):
    st.markdown("## 🔍 RAG Search History")
    st.markdown("---")
    rows = get_user_rag_searches(uid)
    if not rows:
        st.markdown("<div style='text-align:center;color:#5a6080;padding:3rem'>"
                    "<div style='font-size:3rem'>📭</div><p>No RAG searches yet.</p></div>",
                    unsafe_allow_html=True)
        return
    st.markdown(f"**{len(rows)} searches**")
    for s in rows:
        s1,s2,s3 = st.columns([0.7,2.8,.25])
        with s1:
            if os.path.exists(s["image_path"]):
                st.image(Image.open(s["image_path"]), use_container_width=True)
        with s2:
            ts      = s["created_at"][:16].replace("T"," ")
            results = json.loads(s["results_json"]) if s["results_json"] else []
            conf    = "—"
            if s.get("structured_json"):
                try: conf = json.loads(s["structured_json"]).get("confidence_level","—")
                except: pass
            grades = [str(r.get("grade","")) for r in results if r.get("grade") is not None]
            grade_summary = ", ".join(f"Grade {g}" for g in set(grades)) if grades else "—"
            st.markdown(f"""
                <div class='card' style='padding:.8rem 1rem'>
                    <span class='ts'>🕒 {ts}</span>
                    <p style='font-weight:600;color:#dde1f0;margin:4px 0 2px'>
                        🔍 {s['query_text'] or '(image only)'}
                    </p>
                    <p style='color:#aab0cc;font-size:.85rem;margin:0'>
                        DME grades found: {grade_summary} &nbsp;·&nbsp;
                        {len(results)} cases &nbsp;·&nbsp; Confidence: {conf}
                    </p>
                </div>""", unsafe_allow_html=True)
        with s3:
            if st.button("🗑️", key=f"dr_{s['id']}"):
                delete_rag_search(s["id"], uid); st.rerun()
        st.markdown("<hr style='border-color:#1e2235;margin:.3rem 0'>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────
# Router
# ─────────────────────────────────────────
if st.session_state.token and st.session_state.user_id:
    show_app()
else:
    show_auth()
