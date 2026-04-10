"""
Entry point — Login / Sign Up
"""

import streamlit as st
from utils.auth import login, signup
from utils.google_auth import google_configured, render_google_button, process_google_result, save_profile
from utils.styles import inject_styles
from data.sample_data import ROLE_CATEGORIES

st.set_page_config(page_title="GapGenius", page_icon="🎓", layout="centered")
inject_styles()

if st.session_state.get("user"):
    st.switch_page("pages/1_Dashboard.py")

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:36px 0 10px;">
  <div style="font-size:3.5rem; margin-bottom:8px;">🎓</div>
  <h1 style="font-size:2.8rem; margin:0; background:linear-gradient(135deg,#667eea,#764ba2);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">GapGenius</h1>
  <p style="font-size:1.05rem; color:#666; margin-top:10px; margin-bottom:0;">
    Upload your resume &nbsp;·&nbsp; Discover skill gaps &nbsp;·&nbsp; AI-powered career guidance
  </p>
</div>
""", unsafe_allow_html=True)

# Feature pills
st.markdown("""
<div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin:16px 0 28px;">
  <span style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;
               border-radius:20px;padding:5px 16px;font-size:0.82rem;font-weight:600;">
    🤖 AI Career Coach
  </span>
  <span style="background:linear-gradient(135deg,#11998e,#38ef7d);color:white;
               border-radius:20px;padding:5px 16px;font-size:0.82rem;font-weight:600;">
    📊 Skill Gap Analysis
  </span>
  <span style="background:linear-gradient(135deg,#f7971e,#ffd200);color:white;
               border-radius:20px;padding:5px 16px;font-size:0.82rem;font-weight:600;">
    🎤 Voice Emotion Detection
  </span>
  <span style="background:linear-gradient(135deg,#ee0979,#ff6a00);color:white;
               border-radius:20px;padding:5px 16px;font-size:0.82rem;font-weight:600;">
    📚 Course Recommendations
  </span>
</div>
""", unsafe_allow_html=True)

tab_login, tab_signup = st.tabs(["🔐  Login", "✨  Create Account"])

# ── LOGIN ──────────────────────────────────────────────────────────────────
with tab_login:
    st.markdown("<br>", unsafe_allow_html=True)
    if google_configured():
        result = render_google_button(key="google_login")
        if result:
            user = process_google_result(result)
            if user:
                st.session_state.user = user
                st.session_state.chat_history = []
                st.session_state.analysis_result = None
                st.switch_page("pages/0_Complete_Profile.py" if user.get("needs_profile") else "pages/1_Dashboard.py")

        st.markdown("""
<div style="display:flex;align-items:center;margin:14px 0;">
  <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,#ccc);"></div>
  <span style="padding:0 14px;color:#999;font-size:0.85rem;font-weight:500;">or sign in with email</span>
  <div style="flex:1;height:1px;background:linear-gradient(90deg,#ccc,transparent);"></div>
</div>""", unsafe_allow_html=True)

    with st.form("login_form"):
        email    = st.text_input("📧 Email", placeholder="you@example.com")
        password = st.text_input("🔒 Password", type="password")
        submitted = st.form_submit_button("Login  →", use_container_width=True, type="primary")

    if submitted:
        if not email or not password:
            st.error("Please fill in all fields.")
        else:
            ok, user, msg = login(email, password)
            if ok:
                st.session_state.user = user
                st.session_state.chat_history = []
                st.session_state.analysis_result = None
                st.switch_page("pages/1_Dashboard.py")
            else:
                st.error(msg)

# ── SIGN UP ────────────────────────────────────────────────────────────────
with tab_signup:
    st.markdown("<br>", unsafe_allow_html=True)
    if google_configured():
        result_s = render_google_button(key="google_signup")
        if result_s:
            user = process_google_result(result_s)
            if user:
                st.session_state.user = user
                st.session_state.chat_history = []
                st.session_state.analysis_result = None
                st.switch_page("pages/0_Complete_Profile.py")

        st.markdown("""
<div style="display:flex;align-items:center;margin:14px 0;">
  <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,#ccc);"></div>
  <span style="padding:0 14px;color:#999;font-size:0.85rem;font-weight:500;">or create account with email</span>
  <div style="flex:1;height:1px;background:linear-gradient(90deg,#ccc,transparent);"></div>
</div>""", unsafe_allow_html=True)

    with st.form("signup_form"):
        col1, col2 = st.columns(2)
        with col1:
            name       = st.text_input("👤 Full Name",  placeholder="Alex Johnson")
            email_s    = st.text_input("📧 Email",      placeholder="you@example.com")
            password_s = st.text_input("🔒 Password",  type="password", help="At least 6 characters")
        with col2:
            experience   = st.selectbox("💼 Experience Level",
                ["Student / Fresher","0-1 years","1-3 years","3-5 years","5+ years"])
            current_role = st.text_input("🏷️ Current Role / Degree", placeholder="e.g. B.Sc Chemistry Student")
            field        = st.selectbox("🌐 Field / Domain", list(ROLE_CATEGORIES.keys()))
            target_role  = st.selectbox("🎯 Target Job Role", ROLE_CATEGORIES[field])

        agreed    = st.checkbox("I agree to the terms of use")
        submitted_s = st.form_submit_button("Create Account  →", use_container_width=True, type="primary")

    if submitted_s:
        if not all([name, email_s, password_s, current_role]):
            st.error("Please fill in all fields.")
        elif len(password_s) < 6:
            st.error("Password must be at least 6 characters.")
        elif not agreed:
            st.error("Please accept the terms to continue.")
        else:
            ok, msg = signup(name, email_s, password_s, current_role, target_role, experience, field)
            if ok:
                st.success(msg + " Please log in.")
            else:
                st.error(msg)

st.divider()
st.markdown("<p style='text-align:center;color:#aaa;font-size:0.8rem;'>GapGenius — Resume Skill Gap Analyzer | SRMIST Project</p>",
            unsafe_allow_html=True)
