import streamlit as st
import pandas as pd
import os
import time
import json
import zipfile
import io
import shutil
import re
import random
from datetime import datetime
from config import (
    JOB_REQUIREMENT, DATA_SAVE_CONFIG, UI_CONFIG,
    EXPERIMENT_STAGES, STAY_TIME_CONFIG, SORT_OPTIONS,
    RATING_WEIGHTS, BIAS_CONFIG, JOB_DESCRIPTION,
    ALGORITHM_LITERACY_ITEMS, ALGORITHM_DEPENDENCY_ITEMS,
    FIXED_PRESSURE_CONDITION,
    RESUME_FOLDER, PHOTO_FOLDER          # 新增导入
)
from core_rating import (
    read_excel_resume, batch_read_word_resumes, batch_rating,
    clean_temp_files, safe_list_to_str, safe_str_to_list, normalize_path,
    sort_candidates_df,
    init_candidate_stay_time, update_candidate_stay_time,
    save_candidate_stay_time_data,
    get_stage_experiment_config,
    auto_load_candidates                  # 新增导入
)

# ===================== 全局初始化 =====================
def init_session_state():
    default_states = {
        "current_stage": "pre",
        "stage_completed": {"pre": False, "mid": False, "post": False},
        "candidates": [],
        "result_df": pd.DataFrame(),
        "decisions": {},
        "current_page": 1,
        "experimenter_id": None,
        "experimenter_info": {},
        "target_hires": 10,
        "experiment_dir": None,
        "candidate_stay_time": {},
        "algorithm_literacy": [4] * len(ALGORITHM_LITERACY_ITEMS),
        "pressure_condition": FIXED_PRESSURE_CONDITION,
        "info_collected": False,
        "resumes_uploaded": False,
        "stage_start_time": {},
        "candidate_decision_time": {},
        "candidate_decision_modifications": {},
        "stage_total_time": {},
        "pre_order": None,
        "scroll_to_top": False,
        "dependency_completed": False,
        "show_dependency_form": False,
        "show_thanks": False,
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

st.set_page_config(
    page_title="招聘决策实验系统",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()

# ===================== 持久化进度 =====================
def save_progress():
    if not st.session_state.experiment_dir:
        return
    progress = {
        "current_stage": st.session_state.current_stage,
        "stage_completed": st.session_state.stage_completed,
        "resumes_uploaded": st.session_state.resumes_uploaded,
        "target_hires": st.session_state.target_hires,
        "algorithm_literacy": st.session_state.algorithm_literacy,
        "pressure_condition": st.session_state.pressure_condition,
        "dependency_completed": st.session_state.dependency_completed,
    }
    progress_path = os.path.join(st.session_state.experiment_dir, "progress.json")
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_progress():
    if not st.session_state.experiment_dir:
        return
    progress_path = os.path.join(st.session_state.experiment_dir, "progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
            st.session_state.current_stage = progress.get("current_stage", "pre")
            st.session_state.stage_completed = progress.get("stage_completed", {"pre": False, "mid": False, "post": False})
            st.session_state.resumes_uploaded = progress.get("resumes_uploaded", False)
            st.session_state.target_hires = progress.get("target_hires", 5)
            st.session_state.algorithm_literacy = progress.get("algorithm_literacy", [4] * len(ALGORITHM_LITERACY_ITEMS))
            st.session_state.pressure_condition = progress.get("pressure_condition", FIXED_PRESSURE_CONDITION)
            st.session_state.dependency_completed = progress.get("dependency_completed", False)

            if st.session_state.stage_completed.get(st.session_state.current_stage, False):
                csv_path = os.path.join(st.session_state.experiment_dir, f"stage_{st.session_state.current_stage}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, encoding="utf-8-sig")
                    for col in ["技能", "证书"]:
                        if col in df.columns:
                            df[col] = df[col].apply(safe_str_to_list)
                    st.session_state.result_df = df
                    if "招聘决策" in df.columns:
                        st.session_state.decisions = dict(zip(df["候选人姓名"], df["招聘决策"]))
        except Exception as e:
            st.warning(f"恢复进度失败：{e}")

# ===================== 辅助函数 =====================
def get_stage_key_list():
    return ["pre", "mid", "post"]

def get_next_stage(current):
    stages = get_stage_key_list()
    idx = stages.index(current)
    if idx + 1 < len(stages):
        return stages[idx + 1]
    return None

def is_stage_complete():
    if not st.session_state.candidates:
        return False
    missing = [c["name"] for c in st.session_state.candidates if c["name"] not in st.session_state.decisions]
    return len(missing) == 0

def record_decision_time(candidate_name, new_decision, old_decision):
    key = (st.session_state.current_stage, candidate_name)
    if key not in st.session_state.candidate_decision_time:
        start = st.session_state.stage_start_time.get(st.session_state.current_stage, time.time())
        elapsed = time.time() - start
        st.session_state.candidate_decision_time[key] = elapsed
        st.session_state.candidate_decision_modifications[key] = 0
    else:
        if new_decision != old_decision:
            st.session_state.candidate_decision_modifications[key] += 1

def save_current_stage():
    current = st.session_state.current_stage
    stage_config = get_stage_experiment_config(current)

    if current in st.session_state.stage_start_time:
        total_time = time.time() - st.session_state.stage_start_time[current]
        st.session_state.stage_total_time[current] = total_time
    else:
        total_time = 0

    base_rows = []
    for c in st.session_state.candidates:
        base_rows.append({
            "候选人姓名": c["name"],
            "性别": c["gender"],
            "毕业院校": c.get("university", ""),
            "院校等级": c.get("university_rank", ""),
            "专业": c.get("major", ""),
            "学历": c["education"],
            "工作年限": c["work_year"],
            "技能": c["skills"],
            "相关项目数": c["related_project_num"],
            "证书": c["certifications"],
            "自我评价": c.get("self_evaluation", ""),
            "实习经历": c.get("internship", ""),
            "获奖情况": c.get("awards", ""),
            "照片": c.get("photo", ""),
            "联系电话": c.get("phone", ""),
            "邮箱": c.get("email", ""),
            "出生日期": c.get("birthday", ""),
            "年龄": c.get("age", ""),
        })
    stage_df = pd.DataFrame(base_rows)

    if not st.session_state.result_df.empty:
        if "候选人姓名" not in st.session_state.result_df.columns:
            st.error("评分数据缺少“候选人姓名”列，无法保存。")
            return False
        score_cols = ["无偏见基础分", "最终评分", "评分说明", "偏见模式", "排名"]
        available_cols = [col for col in ["候选人姓名"] + score_cols if col in st.session_state.result_df.columns]
        score_df = st.session_state.result_df[available_cols].copy()
        stage_df = stage_df.merge(score_df, on="候选人姓名", how="left")
        for col in score_cols:
            if col not in stage_df.columns:
                stage_df[col] = ""
        if "排名" not in stage_df.columns:
            stage_df["排名"] = range(1, len(stage_df) + 1)
    else:
        stage_df["无偏见基础分"] = ""
        stage_df["最终评分"] = ""
        stage_df["评分说明"] = "无AI评分"
        stage_df["偏见模式"] = False
        stage_df["排名"] = range(1, len(stage_df) + 1)

    if "候选人姓名" not in stage_df.columns:
        stage_df["候选人姓名"] = [row["候选人姓名"] for row in base_rows]

    stage_df["招聘决策"] = stage_df["候选人姓名"].map(st.session_state.decisions)

    decision_times = []
    mod_counts = []
    for _, row in stage_df.iterrows():
        name = row["候选人姓名"]
        key = (current, name)
        dt = st.session_state.candidate_decision_time.get(key, 0)
        decision_times.append(round(dt, 2))
        mod_counts.append(st.session_state.candidate_decision_modifications.get(key, 0))
    stage_df["决策耗时（秒）"] = decision_times
    stage_df["决策修改次数"] = mod_counts

    stage_df["实验阶段"] = stage_config["name"]
    stage_df["AI辅助"] = stage_config["ai_assist"]
    stage_df["偏见模式"] = stage_config["bias_mode"]
    stage_df["阶段总耗时（秒）"] = total_time

    for k, v in st.session_state.experimenter_info.items():
        stage_df[f"实验者_{k}"] = v

    for i, score in enumerate(st.session_state.algorithm_literacy):
        stage_df[f"算法素养_{i+1}"] = score
    stage_df["算法素养总分"] = sum(st.session_state.algorithm_literacy)

    for col in ["技能", "证书"]:
        if col in stage_df.columns:
            stage_df[col] = stage_df[col].apply(safe_list_to_str)

    save_path = os.path.join(st.session_state.experiment_dir, f"stage_{current}.csv")
    stage_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    save_candidate_stay_time_data(st.session_state, os.path.join(st.session_state.experiment_dir, "stay_time_candidates.json"))

    if is_stage_complete():
        st.session_state.stage_completed[current] = True
    else:
        st.session_state.stage_completed[current] = False
    save_progress()
    return True

def load_stage_data(stage_key):
    csv_path = os.path.join(st.session_state.experiment_dir, f"stage_{stage_key}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for col in ["技能", "证书"]:
            if col in df.columns:
                df[col] = df[col].apply(safe_str_to_list)
        st.session_state.result_df = df
        if "招聘决策" in df.columns:
            st.session_state.decisions = dict(zip(df["候选人姓名"], df["招聘决策"]))
        return True
    return False

def generate_non_ai_stage_data(stage_key, candidates):
    rows = []
    for i, c in enumerate(candidates):
        rows.append({
            "候选人姓名": c["name"],
            "性别": c["gender"],
            "毕业院校": c.get("university", ""),
            "院校等级": c.get("university_rank", ""),
            "专业": c.get("major", ""),
            "学历": c["education"],
            "工作年限": c["work_year"],
            "技能": c["skills"],
            "相关项目数": c["related_project_num"],
            "证书": c["certifications"],
            "自我评价": c.get("self_evaluation", ""),
            "实习经历": c.get("internship", ""),
            "获奖情况": c.get("awards", ""),
            "照片": c.get("photo", ""),
            "联系电话": c.get("phone", ""),
            "邮箱": c.get("email", ""),
            "出生日期": c.get("birthday", ""),
            "年龄": c.get("age", ""),
            "无偏见基础分": "",
            "最终评分": "",
            "评分说明": "无AI评分",
            "偏见模式": False,
            "排名": i+1
        })
    random.shuffle(rows)
    if stage_key == "pre":
        st.session_state.pre_order = [row["候选人姓名"] for row in rows]
    if stage_key == "post" and st.session_state.pre_order is not None:
        max_attempts = 3
        for attempt in range(max_attempts):
            current_order = [row["候选人姓名"] for row in rows]
            if current_order == st.session_state.pre_order:
                random.shuffle(rows)
            else:
                break
        if attempt == max_attempts - 1 and [row["候选人姓名"] for row in rows] == st.session_state.pre_order:
            st.warning("⚠️ 无法生成与pre阶段不同的顺序，请检查数据量是否过小。")
    for idx, row in enumerate(rows):
        row["排名"] = idx + 1
    decisions = {row["候选人姓名"]: UI_CONFIG["decision_options"][1] for row in rows}
    return rows, decisions

def initialize_stage_data(stage_key):
    stage_config = get_stage_experiment_config(stage_key)
    if stage_config["ai_assist"]:
        result_df = batch_rating(st.session_state.candidates, bias_mode=stage_config["bias_mode"])
        if result_df is None or result_df.empty:
            st.error("AI评分失败，请检查数据或联系管理员。")
            return False
        st.session_state.result_df = result_df
        st.session_state.decisions = {row["候选人姓名"]: UI_CONFIG["decision_options"][1] for _, row in result_df.iterrows()}
    else:
        rows, decisions = generate_non_ai_stage_data(stage_key, st.session_state.candidates)
        st.session_state.result_df = pd.DataFrame(rows)
        st.session_state.decisions = decisions
    return True

def switch_to_stage(target_stage):
    current = st.session_state.current_stage
    if current == target_stage:
        return
    if not st.session_state.stage_completed[current]:
        if not save_current_stage():
            st.error(f"保存当前阶段 {current} 失败，无法切换。")
            return
    st.session_state.result_df = pd.DataFrame()
    st.session_state.decisions = {}
    st.session_state.current_page = 1
    st.session_state.candidate_stay_time = {}
    st.session_state.candidate_decision_time = {}
    st.session_state.candidate_decision_modifications = {}
    success = False
    if load_stage_data(target_stage):
        success = True
        st.session_state.stage_completed[target_stage] = False
    else:
        if initialize_stage_data(target_stage):
            success = True
            st.session_state.stage_completed[target_stage] = False
        else:
            st.error(f"初始化阶段 {target_stage} 数据失败。")
    if not success:
        st.error(f"切换到阶段 {target_stage} 失败，将保持在当前阶段。")
        if st.session_state.stage_completed[current]:
            load_stage_data(current)
        else:
            if not load_stage_data(current) and not initialize_stage_data(current):
                st.error(f"无法恢复当前阶段 {current} 数据，请重新上传简历。")
        return
    st.session_state.current_stage = target_stage
    if target_stage not in st.session_state.stage_start_time:
        st.session_state.stage_start_time[target_stage] = time.time()
    save_progress()
    st.rerun()

def switch_to_next_stage():
    current = st.session_state.current_stage
    if not is_stage_complete():
        st.warning("请先完成所有候选人的决策！")
        return False
    if not st.session_state.stage_completed[current]:
        if not save_current_stage():
            return False
    next_stage = get_next_stage(current)
    if next_stage is None:
        st.session_state.experiment_completed = True
        st.rerun()
        return True
    st.session_state.current_stage = next_stage
    st.session_state.result_df = pd.DataFrame()
    st.session_state.decisions = {}
    st.session_state.current_page = 1
    st.session_state.candidate_stay_time = {}
    st.session_state.candidate_decision_time = {}
    st.session_state.candidate_decision_modifications = {}
    stage_config = get_stage_experiment_config(next_stage)
    if stage_config["ai_assist"]:
        result_df = batch_rating(st.session_state.candidates, bias_mode=stage_config["bias_mode"])
        if result_df is None or result_df.empty:
            st.error("AI评分失败，请检查数据或联系管理员。")
            return False
        st.session_state.result_df = result_df
        st.session_state.decisions = {row["候选人姓名"]: UI_CONFIG["decision_options"][1] for _, row in result_df.iterrows()}
    else:
        rows, decisions = generate_non_ai_stage_data(next_stage, st.session_state.candidates)
        st.session_state.result_df = pd.DataFrame(rows)
        st.session_state.decisions = decisions
    st.session_state.stage_start_time[next_stage] = time.time()
    save_progress()
    st.success(f"已进入下一阶段：{stage_config['name']}")
    st.rerun()
    return True

def package_experiment_data():
    """打包实验数据并返回 zip 数据流"""
    if not st.session_state.experiment_dir:
        st.error("实验目录未创建，请联系管理员。")
        return None
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(st.session_state.experiment_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, st.session_state.experiment_dir)
                zip_file.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer

def finalize_experiment():
    """提交实验数据：先检查依赖量表，若未填则显示表单，否则打包下载"""
    if not st.session_state.dependency_completed:
        st.session_state.show_dependency_form = True
        st.rerun()
        return False

    # 已填写依赖量表，执行打包下载
    # 先保存当前阶段数据（确保最后阶段已保存）
    if st.session_state.current_stage in get_stage_key_list():
        if not save_current_stage():
            st.error("保存当前阶段失败，请检查错误信息。")
            return False

    if not all(st.session_state.stage_completed.values()):
        missing = [s for s, done in st.session_state.stage_completed.items() if not done]
        st.warning(f"以下阶段未完成：{missing}，无法提交。")
        return False

    zip_buffer = package_experiment_data()
    if zip_buffer is None:
        return False
    st.download_button(
        label="📥 下载实验数据压缩包",
        data=zip_buffer,
        file_name=f"实验数据_{st.session_state.experimenter_id}.zip",
        mime="application/zip"
    )
    st.success(f"数据打包完成！本地备份已保存在：{st.session_state.experiment_dir}")
    return True

def save_dependency_data(scores):
    """保存算法依赖量表数据，并显示感谢界面"""
    # 先保存当前阶段数据，确保最后阶段数据已写入
    if st.session_state.current_stage in get_stage_key_list():
        save_current_stage()
    dep_data = {
        "items": ALGORITHM_DEPENDENCY_ITEMS,
        "scores": scores,
        "total_score": sum(scores),
        "timestamp": datetime.now().isoformat()
    }
    save_path = os.path.join(st.session_state.experiment_dir, "algorithm_dependency.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dep_data, f, ensure_ascii=False, indent=2)
    st.session_state.dependency_completed = True
    st.session_state.show_dependency_form = False
    st.session_state.show_thanks = True
    save_progress()
    st.rerun()

# ===================== 自定义CSS =====================
st.markdown("""
    <style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 8px;
    }
    .sub-header {
        font-size: 20px;
        font-weight: bold;
        color: #4A6FA5;
        margin: 20px 0 10px 0;
    }
    .job-desc-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #2E86AB;
    }
    .stage-progress {
        font-size: 16px;
        margin-bottom: 15px;
        color: #2C3E50;
    }
    .candidate-card {
        background-color: #F9F9F9;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .custom-decision-radio .stRadio > div {
        display: flex;
        justify-content: space-between;
        gap: 20px;
    }
    .custom-decision-radio .stRadio label {
        font-size: 1.2rem;
        font-weight: 500;
        cursor: pointer;
    }
    /* 缩小侧边栏指标数值的字体，避免两位数显示不全 */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== 侧边栏 =====================
with st.sidebar:
    if st.session_state.info_collected:
        load_progress()
        st.markdown("### 👤 实验者信息")
        st.write(f"**姓名**：{st.session_state.experimenter_info.get('姓名', '未知')}")
        st.write(f"**学号**：{st.session_state.experimenter_info.get('学号', '未知')}")

        if not st.session_state.resumes_uploaded:
            if st.button("✏️ 修改信息", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            st.info("实验已开始，无法修改个人信息。")
            if st.button("⚠️ 重置整个实验", use_container_width=True):
                if st.session_state.experiment_dir and os.path.exists(st.session_state.experiment_dir):
                    shutil.rmtree(st.session_state.experiment_dir)
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.markdown("---")
        st.markdown("### 📊 招聘进度")

        # 计划招聘人数固定为10（可根据需要修改初始值）
        target_hires = st.session_state.target_hires  # 使用 session_state 中已有的值，不再显示输入框


        def get_current_stats():
            if not st.session_state.decisions:
                return 0, 0, 0
            hired = sum(1 for d in st.session_state.decisions.values() if d == "进入面试")
            pending = sum(1 for d in st.session_state.decisions.values() if d == "待定")
            rejected = sum(1 for d in st.session_state.decisions.values() if d == "拒绝")
            return hired, pending, rejected


        hired, pending, rejected = get_current_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ 已进入面试", f"{hired} / {target_hires}")
        col2.metric("⏳ 待定", pending)
        col3.metric("❌ 拒绝", rejected)
        if hired > target_hires:
            st.warning(f"⚠️ 已进入面试人数（{hired}）已超过计划招聘人数（{target_hires}）")
        else:
            st.info(f"剩余面试名额：{max(0, target_hires - hired)}")

        with st.expander("📋 决策分类详情", expanded=False):
            if st.session_state.decisions:
                hired_list = [name for name, dec in st.session_state.decisions.items() if dec == "进入面试"]
                pending_list = [name for name, dec in st.session_state.decisions.items() if dec == "待定"]
                rejected_list = [name for name, dec in st.session_state.decisions.items() if dec == "拒绝"]

                st.markdown("**✅ 进入面试**")
                if hired_list:
                    st.write("、".join(hired_list))
                else:
                    st.write("暂无")

                st.markdown("**⏳ 待定**")
                if pending_list:
                    st.write("、".join(pending_list))
                else:
                    st.write("暂无")

                st.markdown("**❌ 拒绝**")
                if rejected_list:
                    st.write("、".join(rejected_list))
                else:
                    st.write("暂无")
            else:
                st.info("尚未做出任何决策")
    else:
        st.info("请填写个人信息并开始实验")

# ===================== 页面主体 =====================
if not st.session_state.info_collected:
    st.markdown('<div class="main-header">📝 招聘决策实验系统</div>', unsafe_allow_html=True)
    st.markdown("请填写以下信息开始实验")

    with st.form("experimenter_form"):
        exp_name = st.text_input("姓名", placeholder="请输入您的姓名")
        exp_id = st.text_input("学号/职业", placeholder="请输入学号或职业内容")
        exp_gender = st.radio("性别", ["男", "女", "不愿透露"], horizontal=True)
        exp_age = st.number_input("年龄", min_value=18, max_value=100, step=1, value=25)
        exp_major = st.text_input("专业(选填)", placeholder="机械工程、计算机科学等")
        exp_education = st.selectbox("最高学历", ["本科", "硕士", "博士", "其他"])
        exp_ai_familiarity = st.slider("对人工智能的熟悉程度", 1, 7, 4, help="1=完全不了解，7=非常熟悉")
        exp_recruitment_exp = st.radio("是否有招聘经验", ["有", "无"], horizontal=True)
        exp_similar_exp = st.radio("是否参加过类似的招聘实验", ["是", "否"], horizontal=True)

        st.markdown("#### 对以下情景内容做出选择（1代表完全不同意-7代表完全同意）")

        alg_lit = []
        for i, q in enumerate(ALGORITHM_LITERACY_ITEMS):
            alg_lit.append(st.slider(f"{i + 1}. {q}", 1, 7, 4, key=f"alg_{i}"))
            if i < len(ALGORITHM_LITERACY_ITEMS) - 1:
                st.write("")

        submitted = st.form_submit_button("开始实验", type="primary")

        if submitted and exp_name and exp_id:
            clean_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', exp_name)
            st.session_state.experimenter_id = f"{clean_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            exp_dir = f"experiment_data/{st.session_state.experimenter_id}"
            os.makedirs(exp_dir, exist_ok=True)
            st.session_state.experiment_dir = exp_dir
            st.session_state.experimenter_info = {
                "姓名": exp_name,
                "学号": exp_id,
                "性别": exp_gender,
                "年龄": exp_age,
                "专业": exp_major,
                "学历": exp_education,
                "AI熟悉程度": exp_ai_familiarity,
                "招聘经验": exp_recruitment_exp,
                "类似实验经验": exp_similar_exp,
                "实验者ID": st.session_state.experimenter_id,
                "任务压力条件": FIXED_PRESSURE_CONDITION
            }
            st.session_state.algorithm_literacy = alg_lit
            st.session_state.pressure_condition = FIXED_PRESSURE_CONDITION
            st.session_state.info_collected = True

            metadata = {
                "experimenter_id": st.session_state.experimenter_id,
                "start_time": datetime.now().isoformat(),
                "experimenter_info": st.session_state.experimenter_info,
                "algorithm_literacy_scores": alg_lit,
                "algorithm_literacy_items": ALGORITHM_LITERACY_ITEMS,
                "pressure_condition": FIXED_PRESSURE_CONDITION,
                "stage_config": EXPERIMENT_STAGES
            }
            with open(os.path.join(exp_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            save_progress()
            st.success("信息已保存，实验开始！")
            st.rerun()
        elif submitted:
            st.warning("请填写姓名和学号")
    st.stop()

if st.session_state.get("scroll_to_top", False):
    st.markdown('<script>window.scrollTo(0,0);</script>', unsafe_allow_html=True)
    st.session_state.scroll_to_top = False

if st.session_state.pressure_condition == "高压力":
    st.warning("⚠️ 请在每个阶段5分钟内完成所有决策。")
    if st.session_state.current_stage not in st.session_state.stage_start_time:
        st.session_state.stage_start_time[st.session_state.current_stage] = time.time()
    elapsed = time.time() - st.session_state.stage_start_time[st.session_state.current_stage]
    if elapsed > 300:
        st.error("⚠️ 当前阶段已超过5分钟！")
    else:
        st.info(f"⏱️ 当前阶段已用时：{int(elapsed // 60)}分{int(elapsed % 60)}秒 / 5分钟")
else:
    st.success("请按照您的真实想法进行决策")

if st.session_state.current_stage not in st.session_state.stage_start_time:
    st.session_state.stage_start_time[st.session_state.current_stage] = time.time()

current_stage_config = get_stage_experiment_config(st.session_state.current_stage)

stage_names = [EXPERIMENT_STAGES[s]["name"] for s in get_stage_key_list()]
current_idx = get_stage_key_list().index(st.session_state.current_stage)
st.markdown(f"""
    <div class="stage-progress">
        实验阶段：{' → '.join([f"<b>{name}</b>" if i==current_idx else name for i, name in enumerate(stage_names)])}
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="main-header">
        {current_stage_config['name']}
    </div>
    <div style="color:#666; font-size:14px; margin-bottom:20px;">
        当前模式：{'AI辅助' if current_stage_config['ai_assist'] else '无AI辅助'}
    </div>
""", unsafe_allow_html=True)

st.markdown(JOB_DESCRIPTION)
st.divider()

# ===================== 算法依赖量表表单与感谢界面 =====================
if st.session_state.get("show_thanks", False):
    st.markdown("### 🎉 实验完成")
    st.success("感谢您的决策与回答！您的数据已成功保存。")
    st.balloons()
    # 再次保存当前阶段，确保最后阶段数据已写入
    if st.session_state.current_stage in get_stage_key_list():
        save_current_stage()
    st.markdown("请点击下方按钮下载实验数据压缩包：")
    zip_buffer = package_experiment_data()
    if zip_buffer:
        st.download_button(
            label="📥 下载实验数据压缩包",
            data=zip_buffer,
            file_name=f"实验数据_{st.session_state.experimenter_id}.zip",
            mime="application/zip",
            use_container_width=True
        )
    st.stop()

if st.session_state.get("show_dependency_form", False):
    st.markdown("### 最后一个问卷")
    st.markdown("请根据您的真实感受，对以下陈述进行评分（1=完全不同意，5=完全同意）")
    with st.form("dependency_form"):
        scores = []
        for i, item in enumerate(ALGORITHM_DEPENDENCY_ITEMS):
            score = st.slider(item, 1, 5, 3, key=f"dep_{i}")
            scores.append(score)
        submitted_dep = st.form_submit_button("提交量表")
        if submitted_dep:
            save_dependency_data(scores)
    st.stop()

# ===================== 简历自动读取（替换手动上传） =====================
if not st.session_state.resumes_uploaded:
    st.markdown('<div class="sub-header">📁 简历自动加载</div>', unsafe_allow_html=True)
    st.info("系统将自动从“resume”文件夹中读取简历文件，并匹配“photo”文件夹中的照片。")

    with st.spinner("正在自动加载简历..."):
        candidates, errors = auto_load_candidates(RESUME_FOLDER, PHOTO_FOLDER)
        if errors:
            for err in errors:
                st.warning(err)
        if not candidates:
            st.error("未能加载任何候选人，请检查“resume”文件夹中的文件。")
            st.stop()
        else:
            st.session_state.candidates = candidates
            # 初始化当前阶段的数据（默认是 pre 阶段）
            if current_stage_config["ai_assist"]:
                result_df = batch_rating(candidates, bias_mode=current_stage_config["bias_mode"])
                if result_df is None or result_df.empty:
                    st.error("AI评分失败，请检查数据或联系管理员。")
                    st.stop()
                st.session_state.result_df = result_df
                st.session_state.decisions = {row["候选人姓名"]: UI_CONFIG["decision_options"][1] for _, row in result_df.iterrows()}
            else:
                rows, decisions = generate_non_ai_stage_data(st.session_state.current_stage, candidates)
                st.session_state.result_df = pd.DataFrame(rows)
                st.session_state.decisions = decisions
            st.session_state.resumes_uploaded = True
            save_progress()
            st.success(f"✅ 成功加载 {len(candidates)} 位候选人！")
            st.rerun()
    st.stop()

# ===================== 招聘决策标注 =====================
if st.session_state.resumes_uploaded:
    if st.session_state.stage_completed.get(st.session_state.current_stage, False):
        st.success(f"✅ {current_stage_config['name']} 已完成！")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        next_stage_key = get_next_stage(st.session_state.current_stage)
        if next_stage_key is None:
            if st.button("📤 提交实验数据", type="primary", use_container_width=True):
                finalize_experiment()
        else:
            next_stage_name = EXPERIMENT_STAGES[next_stage_key]["name"]
            button_text = f"➡️ 进入{next_stage_name}"
            if st.button(button_text, type="primary", use_container_width=True):
                switch_to_next_stage()
    else:
        if st.session_state.result_df.empty:
            st.error("数据加载错误，请重新上传简历。")
            st.stop()

        st.markdown('<div class="sub-header">🎯 招聘决策标注</div>', unsafe_allow_html=True)

        if current_stage_config["ai_assist"]:
            sort_key = st.selectbox(
                "选择候选人排序方式",
                options=list(SORT_OPTIONS.keys()),
                format_func=lambda x: SORT_OPTIONS[x]["name"],
                key="sort_selector",
                label_visibility="collapsed"
            )
            sorted_df = sort_candidates_df(st.session_state.result_df, sort_key)
        else:
            sorted_df = st.session_state.result_df

        total_candidates = len(sorted_df)
        page_size = UI_CONFIG["max_candidates_per_page"]
        total_pages = max(1, (total_candidates + page_size - 1) // page_size)

        current_page = max(1, min(st.session_state.current_page, total_pages))
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_candidates)
        paginated_df = sorted_df.iloc[start_idx:end_idx]

        for idx, row in paginated_df.iterrows():
            name = row["候选人姓名"]
            st.markdown(f"### <b>📄 【{row['排名']}】{name}</b>", unsafe_allow_html=True)
            if current_stage_config["show_score"]:
                st.markdown(f"<span style='color:#666;'>AI评分：{row['最终评分']}</span>", unsafe_allow_html=True)

            init_candidate_stay_time(st.session_state, name)
            update_candidate_stay_time(st.session_state, name)

            col1, col2, col3 = st.columns([1, 2, 2])

            with col1:
                photo_file = row.get("照片", "")
                if photo_file and photo_file.strip():
                    photo_filename = os.path.basename(photo_file)
                    photo_path = os.path.join(UI_CONFIG["photo_folder"], photo_filename)
                    photo_path = normalize_path(photo_path)

                    valid_photo_path = None
                    if os.path.exists(photo_path):
                        valid_photo_path = photo_path
                    else:
                        photo_path_no_ext = os.path.splitext(photo_path)[0]
                        img_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
                        for ext in img_extensions:
                            test_path = photo_path_no_ext + ext
                            if os.path.exists(test_path):
                                valid_photo_path = test_path
                                break

                    if valid_photo_path:
                        st.image(valid_photo_path, width=150, caption="候选人照片")
                    else:
                        st.warning(f"照片文件未找到：{photo_path}")
                else:
                    default_path = os.path.join(UI_CONFIG["photo_folder"], UI_CONFIG["default_photo"])
                    default_path = normalize_path(default_path)
                    if os.path.exists(default_path):
                        st.image(default_path, width=150, caption="无照片")
                    else:
                        st.write("📷 无照片")

            with col2:
                st.markdown("**📋 基础信息**")
                st.write(f"👤 姓名：{row['候选人姓名']}")
                st.write(f"⚥ 性别：{row['性别']}")
                st.write(f"🎂 年龄：{row.get('年龄', '未知')}岁")
                st.write(f"📅 出生日期：{row.get('出生日期', '未知')}")
                st.write(f"📞 电话：{row.get('联系电话', '无')}")
                st.write(f"📧 邮箱：{row.get('邮箱', '无')}")
                st.write(f"🏫 毕业院校：{row.get('毕业院校', '')} ({row.get('院校等级', '')})")
                st.write(f"📚 专业：{row.get('专业', '')}")
                st.write(f"🎓 学历：{row['学历']}")
                st.write(f"💼 工作年限：{row['工作年限']}年")

            with col3:
                st.markdown("**🔧 能力资质**")
                st.write(f"🛠️ 掌握技能：{safe_list_to_str(row['技能'])}")
                st.write(f"📁 相关项目数：{row['相关项目数']}个")
                st.write(f"📜 持有证书：{safe_list_to_str(row['证书'])}")

                st.markdown("**🏆 实习与获奖**")
                internship = row.get('实习经历', '')
                st.write(f"💼 实习经历：{internship if internship else '无'}")
                awards = row.get('获奖情况', '')
                st.write(f"🏅 获奖情况：{awards if awards else '无'}")

                st.markdown("**📝 自我评价**")
                self_eval = row.get('自我评价', '')
                st.write(self_eval if self_eval else '无')

            if current_stage_config["show_score"]:
                st.divider()
                st.write("**📊 AI评分说明**")
                st.info(row["评分说明"])

            current_decision = st.session_state.decisions.get(name, UI_CONFIG["decision_options"][1])
            with st.container():
                st.markdown('<div class="custom-decision-radio">', unsafe_allow_html=True)
                decision = st.radio(
                    f"请选择{name}的招聘决策",
                    UI_CONFIG["decision_options"],
                    index=UI_CONFIG["decision_options"].index(current_decision),
                    key=f"decision_{name}_{st.session_state.current_stage}_{idx}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            if decision != current_decision:
                record_decision_time(name, decision, current_decision)
                st.session_state.decisions[name] = decision
                st.rerun()

            st.divider()

        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("← 上一页", use_container_width=True, disabled=(current_page == 1)):
                st.session_state.current_page -= 1
                st.session_state.scroll_to_top = True
                st.rerun()
        with col_info:
            st.markdown(
                f"<div style='text-align:center; font-size:16px;'>第 {current_page} 页 / 共 {total_pages} 页</div>",
                unsafe_allow_html=True
            )
        with col_next:
            if st.button("→ 下一页", use_container_width=True, disabled=(current_page == total_pages)):
                st.session_state.current_page += 1
                st.session_state.scroll_to_top = True
                st.rerun()

        if is_stage_complete():
            st.success("在完成决策之前请不要误触进入下一阶段！")
            st.write("")
            st.write("")
            next_stage_key = get_next_stage(st.session_state.current_stage)
            if next_stage_key is None:
                if st.button("📤 提交实验数据", type="primary", use_container_width=True):
                    finalize_experiment()
            else:
                next_stage_name = EXPERIMENT_STAGES[next_stage_key]["name"]
                button_text = f"➡️ 进入{next_stage_name}"
                if st.button(button_text, type="primary", use_container_width=True):
                    switch_to_next_stage()
        else:
            missing = [name for name in st.session_state.result_df["候选人姓名"] if name not in st.session_state.decisions]
            if missing:
                st.warning(f"以下候选人尚未做出决策：{', '.join(missing)}")