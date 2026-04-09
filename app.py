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
    RESUME_FOLDER, PHOTO_FOLDER
)
from core_rating import (
    read_excel_resume, batch_read_word_resumes, batch_rating,
    clean_temp_files, safe_list_to_str, safe_str_to_list, normalize_path,
    sort_candidates_df,
    init_candidate_stay_time, update_candidate_stay_time,
    save_candidate_stay_time_data,
    get_stage_experiment_config,
    auto_load_candidates
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
        # 问卷相关状态
        "manipulation_check_done": False,    # 操纵检查是否已完成
        "post_confidence": {},               # post阶段信心值
        "debriefing_done": False,            # 事后回顾是否已完成
        "dependency_done": False,            # 依赖量表是否已完成
        # 以下用于控制表单显示顺序（不再单独使用多个show_xxx，统一用一个问卷流程）
        "questionnaire_step": 0,             # 0=未开始,1=操纵检查,2=事后回顾,3=依赖量表,4=完成
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

st.set_page_config(page_title="招聘决策实验系统", page_icon="📝", layout="wide", initial_sidebar_state="expanded")
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
        "manipulation_check_done": st.session_state.manipulation_check_done,
        "debriefing_done": st.session_state.debriefing_done,
        "dependency_done": st.session_state.dependency_done,
        "questionnaire_step": st.session_state.questionnaire_step,
    }
    with open(os.path.join(st.session_state.experiment_dir, "progress.json"), "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_progress():
    if not st.session_state.experiment_dir:
        return
    prog_path = os.path.join(st.session_state.experiment_dir, "progress.json")
    if os.path.exists(prog_path):
        try:
            with open(prog_path, "r", encoding="utf-8") as f:
                prog = json.load(f)
            st.session_state.current_stage = prog.get("current_stage", "pre")
            st.session_state.stage_completed = prog.get("stage_completed", {"pre": False, "mid": False, "post": False})
            st.session_state.resumes_uploaded = prog.get("resumes_uploaded", False)
            st.session_state.target_hires = prog.get("target_hires", 10)
            st.session_state.algorithm_literacy = prog.get("algorithm_literacy", [4]*len(ALGORITHM_LITERACY_ITEMS))
            st.session_state.pressure_condition = prog.get("pressure_condition", FIXED_PRESSURE_CONDITION)
            st.session_state.manipulation_check_done = prog.get("manipulation_check_done", False)
            st.session_state.debriefing_done = prog.get("debriefing_done", False)
            st.session_state.dependency_done = prog.get("dependency_done", False)
            st.session_state.questionnaire_step = prog.get("questionnaire_step", 0)

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
                    if st.session_state.current_stage == "post" and "决策信心" in df.columns:
                        st.session_state.post_confidence = dict(zip(df["候选人姓名"], df["决策信心"]))
        except Exception as e:
            st.warning(f"恢复进度失败：{e}")

# ===================== 辅助函数 =====================
def get_stage_key_list():
    return ["pre", "mid", "post"]

def get_next_stage(current):
    stages = get_stage_key_list()
    idx = stages.index(current)
    return stages[idx+1] if idx+1 < len(stages) else None

def is_stage_complete():
    if not st.session_state.candidates:
        return False
    missing = [c["name"] for c in st.session_state.candidates if c["name"] not in st.session_state.decisions]
    return len(missing) == 0

def record_decision_time(candidate_name, new_decision, old_decision):
    key = (st.session_state.current_stage, candidate_name)
    if key not in st.session_state.candidate_decision_time:
        start = st.session_state.stage_start_time.get(st.session_state.current_stage, time.time())
        st.session_state.candidate_decision_time[key] = time.time() - start
        st.session_state.candidate_decision_modifications[key] = 0
    elif new_decision != old_decision:
        st.session_state.candidate_decision_modifications[key] += 1

def save_current_stage():
    current = st.session_state.current_stage
    stage_config = get_stage_experiment_config(current)
    total_time = time.time() - st.session_state.stage_start_time.get(current, time.time())
    st.session_state.stage_total_time[current] = total_time

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

    if not st.session_state.result_df.empty and "候选人姓名" in st.session_state.result_df.columns:
        score_cols = ["无偏见基础分", "最终评分", "评分说明", "偏见模式", "排名"]
        available_cols = [col for col in ["候选人姓名"] + score_cols if col in st.session_state.result_df.columns]
        score_df = st.session_state.result_df[available_cols].copy()
        stage_df = stage_df.merge(score_df, on="候选人姓名", how="left")
    else:
        for col in ["无偏见基础分", "最终评分", "评分说明", "偏见模式", "排名"]:
            stage_df[col] = ""
        stage_df["排名"] = range(1, len(stage_df)+1)

    stage_df["招聘决策"] = stage_df["候选人姓名"].map(st.session_state.decisions)
    if current == "post":
        stage_df["决策信心"] = [st.session_state.post_confidence.get(name, 4) for name in stage_df["候选人姓名"]]
    else:
        stage_df["决策信心"] = ""

    decision_times = []
    mod_counts = []
    for name in stage_df["候选人姓名"]:
        key = (current, name)
        decision_times.append(round(st.session_state.candidate_decision_time.get(key, 0), 2))
        mod_counts.append(st.session_state.candidate_decision_modifications.get(key, 0))
    stage_df["决策耗时（秒）"] = decision_times
    stage_df["决策修改次数"] = mod_counts

    stage_df["实验阶段"] = stage_config["name"]
    stage_df["AI辅助"] = stage_config["ai_assist"]
    stage_df["偏见模式"] = stage_config["bias_mode"]
    stage_df["阶段总耗时（秒）"] = total_time
    stage_df["压力条件"] = st.session_state.pressure_condition
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
    st.session_state.stage_completed[current] = is_stage_complete()
    save_progress()
    return True

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
        st.session_state.pre_order = [r["候选人姓名"] for r in rows]
    if stage_key == "post" and st.session_state.pre_order is not None:
        # 确保顺序与pre不同
        for _ in range(3):
            if [r["候选人姓名"] for r in rows] == st.session_state.pre_order:
                random.shuffle(rows)
            else:
                break
    for idx, r in enumerate(rows):
        r["排名"] = idx+1
    decisions = {r["候选人姓名"]: UI_CONFIG["decision_options"][1] for r in rows}
    return rows, decisions

def initialize_stage_data(stage_key):
    cfg = get_stage_experiment_config(stage_key)
    if cfg["ai_assist"]:
        df = batch_rating(st.session_state.candidates, bias_mode=cfg["bias_mode"])
        if df is None or df.empty:
            st.error("AI评分失败")
            return False
        st.session_state.result_df = df
        st.session_state.decisions = {row["候选人姓名"]: UI_CONFIG["decision_options"][1] for _, row in df.iterrows()}
    else:
        rows, decisions = generate_non_ai_stage_data(stage_key, st.session_state.candidates)
        st.session_state.result_df = pd.DataFrame(rows)
        st.session_state.decisions = decisions
    return True

def switch_to_next_stage():
    current = st.session_state.current_stage
    if not is_stage_complete():
        st.warning("请先完成所有候选人的决策！")
        return False
    if not st.session_state.stage_completed[current]:
        save_current_stage()
    next_stage = get_next_stage(current)
    if next_stage is None:
        # 所有阶段完成，开始问卷流程
        st.session_state.questionnaire_step = 1
        st.rerun()
        return True

    # 进入 mid 或 post 前重置相关状态
    if next_stage == "post":
        st.session_state.post_confidence = {}
        # 如果需要操纵检查且未做，先做
        if not st.session_state.manipulation_check_done:
            st.session_state.questionnaire_step = 1   # 1=操纵检查
            st.rerun()
            return False

    # 初始化下一个阶段的数据
    st.session_state.current_stage = next_stage
    st.session_state.result_df = pd.DataFrame()
    st.session_state.decisions = {}
    st.session_state.current_page = 1
    st.session_state.candidate_stay_time = {}
    st.session_state.candidate_decision_time = {}
    st.session_state.candidate_decision_modifications = {}
    cfg = get_stage_experiment_config(next_stage)
    if cfg["ai_assist"]:
        df = batch_rating(st.session_state.candidates, bias_mode=cfg["bias_mode"])
        if df is None or df.empty:
            st.error("AI评分失败")
            return False
        st.session_state.result_df = df
        st.session_state.decisions = {row["候选人姓名"]: UI_CONFIG["decision_options"][1] for _, row in df.iterrows()}
    else:
        rows, decisions = generate_non_ai_stage_data(next_stage, st.session_state.candidates)
        st.session_state.result_df = pd.DataFrame(rows)
        st.session_state.decisions = decisions
    st.session_state.stage_start_time[next_stage] = time.time()
    save_progress()
    st.success(f"已进入下一阶段：{cfg['name']}")
    st.rerun()
    return True

def package_experiment_data():
    if not st.session_state.experiment_dir:
        return None
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(st.session_state.experiment_dir):
            for f in files:
                path = os.path.join(root, f)
                zf.write(path, os.path.relpath(path, st.session_state.experiment_dir))
    zip_buffer.seek(0)
    return zip_buffer

def generate_summary_csv():
    """生成包含所有问卷和阶段统计的汇总CSV"""
    exp_dir = st.session_state.experiment_dir
    if not exp_dir:
        return
    data = {}

    # 基本信息
    data["实验者ID"] = st.session_state.experimenter_id
    info = st.session_state.experimenter_info
    for k in ["姓名", "学号", "性别", "年龄", "专业", "学历", "AI熟悉程度", "招聘经验", "类似实验经验", "任务压力条件"]:
        data[k] = info.get(k, "")
    data["目标招聘人数"] = st.session_state.target_hires

    # 算法素养
    for i, s in enumerate(st.session_state.algorithm_literacy, 1):
        data[f"算法素养_{i}"] = s
    data["算法素养总分"] = sum(st.session_state.algorithm_literacy)

    # 操纵检查
    manip_path = os.path.join(exp_dir, "manipulation_check.json")
    if os.path.exists(manip_path):
        with open(manip_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        data["是否注意到偏差"] = m.get("bias_awareness", "")
        data["偏差详情"] = m.get("bias_detail", "")

    # 事后回顾
    debrief_path = os.path.join(exp_dir, "debriefing.json")
    if os.path.exists(debrief_path):
        with open(debrief_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        data["AI公平性评分"] = d.get("fairness", "")
        data["不自觉回忆AI分数"] = d.get("recall_ai_score", "")
        data["AI影响程度"] = d.get("influence", "")
        data["纠正行为"] = d.get("correction_behavior", "")
        data["反馈评论"] = d.get("comments", "")

    # 算法依赖量表
    dep_path = os.path.join(exp_dir, "algorithm_dependency.json")
    if os.path.exists(dep_path):
        with open(dep_path, "r", encoding="utf-8") as f:
            dep = json.load(f)
        for i, s in enumerate(dep.get("scores", []), 1):
            data[f"依赖量表_{i}"] = s
        data["依赖量表总分"] = dep.get("total_score", 0)

    # 各阶段统计
    for stage in ["pre", "mid", "post"]:
        csv_path = os.path.join(exp_dir, f"stage_{stage}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "招聘决策" in df.columns:
                data[f"{stage}_录用"] = (df["招聘决策"] == "进入面试").sum()
                data[f"{stage}_待定"] = (df["招聘决策"] == "待定").sum()
                data[f"{stage}_拒绝"] = (df["招聘决策"] == "拒绝").sum()
            if "阶段总耗时（秒）" in df.columns:
                data[f"{stage}_耗时(秒)"] = df["阶段总耗时（秒）"].iloc[0] if len(df) else 0

    df_summary = pd.DataFrame([data])
    df_summary.to_csv(os.path.join(exp_dir, "实验汇总.csv"), index=False, encoding="utf-8-sig")

# ===================== 问卷表单（统一处理） =====================
def show_questionnaire():
    step = st.session_state.questionnaire_step
    # 步骤1: 操纵检查（仅在mid完成后且post开始前需要）
    if step == 1 and not st.session_state.manipulation_check_done:
        st.markdown("### 请回答以下问题（必填）")
        st.markdown("在刚才的 **【实验2】AI辅助** 阶段，你观察到 AI 评分是否存在系统性偏差？")
        with st.form("manipulation_form"):
            bias_awareness = st.radio(
                "你对 AI 评分的看法：",
                ["没有注意到明显的偏差", "注意到女性候选人得分普遍偏低", "注意到男性候选人得分普遍偏低", "注意到其他类型的偏差"],
                index=0
            )
            bias_detail = st.text_area("如果选择了“其他”，请具体描述：", placeholder="例如：AI 对某些学校背景的人打分偏低...")
            submitted = st.form_submit_button("提交")
            if submitted:
                manip_data = {"bias_awareness": bias_awareness, "bias_detail": bias_detail, "timestamp": datetime.now().isoformat()}
                with open(os.path.join(st.session_state.experiment_dir, "manipulation_check.json"), "w", encoding="utf-8") as f:
                    json.dump(manip_data, f, ensure_ascii=False, indent=2)
                st.session_state.manipulation_check_done = True
                st.session_state.questionnaire_step = 0   # 完成操纵检查，返回实验
                save_progress()
                st.rerun()
        st.stop()

    # 步骤2: 事后回顾（所有阶段完成后）
    if step == 2 and not st.session_state.debriefing_done:
        st.markdown("### 实验结束后的简短问卷")
        st.markdown("请根据您的真实感受回答以下问题")
        with st.form("debriefing_form"):
            fairness = st.slider("我觉得 AI 评分的公平性如何？", 1, 7, 4, help="1=非常不公平，7=非常公平")
            recall = st.slider("在后续独立决策时，我会不自觉地回忆起 AI 给出的分数。", 1, 7, 4)
            influence = st.slider("AI 辅助阶段对我最后的独立决策产生了很大影响。", 1, 7, 4)
            correction = st.radio(
                "在刚才的独立决策阶段（实验3），你是否有意识地尝试纠正你察觉到的偏差？",
                ["是，我尽量反着 AI 的建议来", "是，但我仍部分参考了 AI", "否，我认为 AI 的评分有道理", "我根本没注意到偏差"],
                index=3
            )
            comments = st.text_area("您对本次实验的任何其他反馈或感想：", placeholder="可选")
            submitted = st.form_submit_button("提交")
            if submitted:
                debrief_data = {
                    "fairness": fairness, "recall_ai_score": recall, "influence": influence,
                    "correction_behavior": correction, "comments": comments,
                    "timestamp": datetime.now().isoformat()
                }
                with open(os.path.join(st.session_state.experiment_dir, "debriefing.json"), "w", encoding="utf-8") as f:
                    json.dump(debrief_data, f, ensure_ascii=False, indent=2)
                st.session_state.debriefing_done = True
                st.session_state.questionnaire_step = 3   # 下一步依赖量表
                save_progress()
                st.rerun()
        st.stop()

    # 步骤3: 算法依赖量表
    if step == 3 and not st.session_state.dependency_done:
        st.markdown("### 最后一个问卷")
        st.markdown("请根据您的真实感受，对以下陈述进行评分（1=完全不同意，5=完全同意）")
        with st.form("dependency_form"):
            scores = []
            for i, item in enumerate(ALGORITHM_DEPENDENCY_ITEMS):
                if "压力" in item:
                    score = st.slider(item, 1, 7, 4, key=f"dep_{i}")
                else:
                    score = st.slider(item, 1, 5, 3, key=f"dep_{i}")
                scores.append(score)
            submitted = st.form_submit_button("提交量表")
            if submitted:
                dep_data = {"items": ALGORITHM_DEPENDENCY_ITEMS, "scores": scores, "total_score": sum(scores), "timestamp": datetime.now().isoformat()}
                with open(os.path.join(st.session_state.experiment_dir, "algorithm_dependency.json"), "w", encoding="utf-8") as f:
                    json.dump(dep_data, f, ensure_ascii=False, indent=2)
                st.session_state.dependency_done = True
                st.session_state.questionnaire_step = 4   # 完成
                save_progress()
                st.rerun()
        st.stop()

    # 步骤4: 生成汇总并显示感谢界面
    if step == 4:
        generate_summary_csv()
        st.markdown("### 🎉 实验完成")
        st.success("感谢您的决策与回答！您的数据已成功保存。")
        st.balloons()
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

# ===================== 自定义CSS =====================
st.markdown("""
    <style>
    .main-header { font-size: 28px; font-weight: bold; color: #2E86AB; margin-bottom: 8px; }
    .sub-header { font-size: 20px; font-weight: bold; color: #4A6FA5; margin: 20px 0 10px 0; }
    .job-desc-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #2E86AB; }
    .stage-progress { font-size: 16px; margin-bottom: 15px; color: #2C3E50; }
    .custom-decision-radio .stRadio > div { display: flex; justify-content: space-between; gap: 20px; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1rem !important; }
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
        target = st.session_state.target_hires
        def stats():
            if not st.session_state.decisions:
                return 0,0,0
            h = sum(1 for d in st.session_state.decisions.values() if d=="进入面试")
            p = sum(1 for d in st.session_state.decisions.values() if d=="待定")
            r = sum(1 for d in st.session_state.decisions.values() if d=="拒绝")
            return h,p,r
        hired, pending, rejected = stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ 已进入面试", f"{hired}/{target}")
        col2.metric("⏳ 待定", pending)
        col3.metric("❌ 拒绝", rejected)
        if hired > target:
            st.warning(f"⚠️ 已超过计划招聘人数（{target}）")
        else:
            st.info(f"剩余名额：{max(0, target-hired)}")
    else:
        st.info("请填写个人信息并开始实验")

# ===================== 主界面 =====================
if not st.session_state.info_collected:
    st.markdown('<div class="main-header">📝 招聘决策实验系统</div>', unsafe_allow_html=True)
    with st.form("experimenter_form"):
        exp_name = st.text_input("姓名")
        exp_id = st.text_input("学号/学校")
        exp_gender = st.radio("性别", ["男", "女"], horizontal=True)
        exp_age = st.number_input("年龄", 18, 100, 25)
        exp_major = st.text_input("专业")
        exp_education = st.selectbox("最高学历", ["本科", "硕士", "博士", "其他"])
        exp_ai_familiarity = st.slider("对人工智能的熟悉程度", 1, 7, 4)
        exp_recruitment_exp = st.radio("是否有招聘经验", ["有", "无"], horizontal=True)
        exp_similar_exp = st.radio("是否参加过类似的招聘实验", ["是", "否"], horizontal=True)

        st.markdown("#### 算法素养量表（1=完全不同意，7=完全同意）")
        alg_lit = []
        for i, q in enumerate(ALGORITHM_LITERACY_ITEMS):
            alg_lit.append(st.slider(f"{i+1}. {q}", 1, 7, 4, key=f"alg_{i}"))

        submitted = st.form_submit_button("开始实验", type="primary")
        if submitted and exp_name and exp_id:
            clean_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', exp_name)
            st.session_state.experimenter_id = f"{clean_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            exp_dir = f"experiment_data/{st.session_state.experimenter_id}"
            os.makedirs(exp_dir, exist_ok=True)
            st.session_state.experiment_dir = exp_dir
            st.session_state.experimenter_info = {
                "姓名": exp_name, "学号": exp_id, "性别": exp_gender, "年龄": exp_age,
                "专业": exp_major, "学历": exp_education, "AI熟悉程度": exp_ai_familiarity,
                "招聘经验": exp_recruitment_exp, "类似实验经验": exp_similar_exp,
                "实验者ID": st.session_state.experimenter_id, "任务压力条件": FIXED_PRESSURE_CONDITION
            }
            st.session_state.algorithm_literacy = alg_lit
            st.session_state.pressure_condition = FIXED_PRESSURE_CONDITION
            st.session_state.info_collected = True
            metadata = {"experimenter_id": st.session_state.experimenter_id, "start_time": datetime.now().isoformat(),
                        "experimenter_info": st.session_state.experimenter_info,
                        "algorithm_literacy_scores": alg_lit, "algorithm_literacy_items": ALGORITHM_LITERACY_ITEMS,
                        "pressure_condition": FIXED_PRESSURE_CONDITION, "stage_config": EXPERIMENT_STAGES}
            with open(os.path.join(exp_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            save_progress()
            st.rerun()
        elif submitted:
            st.warning("请填写姓名和学号")
    st.stop()

# 如果处于问卷流程，显示问卷界面
if st.session_state.questionnaire_step > 0:
    show_questionnaire()

# 压力提示
if st.session_state.pressure_condition == "高压力":
    st.warning("⚠️ 每个阶段决策必须在5分钟内完成")
    if st.session_state.current_stage not in st.session_state.stage_start_time:
        st.session_state.stage_start_time[st.session_state.current_stage] = time.time()
    elapsed = time.time() - st.session_state.stage_start_time[st.session_state.current_stage]
    if elapsed > 300:
        st.error("⚠️ 当前阶段已超过5分钟！")
    else:
        st.info(f"⏱️ 已用时：{int(elapsed//60)}分{int(elapsed%60)}秒 / 5分钟")
else:
    st.success("请按照您的真实想法进行决策")

if st.session_state.current_stage not in st.session_state.stage_start_time:
    st.session_state.stage_start_time[st.session_state.current_stage] = time.time()

current_cfg = get_stage_experiment_config(st.session_state.current_stage)
stage_names = [EXPERIMENT_STAGES[s]["name"] for s in get_stage_key_list()]
cur_idx = get_stage_key_list().index(st.session_state.current_stage)
st.markdown(f'<div class="stage-progress">实验阶段：{" → ".join([f"<b>{n}</b>" if i==cur_idx else n for i,n in enumerate(stage_names)])}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-header">{current_cfg["name"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div style="color:#666;">当前模式：{"AI辅助" if current_cfg["ai_assist"] else "无AI辅助"}</div>', unsafe_allow_html=True)
st.markdown(JOB_DESCRIPTION)
st.divider()

# 自动加载简历
if not st.session_state.resumes_uploaded:
    st.markdown('<div class="sub-header">📁 简历自动加载</div>', unsafe_allow_html=True)
    with st.spinner("正在加载简历..."):
        cands, errs = auto_load_candidates(RESUME_FOLDER, PHOTO_FOLDER)
        for e in errs:
            st.warning(e)
        if not cands:
            st.error("未找到有效简历，请检查resume文件夹")
            st.stop()
        st.session_state.candidates = cands
        if initialize_stage_data(st.session_state.current_stage):
            st.session_state.resumes_uploaded = True
            save_progress()
            st.rerun()
        else:
            st.stop()

# 招聘决策主界面
if st.session_state.resumes_uploaded:
    if st.session_state.stage_completed.get(st.session_state.current_stage, False):
        st.success(f"✅ {current_cfg['name']} 已完成！")
        next_key = get_next_stage(st.session_state.current_stage)
        if next_key is None:
            if st.button("📤 提交实验数据", type="primary", use_container_width=True):
                st.session_state.questionnaire_step = 2   # 开始事后回顾
                st.rerun()
        else:
            next_name = EXPERIMENT_STAGES[next_key]["name"]
            if st.button(f"➡️ 进入{next_name}", type="primary", use_container_width=True):
                switch_to_next_stage()
    else:
        if st.session_state.result_df.empty:
            st.error("数据加载错误")
            st.stop()

        st.markdown('<div class="sub-header">🎯 招聘决策标注</div>', unsafe_allow_html=True)
        if current_cfg["ai_assist"]:
            sort_key = st.selectbox("排序方式", list(SORT_OPTIONS.keys()), format_func=lambda x: SORT_OPTIONS[x]["name"], key="sort")
            sorted_df = sort_candidates_df(st.session_state.result_df, sort_key)
        else:
            sorted_df = st.session_state.result_df

        total = len(sorted_df)
        page_size = UI_CONFIG["max_candidates_per_page"]
        total_pages = max(1, (total + page_size - 1)//page_size)
        page = max(1, min(st.session_state.current_page, total_pages))
        start = (page-1)*page_size
        end = min(start+page_size, total)
        page_df = sorted_df.iloc[start:end]

        for _, row in page_df.iterrows():
            name = row["候选人姓名"]
            st.markdown(f"### 📄 【{row['排名']}】{name}")
            if current_cfg["show_score"]:
                st.markdown(f"<span style='color:#666;'>AI评分：{row['最终评分']}</span>", unsafe_allow_html=True)

            init_candidate_stay_time(st.session_state, name)
            update_candidate_stay_time(st.session_state, name)

            col1, col2, col3 = st.columns([1,2,2])
            with col1:
                photo = row.get("照片", "")
                if photo and os.path.exists(photo):
                    st.image(photo, width=150)
                else:
                    st.write("📷 无照片")
            with col2:
                st.write(f"**姓名**：{name}")
                st.write(f"**性别**：{row['性别']}")
                st.write(f"**年龄**：{row.get('年龄','未知')}")
                st.write(f"**学历**：{row['学历']}")
                st.write(f"**工作年限**：{row['工作年限']}年")
                st.write(f"**毕业院校**：{row.get('毕业院校','')} ({row.get('院校等级','')})")
            with col3:
                st.write(f"**技能**：{safe_list_to_str(row['技能'])}")
                st.write(f"**项目数**：{row['相关项目数']}")
                st.write(f"**证书**：{safe_list_to_str(row['证书'])}")
                st.write(f"**实习**：{row.get('实习经历','无')[:50]}...")
                st.write(f"**获奖**：{row.get('获奖情况','无')[:50]}...")

            if current_cfg["show_score"]:
                st.info(row["评分说明"])

            curr_dec = st.session_state.decisions.get(name, UI_CONFIG["decision_options"][1])
            decision = st.radio(
                f"决策",
                UI_CONFIG["decision_options"],
                index=UI_CONFIG["decision_options"].index(curr_dec),
                key=f"dec_{st.session_state.current_stage}_{name}",
                horizontal=True,
                label_visibility="collapsed"
            )
            if st.session_state.current_stage == "post":
                conf = st.slider(f"信心 (1-7)", 1, 7, st.session_state.post_confidence.get(name, 4), key=f"conf_{name}")
                st.session_state.post_confidence[name] = conf
            if decision != curr_dec:
                record_decision_time(name, decision, curr_dec)
                st.session_state.decisions[name] = decision
                st.rerun()
            st.divider()

        # 分页控件
        c1, c2, c3 = st.columns([1,2,1])
        with c1:
            if st.button("← 上一页", disabled=(page==1)):
                st.session_state.current_page -= 1
                st.session_state.scroll_to_top = True
                st.rerun()
        with c2:
            st.markdown(f"<div style='text-align:center'>第 {page} / {total_pages} 页</div>", unsafe_allow_html=True)
        with c3:
            if st.button("下一页 →", disabled=(page==total_pages)):
                st.session_state.current_page += 1
                st.session_state.scroll_to_top = True
                st.rerun()

        if is_stage_complete():
            st.success("✅ 本阶段决策已完成，点击下方按钮进入下一阶段")
            next_key = get_next_stage(st.session_state.current_stage)
            if next_key is None:
                if st.button("📤 提交实验数据", type="primary"):
                    st.session_state.questionnaire_step = 2
                    st.rerun()
            else:
                if st.button(f"➡️ 进入{EXPERIMENT_STAGES[next_key]['name']}", type="primary"):
                    switch_to_next_stage()
        else:
            missing = [n for n in st.session_state.result_df["候选人姓名"] if n not in st.session_state.decisions]
            if missing:
                st.warning(f"未决策：{', '.join(missing)}")
