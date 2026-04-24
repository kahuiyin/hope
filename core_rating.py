import pandas as pd
import numpy as np
from docx import Document
import os
import re
import time
import json
from datetime import datetime
import pathlib
from config import *


# ===================== 工具函数 =====================
def clean_temp_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"清理临时文件失败 {file_path}：{e}")
        try:
            os.rmdir(folder_path)
        except Exception as e:
            print(f"删除临时文件夹失败 {folder_path}：{e}")


def normalize_path(file_path):
    if not file_path:
        return ""
    try:
        return str(pathlib.Path(file_path).resolve())
    except Exception:
        return file_path


def safe_list_to_str(list_data, sep="；"):
    if isinstance(list_data, (np.ndarray, pd.Series)):
        list_data = list_data.tolist()
    if not isinstance(list_data, list):
        list_data = [str(list_data).strip()] if list_data is not None and str(list_data).strip() else []
    return sep.join([str(item).strip() for item in list_data if item and str(item).strip()])


def safe_str_to_list(str_data, sep="；"):
    if isinstance(str_data, (np.ndarray, pd.Series)):
        str_data = str_data.item() if str_data.size == 1 else ""
    if not isinstance(str_data, str):
        str_data = str(str_data).strip() if str_data is not None else ""
    if pd.isna(str_data) or str_data.strip() == "":
        return []
    return [item.strip() for item in str_data.split(sep) if item.strip()]


# ===================== 简历读取 =====================
def read_excel_resume(file_obj):
    candidates = []
    error_msg = None
    try:
        df = pd.read_excel(file_obj)
        base_cols = ["姓名", "性别", "学历", "工作年限", "技能", "相关项目数", "证书"]
        missing_base = [col for col in base_cols if col not in df.columns]
        if missing_base:
            raise ValueError(f"缺少必要列：{', '.join(missing_base)}")

        optional_cols = [
            "毕业院校", "院校等级", "专业", "自我评价", "实习经历", "获奖情况", "照片",
            "联系电话", "邮箱", "出生日期", "年龄"
        ]
        for col in optional_cols:
            if col not in df.columns:
                df[col] = ""

        df = df.fillna("")
        df["工作年限"] = pd.to_numeric(df["工作年限"], errors="coerce").fillna(0).astype(int)
        df["相关项目数"] = pd.to_numeric(df["相关项目数"], errors="coerce").fillna(0).astype(int)
        df["技能"] = df["技能"].apply(safe_str_to_list)
        df["证书"] = df["证书"].apply(safe_str_to_list)

        for _, row in df.iterrows():
            candidate = {
                "name": str(row["姓名"]).strip(),
                "gender": str(row["性别"]).strip(),
                "education": str(row["学历"]).strip(),
                "work_year": row["工作年限"],
                "skills": row["技能"],
                "related_project_num": row["相关项目数"],
                "certifications": row["证书"],
                "university": str(row["毕业院校"]).strip(),
                "university_rank": str(row["院校等级"]).strip(),
                "major": str(row["专业"]).strip(),
                "self_evaluation": str(row["自我评价"]).strip(),
                "internship": str(row["实习经历"]).strip(),
                "awards": str(row["获奖情况"]).strip(),
                "photo": str(row["照片"]).strip() if "照片" in df.columns else "",  # 【修改】直接保留文件名
                "phone": str(row["联系电话"]).strip(),
                "email": str(row["邮箱"]).strip(),
                "birthday": str(row["出生日期"]).strip(),
                "age": str(row["年龄"]).strip()
            }
            if candidate["name"]:
                candidates.append(candidate)

        if not candidates:
            raise ValueError("Excel中无有效候选人（姓名为空）")
    except Exception as e:
        error_msg = f"读取Excel失败：{str(e)}"
    return candidates, error_msg


def read_word_resume(file_path):
    candidate = None
    error_msg = None
    try:
        if not os.path.exists(file_path):
            raise ValueError("文件不存在")
        doc = Document(file_path)
        text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        if not text:
            raise ValueError("Word文件内容为空")

        name_patterns = [r"姓名[:：]\s*(\w+)", r"姓名\s*(\w+)", r"(\w+)\s*的个人简历"]
        gender_patterns = [r"性别[:：]\s*([男女])", r"性别\s*([男女])"]
        education_patterns = [r"学历[:：]\s*(大专|本科|硕士|博士)", r"学历\s*(大专|本科|硕士|博士)"]
        work_year_patterns = [r"工作年限[:：]\s*(\d+)年", r"工作经验[:：]\s*(\d+)年", r"从业年限[:：]\s*(\d+)年"]
        skills_patterns = [r"技能[:：]\s*(.*?)(\n|$)", r"掌握技能[:：]\s*(.*?)(\n|$)", r"技能特长[:：]\s*(.*?)(\n|$)"]
        project_num_patterns = [r"项目经历[:：]\s*(\d+)个", r"相关项目[:：]\s*(\d+)个"]
        cert_patterns = [r"证书[:：]\s*(.*?)(\n|$)", r"资质证书[:：]\s*(.*?)(\n|$)"]

        def extract_first_match(patterns, text):
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()
            return ""

        name = extract_first_match(name_patterns, text) or os.path.basename(file_path).split(".")[0]
        gender = extract_first_match(gender_patterns, text) or "未知"
        education = extract_first_match(education_patterns, text) or "本科"
        work_year = extract_first_match(work_year_patterns, text)
        skills = extract_first_match(skills_patterns, text)
        project_num = extract_first_match(project_num_patterns, text)
        certifications = extract_first_match(cert_patterns, text)

        candidate = {
            "name": name,
            "gender": gender,
            "education": education,
            "work_year": int(work_year) if work_year else 0,
            "skills": safe_str_to_list(skills),
            "related_project_num": int(project_num) if project_num else 0,
            "certifications": safe_str_to_list(certifications),
            "university": "",
            "university_rank": "",
            "major": "",
            "self_evaluation": "",
            "internship": "",
            "awards": "",
            "photo": "",  # Word文件无照片列
            "phone": "",
            "email": "",
            "birthday": "",
            "age": ""
        }
    except Exception as e:
        error_msg = f"读取失败：{str(e)}"
    return candidate, error_msg


def batch_read_word_resumes(folder_path):
    candidates = []
    errors = []
    if not os.path.exists(folder_path):
        return candidates, ["文件夹不存在"]
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".docx"):
            file_path = os.path.join(folder_path, file_name)
            candidate, err = read_word_resume(file_path)
            if err:
                errors.append(f"{file_name}：{err}")
            elif candidate and candidate["name"]:
                candidates.append(candidate)
    return candidates, errors


# ===================== 新增自动匹配照片和自动加载简历 =====================
def auto_match_photo(candidate_name, photo_folder=PHOTO_FOLDER, default_photo=UI_CONFIG["default_photo"],
                     photo_filename=None):
    """
    根据候选人姓名或提供的照片文件名在 photo_folder 中查找匹配的照片文件。
    如果提供了 photo_filename，则优先使用该文件名查找。
    否则，根据 candidate_name 尝试常见扩展名。
    如果找到则返回完整路径，否则返回空字符串。
    """
    # 1. 如果传入了照片文件名，优先使用
    if photo_filename:
        candidate_path = os.path.join(photo_folder, photo_filename)
        if os.path.exists(candidate_path):
            return candidate_path
        # 如果提供的文件名不存在，可以选择回退到姓名匹配，这里选择不回退（避免误匹配）

    # 2. 使用姓名匹配
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    for ext in extensions:
        candidate_path = os.path.join(photo_folder, candidate_name + ext)
        if os.path.exists(candidate_path):
            return candidate_path

    # 3. 尝试默认照片
    default_path = os.path.join(photo_folder, default_photo)
    if os.path.exists(default_path):
        return default_path

    return ""  # 无照片


def auto_load_candidates(resume_folder=RESUME_FOLDER, photo_folder=PHOTO_FOLDER):
    """
    自动读取 resume_folder 中的简历文件（.xlsx/.xls 和 .docx），
    并自动匹配 photo_folder 中的照片，返回 (candidates, errors)。
    """
    candidates = []
    errors = []
    if not os.path.exists(resume_folder):
        errors.append(f"简历文件夹不存在：{resume_folder}")
        return candidates, errors

    # 遍历文件夹中的所有文件
    for filename in os.listdir(resume_folder):
        filepath = os.path.join(resume_folder, filename)
        if not os.path.isfile(filepath):
            continue

        # 处理 Excel 文件（可包含多个候选人）
        if filename.lower().endswith(('.xlsx', '.xls')):
            file_obj = open(filepath, 'rb')
            try:
                cand_list, err = read_excel_resume(file_obj)
                if err:
                    errors.append(f"{filename}: {err}")
                else:
                    # 为每个候选人匹配照片
                    for cand in cand_list:
                        photo_filename = cand.get("photo", "")
                        photo_path = auto_match_photo(cand["name"], photo_folder, photo_filename=photo_filename)
                        cand["photo"] = photo_path
                        candidates.append(cand)
            finally:
                file_obj.close()
        # 处理 Word 文件（每个文件一个候选人）
        elif filename.lower().endswith('.docx'):
            cand, err = read_word_resume(filepath)
            if err:
                errors.append(f"{filename}: {err}")
            elif cand and cand["name"]:
                # Word简历没有照片列，用姓名匹配
                photo_path = auto_match_photo(cand["name"], photo_folder, photo_filename=None)
                cand["photo"] = photo_path
                candidates.append(cand)
            else:
                errors.append(f"{filename}: 未读取到有效候选人信息")
        # 其他文件忽略
        else:
            continue

    if not candidates:
        errors.append("未找到任何有效简历文件。请确保“resume”文件夹中包含 .xlsx/.xls 或 .docx 文件。")

    return candidates, errors


# ===================== 评分函数 =====================
def safe_get(candidate, key, default=""):
    val = candidate.get(key, default)
    if val is None:
        return default
    return val


def get_education_score(candidate):
    try:
        edu_map = {"大专": 2, "本科": 5, "硕士": 7, "博士": 8}
        edu_score = edu_map.get(safe_get(candidate, "education", "本科"), 0)

        school_val = JOB_REQUIREMENT["school_rank"].get(safe_get(candidate, "university_rank", "未知"), 0)
        max_school = max(JOB_REQUIREMENT["school_rank"].values())
        school_score = 8 * (school_val / max_school) if max_school > 0 else 0

        major = safe_get(candidate, "major", "").lower()
        preferred_majors = [m.lower() for m in JOB_REQUIREMENT["preferred_majors"]]

        # 专业匹配分 (0~6)
        if major in preferred_majors:
            major_score = 6
        elif any(pref in major for pref in preferred_majors):
            major_score = 4
        elif any(kw in major for kw in ["工程", "技术", "设计", "制造"]):
            major_score = 2
        else:
            major_score = 0

        total = edu_score + school_score + major_score
        return round(total, 2)
    except Exception as e:
        print(f"教育背景评分异常：{e}")
        return 0


def get_work_experience_score(candidate):
    try:
        work_year = safe_get(candidate, "work_year", 0)
        if isinstance(work_year, str):
            work_year = int(work_year) if work_year.isdigit() else 0
        if work_year <= 0:
            return 0
        base = 10
        inc = max(0, work_year - 2) * 3
        score = min(25, base + inc)
        return round(score, 2)
    except Exception as e:
        print(f"工作经验评分异常：{e}")
        return 0


def get_skills_score(candidate):
    try:
        skills = safe_get(candidate, "skills", [])
        if not skills:
            return 0
        total_skill_points = 0
        for skill in skills:
            level = JOB_REQUIREMENT["skill_keywords"].get(skill, "基础")
            points = JOB_REQUIREMENT["skill_level_scores"].get(level, 3)
            total_skill_points += points
        max_points = 48
        normalized = (total_skill_points / max_points) * 25
        return round(min(25, normalized), 2)
    except Exception as e:
        print(f"技能评分异常：{e}")
        return 0


def get_projects_score(candidate):
    try:
        project_num = safe_get(candidate, "related_project_num", 0)
        if isinstance(project_num, str):
            project_num = int(project_num) if project_num.isdigit() else 0
        score = min(15, project_num * 3)
        return round(score, 2)
    except Exception as e:
        print(f"项目评分异常：{e}")
        return 0


def get_certifications_score(candidate):
    try:
        certs = safe_get(candidate, "certifications", [])
        if not certs:
            return 0
        total_cert_points = 0
        for cert in certs:
            level = JOB_REQUIREMENT["cert_keywords"].get(cert, "基础")
            points = JOB_REQUIREMENT["cert_level_scores"].get(level, 2)
            total_cert_points += points
        max_points = 18
        normalized = (total_cert_points / max_points) * 5
        return round(min(5, normalized), 2)
    except Exception as e:
        print(f"证书评分异常：{e}")
        return 0


def get_internship_awards_score(candidate):
    try:
        internship = safe_get(candidate, "internship", "")
        awards = safe_get(candidate, "awards", "")

        score = 0
        # 实习加分：有内容则加分，根据文本长度再额外加分（最多3分）
        if internship and internship.strip():
            score += 3
            # 如果实习描述长度超过50字，再加1分；超过100字再加1分
            if len(internship) > 50:
                score += 1
            if len(internship) > 100:
                score += 1

        # 获奖加分：有内容则加分，根据文本长度再额外加分（最多3分）
        if awards and awards.strip():
            score += 3
            if len(awards) > 50:
                score += 1
            if len(awards) > 100:
                score += 1

        # 上限10分
        return round(min(10, score), 2)
    except Exception as e:
        print(f"实习获奖评分异常：{e}")
        return 0


def get_soft_skills_score(candidate):
    try:
        # 获取所有文本字段
        self_eval = safe_get(candidate, "self_evaluation", "").lower()
        internship = safe_get(candidate, "internship", "").lower()
        awards = safe_get(candidate, "awards", "").lower()
        full_text = f"{self_eval} {internship} {awards}"

        # 从 config 中获取软素质关键词
        keywords = JOB_REQUIREMENT.get("soft_keywords", [])
        if not keywords:
            # 如果 config 中没有，则使用内置扩展列表（实际不会发生）
            keywords = [
                "团队合作", "沟通能力", "协作", "跨部门", "协调", "对接", "客户",
                "责任心", "负责", "担当", "认真负责", "注重细节", "细心", "耐心", "严谨",
                "解决问题", "故障诊断", "分析", "总结", "优化", "改进", "创新", "新技术", "专利", "创新思维",
                "学习能力强", "快速学习", "掌握", "新技能", "半年内", "自学",
                "领导", "带领", "团队管理", "项目管理", "项目经验", "交付", "统筹",
                "抗压能力", "抗压", "压力", "紧急", "加班", "执行力", "落地", "攻坚",
                "动手能力", "独立完成", "行业经验", "深耕", "仿真软件", "精通", "熟悉",
                "全流程", "工艺瓶颈", "非标设备", "自动化产线", "仿真验证", "数控编程", "CNC",
                "机械理论", "实践经验", "设计任务", "质量体系", "过程管控", "设备调试", "故障排查",
                "技术创新", "成本意识", "材料选型", "加工工艺", "经济性", "复盘", "文档", "手册",
                "编写", "英语", "技术手册", "客户需求", "方案设计", "装配调试", "精益生产", "工时测算",
                "结构优化", "图纸审核", "热分析", "仿真验证", "标准制定", "工艺文件", "操作说明书",
                "逻辑清晰", "适应出差", "现场服务", "安全生产", "车间安全规范", "成本意识",
                "数据化分析", "模具设计", "低调务实", "落地效果", "新能源", "故障诊断", "非标",
                "智能化", "机器人", "PLC", "传感器", "视觉", "数字孪生", "3D打印", "增材制造",
                "产线改造", "设备维护", "点检", "寿命", "效率提升", "降本增效", "质量改善"
            ]

        # 去重计数
        matched = set()
        for kw in keywords:
            if kw in full_text:
                matched.add(kw)

        match_count = len(matched)

        # 基础分0，每匹配一个关键词+1，上限12分
        score = min(12, match_count)
        return round(score, 2)
    except Exception as e:
        print(f"软素质评分异常：{e}")
        return 5.0


# ========== generate_ai_analysis 支持传入 total_score ==========
def generate_ai_analysis(candidate, scores_dict, total_score=None):
    try:
        edu = scores_dict.get("education", 0)
        work = scores_dict.get("work_experience", 0)
        skills = scores_dict.get("skills", 0)
        projects = scores_dict.get("projects", 0)
        certs = scores_dict.get("certifications", 0)
        intern_awards = scores_dict.get("internship_awards", 0)
        soft = scores_dict.get("soft_skills", 0)

        # 如果提供了 total_score，则使用它作为总分；否则用原始求和
        if total_score is None:
            total = edu + work + skills + projects + certs + intern_awards + soft
        else:
            total = total_score

        analysis = []
        if edu >= 16:
            analysis.append(
                f"🎓 教育背景：{safe_get(candidate, 'university', '未知')}（{safe_get(candidate, 'university_rank', '未知')}）{safe_get(candidate, 'major', '未知')}专业，学历与院校水平优秀。")
        elif edu >= 10:
            analysis.append("🎓 教育背景：学历达标，院校/专业匹配度一般。")
        else:
            analysis.append("🎓 教育背景：学历或专业与岗位要求有一定差距。")
        if work >= 22:
            analysis.append(f"💼 工作经验：{safe_get(candidate, 'work_year', 0)}年经验，远超岗位要求，实践经历丰富。")
        elif work >= 15:
            analysis.append(f"💼 工作经验：{safe_get(candidate, 'work_year', 0)}年经验，接近或达到岗位要求。")
        else:
            analysis.append("💼 工作经验：经验较少，可能需要更多培养。")
        if skills >= 20:
            analysis.append("🔧 专业技能：掌握多项高级技能，与岗位高度匹配。")
        elif skills >= 12:
            analysis.append("🔧 专业技能：掌握部分核心技能，可进一步培训。")
        else:
            analysis.append("🔧 专业技能：技能匹配度较低，需补充学习。")
        if projects >= 12:
            analysis.append(f"📁 项目经历：{safe_get(candidate, 'related_project_num', 0)}个项目经验，表现突出。")
        elif projects >= 6:
            analysis.append("📁 项目经历：有项目经验，但数量或质量一般。")
        else:
            analysis.append("📁 项目经历：项目经验较少。")
        if certs >= 4:
            analysis.append("📜 证书资质：持有相关证书，证明专业能力。")
        elif certs > 0:
            analysis.append("📜 证书资质：有部分证书，但非核心。")
        else:
            analysis.append("📜 证书资质：无相关证书。")
        if intern_awards >= 4:
            analysis.append("🏆 实习与获奖：有实习经历和获奖情况，综合素质良好。")
        elif intern_awards > 0:
            analysis.append("🏆 实习与获奖：有实习或获奖经历，为加分项。")
        else:
            analysis.append("🏆 实习与获奖：无相关经历。")
        if soft >= 8:
            analysis.append("🧠 软素质：自我评价充分体现了优秀的工作态度和综合能力。")
        elif soft >= 6:
            analysis.append("🧠 软素质：自我评价体现较好的综合素质。")
        else:
            analysis.append("🧠 软素质：软素质描述较模糊，建议面试中进一步考察。")
        # 总体评价使用 total（可能是原始总分或有偏总分）
        if total >= 70:
            analysis.append("🌟 根据历史数据及简历综合分析：非常匹配，强烈建议进入面试。")
        elif total >= 50:
            analysis.append("📌 根据历史数据及简历综合分析：基本匹配，可进入面试进一步考察。")
        else:
            analysis.append("⚠️ 根据历史数据及简历综合分析：匹配度较低，建议待定或拒绝。")
        return "\n".join(analysis)
    except Exception as e:
        print(f"生成分析文本异常：{e}")
        return "AI分析生成失败"


def get_candidate_rating(candidate, bias_mode=False):
    try:
        edu_score = get_education_score(candidate)
        work_score = get_work_experience_score(candidate)
        skill_score = get_skills_score(candidate)
        project_score = get_projects_score(candidate)
        cert_score = get_certifications_score(candidate)
        intern_awards_score = get_internship_awards_score(candidate)
        soft_score = get_soft_skills_score(candidate)

        scores = {
            "education": edu_score,
            "work_experience": work_score,
            "skills": skill_score,
            "projects": project_score,
            "certifications": cert_score,
            "internship_awards": intern_awards_score,
            "soft_skills": soft_score
        }

        basic_score = sum(scores.values())
        final_score = basic_score

        if bias_mode:
            gender = safe_get(candidate, "gender", "未知").lower()
            if gender == "女":
                coeff = BIAS_CONFIG["gender"]["female_coeff"]
                final_score = basic_score * coeff
            elif gender == "男":
                coeff = BIAS_CONFIG["gender"]["male_coeff"]
                final_score = basic_score * coeff

        final_score = round(final_score, 2)  # 【修改】去掉 +20
        ai_analysis = generate_ai_analysis(candidate, scores, total_score=final_score)

        score_detail = f"【AI分析详情】\n{ai_analysis}"

        return {
            "候选人姓名": safe_get(candidate, "name", "未知"),
            "性别": safe_get(candidate, "gender", "未知"),
            "毕业院校": safe_get(candidate, "university", ""),
            "院校等级": safe_get(candidate, "university_rank", ""),
            "专业": safe_get(candidate, "major", ""),
            "学历": safe_get(candidate, "education", "本科"),
            "工作年限": safe_get(candidate, "work_year", 0),
            "技能": safe_get(candidate, "skills", []),
            "相关项目数": safe_get(candidate, "related_project_num", 0),
            "证书": safe_get(candidate, "certifications", []),
            "自我评价": safe_get(candidate, "self_evaluation", ""),
            "实习经历": safe_get(candidate, "internship", ""),
            "获奖情况": safe_get(candidate, "awards", ""),
            "照片": safe_get(candidate, "photo", ""),
            "联系电话": safe_get(candidate, "phone", ""),
            "邮箱": safe_get(candidate, "email", ""),
            "出生日期": safe_get(candidate, "birthday", ""),
            "年龄": safe_get(candidate, "age", ""),
            "无偏见基础分": basic_score,
            "最终评分": final_score,
            "评分说明": score_detail,
            "偏见模式": "开启" if bias_mode else "关闭"
        }
    except Exception as e:
        print(f"候选人评分彻底失败: {e}")
        return {
            "候选人姓名": safe_get(candidate, "name", "未知"),
            "性别": safe_get(candidate, "gender", "未知"),
            "毕业院校": safe_get(candidate, "university", ""),
            "院校等级": safe_get(candidate, "university_rank", ""),
            "专业": safe_get(candidate, "major", ""),
            "学历": safe_get(candidate, "education", "本科"),
            "工作年限": safe_get(candidate, "work_year", 0),
            "技能": safe_get(candidate, "skills", []),
            "相关项目数": safe_get(candidate, "related_project_num", 0),
            "证书": safe_get(candidate, "certifications", []),
            "自我评价": safe_get(candidate, "self_evaluation", ""),
            "实习经历": safe_get(candidate, "internship", ""),
            "获奖情况": safe_get(candidate, "awards", ""),
            "照片": safe_get(candidate, "photo", ""),
            "联系电话": safe_get(candidate, "phone", ""),
            "邮箱": safe_get(candidate, "email", ""),
            "出生日期": safe_get(candidate, "birthday", ""),
            "年龄": safe_get(candidate, "age", ""),
            "无偏见基础分": 0,
            "最终评分": 0,
            "评分说明": "评分过程发生错误，使用默认值",
            "偏见模式": "错误"
        }


def batch_rating(candidates_list, bias_mode=False):
    if not candidates_list:
        return pd.DataFrame()
    result_list = []
    for c in candidates_list:
        try:
            result_list.append(get_candidate_rating(c, bias_mode))
        except Exception as e:
            print(f"批量评分中候选人 {c.get('name', '未知')} 失败: {e}")
            result_list.append({
                "候选人姓名": c.get("name", "未知"),
                "性别": c.get("gender", "未知"),
                "毕业院校": c.get("university", ""),
                "院校等级": c.get("university_rank", ""),
                "专业": c.get("major", ""),
                "学历": c.get("education", "本科"),
                "工作年限": c.get("work_year", 0),
                "技能": c.get("skills", []),
                "相关项目数": c.get("related_project_num", 0),
                "证书": c.get("certifications", []),
                "自我评价": c.get("self_evaluation", ""),
                "实习经历": c.get("internship", ""),
                "获奖情况": c.get("awards", ""),
                "照片": c.get("photo", ""),
                "联系电话": c.get("phone", ""),
                "邮箱": c.get("email", ""),
                "出生日期": c.get("birthday", ""),
                "年龄": c.get("age", ""),
                "无偏见基础分": 0,
                "最终评分": 0,
                "评分说明": "评分失败",
                "偏见模式": "错误"
            })
    result_df = pd.DataFrame(result_list)
    if not result_df.empty:
        result_df["排名"] = result_df["最终评分"].rank(method="min", ascending=False).astype(int)
        result_df = result_df.sort_values(["排名", "候选人姓名"])
    return result_df


# ===================== 停留时间统计 =====================
def init_candidate_stay_time(session_state, candidate_name):
    if "candidate_stay_time" not in session_state:
        session_state.candidate_stay_time = {}
    if candidate_name not in session_state.candidate_stay_time:
        session_state.candidate_stay_time[candidate_name] = {
            "start_time": time.time(),
            "last_update": time.time(),
            "total_seconds": 0,
            "candidate": candidate_name,
            "experiment_stage": session_state.get("current_stage", "unknown")
        }


def update_candidate_stay_time(session_state, candidate_name):
    if "candidate_stay_time" in session_state and candidate_name in session_state.candidate_stay_time:
        now = time.time()
        time_diff = now - session_state.candidate_stay_time[candidate_name]["last_update"]
        if time_diff < 60:
            session_state.candidate_stay_time[candidate_name]["total_seconds"] += time_diff
        session_state.candidate_stay_time[candidate_name]["last_update"] = now


def end_candidate_stay_time(session_state, candidate_name):
    if "candidate_stay_time" in session_state and candidate_name in session_state.candidate_stay_time:
        now = time.time()
        time_diff = now - session_state.candidate_stay_time[candidate_name]["last_update"]
        if time_diff < 60:
            session_state.candidate_stay_time[candidate_name]["total_seconds"] += time_diff
        session_state.candidate_stay_time[candidate_name]["last_update"] = now


def save_candidate_stay_time_data(session_state, save_path="stay_time_candidates.json"):
    if "candidate_stay_time" not in session_state:
        return False, "无停留时间数据"
    valid_data = []
    for cand, data in session_state.candidate_stay_time.items():
        if data["total_seconds"] >= STAY_TIME_CONFIG["ignore_seconds"]:
            valid_data.append({
                "candidate": cand,
                "experiment_stage": data["experiment_stage"],
                "total_seconds": round(data["total_seconds"], 2),
                "start_time": datetime.fromtimestamp(data["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
                "is_valid": data["total_seconds"] >= STAY_TIME_CONFIG["ignore_seconds"]
            })
    try:
        existing_data = []
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        combined_data = existing_data + valid_data
        seen = set()
        unique_data = []
        for item in combined_data:
            key = (item["candidate"], item["start_time"])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
        return True, f"成功保存{len(valid_data)}条候选人停留时间数据"
    except Exception as e:
        return False, f"保存失败：{str(e)}"


# ===================== 实验数据处理 =====================
def get_stage_experiment_config(stage_key):
    return EXPERIMENT_STAGES.get(stage_key, EXPERIMENT_STAGES["pre"])


def sort_candidates_df(df, sort_key):
    if df.empty or sort_key not in SORT_OPTIONS:
        return df
    df_copy = df.copy()
    sort_config = SORT_OPTIONS[sort_key]
    if sort_key == "school_rank_desc":
        df_copy["院校等级分"] = df_copy["院校等级"].map(JOB_REQUIREMENT["school_rank"]).fillna(0)
        df_copy = df_copy.sort_values(by="院校等级分", ascending=False)
    else:
        df_copy = df_copy.sort_values(
            by=sort_config["col"],
            ascending=sort_config["ascending"],
            na_position="last"
        )
    df_copy["排名"] = range(1, len(df_copy) + 1)
    return df_copy
