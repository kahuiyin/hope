import os
import random
import time
import shutil
import pandas as pd
from datetime import datetime, timedelta
import sys

# ========== 导入 core_rating 评分所需模块 ==========
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import JOB_REQUIREMENT
from core_rating import (
    get_education_score,
    get_work_experience_score,
    get_skills_score,
    get_projects_score,
    get_certifications_score,
    get_internship_awards_score,
    get_soft_skills_score,
    safe_str_to_list
)
# =====================================================

# ===================== 配置参数 =====================
PHOTO_FOLDER = "photos"                # 最终生成的照片存放目录
EXCEL_FILE = "resume.xlsx"
CANDIDATE_COUNT = 60                   # 候选简历总数（男女各半）
MATCH_SCORE_DIFF = 1                   # 评分相差1分内可匹配
MATCH_AGE_DIFF = 3                     # 年龄相差3岁内可匹配
DOWNLOAD_PHOTOS = True                 # 是否复制照片（True=复制本地照片，False=不复制）
API_RETRY = 2                          # 保留未使用（但保留变量）
SLEEP_BETWEEN_PHOTOS = 1.5             # 保留未使用

# 本地照片库路径（按性别分子文件夹）
PHOTO_LIBRARY_MALE = r"E:\办公\毕业设计\pre-coding\人脸样本集\male"
PHOTO_LIBRARY_FEMALE = r"E:\办公\毕业设计\pre-coding\人脸样本集\female"

# 分值区间配置（低/中/高）
SCORE_LEVEL_CONFIG = {
    "low": {"min": 30, "max": 50},
    "medium": {"min": 50, "max": 75},
    "high": {"min": 75, "max": 95}
}
SCORE_LEVEL_RATIO = {"low": 0.3, "medium": 0.4, "high": 0.3}

# ===================== 多维度自我评价池 =====================
# 每个维度下包含足够多的短句，确保组合时多样性
self_eval_dims = {
    "work_attitude": [
        "工作认真负责，责任心强。",
        "抗压能力强，能应对紧急项目。",
        "注重细节，追求零缺陷交付。",
        "积极主动，善于主动解决问题。",
        "具备良好的职业道德和保密意识。",
        "执行力强，能高效完成既定目标。",
        "勇于承担责任，不推诿。",
        "踏实肯干，适应高强度工作。",
        "严谨细致，注重过程管控。",
        "有强烈的主人翁意识，主动推进工作。",
        "善于自我驱动，无需督促。",
        "具备良好的抗挫折能力，面对困难不退缩。",
        "注重安全生产，熟悉车间安全规范。",
        "善于在压力下保持冷静，高效应对挑战。",
    ],
    "technical_skill": [
        "熟练掌握SolidWorks、AutoCAD等设计软件。",
        "精通机械原理与制造工艺。",
        "具备有限元分析能力，能独立完成结构仿真。",
        "熟悉数控编程与CNC操作。",
        "了解PLC编程与自动化控制。",
        "擅长非标设备设计与优化。",
        "熟悉TS16949质量体系，注重过程管控。",
        "熟练使用ANSYS/ABAQUS进行结构强度分析。",
        "精通机电一体化设计，能独立完成机电系统集成。",
        "熟悉机器人编程与数字孪生技术。",
        "具备模具设计与制造全流程经验。",
        "熟练掌握液压与气动系统设计。",
        "熟悉智能制造相关技术，能结合物联网进行产线升级。",
        "精通3D打印与增材制造工艺。",
        "熟悉工业机器人应用方案设计。",
        "掌握六西格玛、精益生产等工具。",
    ],
    "project_exp": [
        "曾主导3款自动化设备开发，交付后运行稳定。",
        "参与多个大型项目，负责关键模块设计。",
        "有完整的项目开发经验，从概念到量产全程跟进。",
        "多次在项目中担任核心角色，推动技术难题攻关。",
        "具备项目管理能力，曾带领5人团队完成项目。",
        "善于总结复盘，形成知识库，优化流程。",
        "曾主导成本降低项目，单件成本降低15%。",
        "有多次专利申请经验，成功授权多项实用新型专利。",
        "负责过跨部门协作项目，有效协调研发、生产、采购。",
        "曾主导设备改造项目，提升设备OEE 20%。",
        "参与过千万级项目，负责技术方案制定与实施。",
        "有海外项目经验，与国外团队协同完成开发。",
        "负责过客户定制化项目，满意度100%。",
        "曾主导质量改善项目，降低不良率30%。",
        "负责过产线布局优化，提升生产效率25%。",
    ],
    "soft_skill": [
        "具备良好的团队合作精神和沟通能力。",
        "善于跨部门协作，能有效对接研发、生产、采购。",
        "学习能力强，能快速掌握新工具和新领域知识。",
        "具备客户思维，能准确理解并转化需求。",
        "逻辑清晰，善于分析和总结。",
        "适应出差和现场服务，客户满意度高。",
        "具备良好的英语读写能力，能阅读英文技术手册。",
        "善于组织协调，能高效推动会议决策。",
        "乐于分享知识，经常组织技术培训。",
        "具备良好的时间管理能力，能合理规划优先级。",
        "善于与不同性格的人合作，能化解团队冲突。",
        "具备较强的商业敏感度，能结合市场进行产品定义。",
        "有良好的演讲能力，能自信向客户展示方案。",
        "善于从失败中学习，形成改进措施。",
        "具备良好的文档编写能力，能撰写高质量报告。",
    ]
}

# 定义不同分值区间的组合策略（每个策略指定选取哪些维度及每个维度选取几条）
self_eval_strategy = {
    "low": {
        "dims": ["work_attitude", "soft_skill"],      # 低分段仅选取工作态度和软素质
        "counts": [1, 1]                              # 每个维度取1条
    },
    "medium": {
        "dims": ["work_attitude", "technical_skill", "soft_skill"],  # 中分段增加技术技能
        "counts": [1, 1, 1]
    },
    "high": {
        "dims": ["work_attitude", "technical_skill", "project_exp", "soft_skill"],  # 高分段全维度
        "counts": [1, 1, 1, 1]
    }
}

# 真实大学列表（985、211、双一流、普通一本）
universities = [
    # 985 院校（分值10）
    ("清华大学", "985", 10), ("北京大学", "985", 10), ("复旦大学", "985", 10), ("上海交通大学", "985", 10),
    ("浙江大学", "985", 10), ("中国科学技术大学", "985", 10), ("南京大学", "985", 10), ("西安交通大学", "985", 10),
    ("哈尔滨工业大学", "985", 10), ("武汉大学", "985", 10), ("华中科技大学", "985", 10), ("中山大学", "985", 10),
    ("四川大学", "985", 10), ("南开大学", "985", 10), ("天津大学", "985", 10), ("山东大学", "985", 10),
    ("东南大学", "985", 10), ("吉林大学", "985", 10), ("厦门大学", "985", 10), ("同济大学", "985", 10),
    ("北京师范大学", "985", 10), ("国防科技大学", "985", 10), ("中国人民大学", "985", 10), ("兰州大学", "985", 10),
    # 211 院校（分值8）
    ("北京工业大学", "211", 8), ("北京化工大学", "211", 8), ("北京邮电大学", "211", 8), ("北京林业大学", "211", 8),
    ("北京中医药大学", "211", 8), ("北京外国语大学", "211", 8), ("中国传媒大学", "211", 8), ("中央财经大学", "211", 8),
    ("对外经济贸易大学", "211", 8), ("中国政法大学", "211", 8), ("华北电力大学", "211", 8), ("中国矿业大学", "211", 8),
    ("中国石油大学", "211", 8), ("中国地质大学", "211", 8), ("上海外国语大学", "211", 8), ("东华大学", "211", 8),
    ("上海财经大学", "211", 8), ("南京航空航天大学", "211", 8), ("南京理工大学", "211", 8), ("河海大学", "211", 8),
    ("江南大学", "211", 8), ("南京农业大学", "211", 8), ("中国药科大学", "211", 8), ("合肥工业大学", "211", 8),
    ("武汉理工大学", "211", 8), ("华中农业大学", "211", 8), ("华中师范大学", "211", 8), ("中南财经政法大学", "211", 8),
    ("西南交通大学", "211", 8), ("电子科技大学", "211", 8), ("西南财经大学", "211", 8), ("西安电子科技大学", "211", 8),
    ("长安大学", "211", 8), ("西北大学", "211", 8), ("陕西师范大学", "211", 8), ("福州大学", "211", 8),
    ("南昌大学", "211", 8), ("郑州大学", "211", 8), ("湖南师范大学", "211", 8), ("暨南大学", "211", 8),
    ("华南师范大学", "211", 8), ("广西大学", "211", 8), ("云南大学", "211", 8), ("贵州大学", "211", 8),
    ("新疆大学", "211", 8), ("宁夏大学", "211", 8), ("青海大学", "211", 8), ("石河子大学", "211", 8),
    # 双一流建设高校（非211，分值7）
    ("中国科学院大学", "双一流（非211）", 7), ("外交学院", "双一流（非211）", 7), ("首都师范大学", "双一流（非211）", 7),
    ("天津工业大学", "双一流（非211）", 7), ("天津医科大学", "双一流（非211）", 7), ("天津中医药大学", "双一流（非211）", 7),
    ("上海海洋大学", "双一流（非211）", 7), ("上海中医药大学", "双一流（非211）", 7), ("南京邮电大学", "双一流（非211）", 7),
    ("南京林业大学", "双一流（非211）", 7), ("南京信息工程大学", "双一流（非211）", 7),
    ("南京中医药大学", "双一流（非211）", 7),
    ("中国美术学院", "双一流（非211）", 7), ("河南大学", "双一流（非211）", 7), ("广州中医药大学", "双一流（非211）", 7),
    ("成都理工大学", "双一流（非211）", 7), ("西南石油大学", "双一流（非211）", 7), ("宁波大学", "双一流（非211）", 7),
    ("山西大学", "双一流（非211）", 7), ("湘潭大学", "双一流（非211）", 7),
    # 普通一本（分值6）
    ("燕山大学", "一本", 6), ("浙江工业大学", "一本", 6), ("南京工业大学", "一本", 6), ("江苏大学", "一本", 6),
    ("扬州大学", "一本", 6), ("杭州电子科技大学", "一本", 6), ("重庆邮电大学", "一本", 6),
    ("西安建筑科技大学", "一本", 6),
    ("昆明理工大学", "一本", 6), ("上海理工大学", "一本", 6), ("湖北大学", "一本", 6), ("华侨大学", "一本", 6),
    ("青岛大学", "一本", 6), ("广州大学", "一本", 6), ("深圳大学", "一本", 6), ("汕头大学", "一本", 6),
    ("长春理工大学", "一本", 6), ("哈尔滨理工大学", "一本", 6), ("沈阳工业大学", "一本", 6),
    ("大连交通大学", "一本", 6),
    # 二本/其他（分值5）
    ("山东科技大学", "二本/其他", 5), ("青岛科技大学", "二本/其他", 5), ("济南大学", "二本/其他", 5),
    ("烟台大学", "二本/其他", 5),
    ("辽宁科技大学", "二本/其他", 5), ("沈阳化工大学", "二本/其他", 5), ("西安工业大学", "二本/其他", 5),
    ("陕西科技大学", "二本/其他", 5)
]

# 扩充后的机械类专业列表
majors = [
    "机械工程", "机械设计制造及其自动化", "车辆工程", "机电一体化技术", "材料成型及控制工程",
    "过程装备与控制工程", "工业设计", "机械电子工程", "智能制造工程", "工业工程",
    "工程机械运用技术", "汽车服务工程", "机械工艺技术", "焊接技术与工程", "模具设计与制造",
    "数控技术", "精密机械技术", "液压与气动技术", "机械装备制造技术", "航空航天工程（机械方向）",
    "农业机械化及其自动化", "船舶与海洋工程（机械方向）", "铁道机车车辆制造与维护", "新能源汽车工程",
    "智能装备与系统", "增材制造工程", "机器人工程", "轴承设计与制造", "矿山机械工程",
    "纺织机械设计与自动化", "包装工程（机械方向）", "制冷与空调工程", "起重运输机械工程",
    "冶金机械工程", "化工机械", "医疗设备工程（机械方向）", "智能焊接技术", "机械维修及检测技术教育"
]

# 技能池（按技能等级分档）
skills_pool = {
    3: ["机械设计", "CAD", "机械原理", "设备维护", "机械制图", "车铣刨磨加工", "设备点检", "基础工艺"],  # 基础技能（3分/个）
    5: ["SolidWorks", "UG/NX", "Pro/E/Creo", "AutoCAD", "数控编程", "CNC操作", "焊接工艺", "模具设计"],  # 中级技能（5分/个）
    8: ["ANSYS", "ABAQUS", "ADAMS", "PLC", "有限元分析", "伺服控制系统", "机器人编程", "数字孪生"],  # 高级技能（8分/个）
}

# 证书池（按等级分档）
cert_pool = {
    2: ["数控车工（高级/技师）", "数控铣工（高级/技师）", "焊工（高级/技师）", "电工证（高低压）"],  # 基础证书（2分/个）
    4: ["机械工程师", "CAD工程师", "SolidWorks认证", "模具设计师（中级/高级）"],  # 中级证书（4分/个）
    6: ["注册机械工程师", "PMP项目管理师", "六西格玛绿带/黑带", "智能制造工程师"],  # 高级证书（6分/个）
}

# 实习经历
internships = [
    "2023.07-2023.12 在某机械制造公司实习，参与自动化设备调试，协助完成CAD图纸修改。",
    "2024.01-2024.06 在某汽车零部件厂实习，负责质量检测与工艺优化。",
    "2023.09-2024.01 参与校内智能制造实验室项目，完成机械臂运动仿真。",
    "2022.07-2022.09 在某重工企业实习，协助工程师进行产品结构设计。",
    "2023.03-2023.06 在某数控设备厂实习，学习数控编程与CNC机床操作，独立完成简单零件加工。",
    "2024.02-2024.05 在某新能源汽车厂实习，参与电池包夹具设计，跟进样件试制与验证。",
    "2023.08-2023.11 在某工程机械厂实习，协助完成挖掘机底盘结构有限元分析，优化结构强度。",
    "2022.09-2023.01 在某模具厂实习，学习注塑模具设计，参与模具拆装和试模过程记录。",
    "2024.04-2024.07 在某3D打印公司实习，负责产品三维模型修复，完成小批量样件打印与后处理。",
    "2023.01-2023.04 在某自动化设备公司实习，参与PLC程序编写和触摸屏界面设计，完成设备调试。",
    "2022.06-2022.08 在某轴承制造企业实习，学习轴承加工工艺，协助完成尺寸检测和质量统计。",
    "2023.10-2024.01 在某航空零部件厂实习，参与航空零件数控加工工艺编制，熟悉航空件质量要求。",
    "2024.03-2024.06 在某医疗设备公司实习，协助完成医用机械臂结构设计，遵循医疗产品设计规范。",
    "2022.10-2023.01 在某冶金机械厂实习，参与轧钢设备维护与改造，记录设备故障和维修过程。",
    "2023.05-2023.08 在某液压设备厂实习，学习液压系统原理图绘制，协助完成液压站装配与调试。",
    "2024.01-2024.04 在某机器人公司实习，参与工业机器人应用方案设计，完成机器人轨迹调试。",
    "2022.07-2022.10 在某农机企业实习，协助完成播种机结构优化，跟进田间试验数据收集。",
    "2023.06-2023.09 在某包装机械公司实习，参与全自动包装机设计，完成零部件选型和图纸审核。",
    "2024.05-2024.08 在某船舶机械厂实习，学习船舶动力系统零部件加工工艺，协助完成装配指导。",
    "2022.08-2022.11 在某制冷设备厂实习，参与空调压缩机结构设计，完成热分析仿真验证。",
    "2023.02-2023.05 在某矿山机械公司实习，协助完成破碎机耐磨件设计，优化材料选型。",
    "2024.06-2024.09 在某非标设备厂实习，全程参与自动化上下料设备开发，从方案设计到现场调试。",
    "2022.11-2023.02 在某表面处理厂实习，学习电镀、喷涂工艺，协助完成产品表面质量检测。",
    "2023.09-2023.12 在某精益生产咨询公司实习，协助企业完成生产线布局优化和工时测算。",
    "2024.07-2024.09 在某高校实验室实习，参与国家自然科学基金项目，完成机械结构创新设计。",
    "2022.05-2022.07 在某汽车改装厂实习，参与越野车底盘改装，完成零部件加工和装配调试。",
    "2023.04-2023.07 在某齿轮厂实习，学习齿轮加工工艺（滚齿、磨齿），协助完成齿轮精度检测。",
    "2024.08-2024.10 在某检测机构实习，学习无损检测技术，协助完成机械零件缺陷检测。",
    "2022.12-2023.03 在某智能仓储公司实习，参与AGV小车机械结构设计，完成行走机构调试。",
    "2023.07-2023.10 在某家电企业实习，参与洗衣机内桶模具设计，跟进模具试模和改进。"
]

# 获奖情况
awards = [
    "全国大学生机械创新设计大赛省一等奖",
    "校一等奖学金",
    "全国三维数字化创新设计大赛二等奖",
    "优秀毕业生",
    "国家励志奖学金",
    "挑战杯省级二等奖",
    "全国大学生工程训练综合能力竞赛国家级三等奖",
    "机械创新设计大赛国家级二等奖",
    "省级优秀学生干部",
    "校级三好学生",
    "校二等奖学金",
    "校三等奖学金",
    "校级机械设计大赛一等奖",
    "省级数控技能大赛三等奖",
    "全国大学生先进成图技术与产品信息建模创新大赛省级一等奖",
    "挑战杯大学生创业计划竞赛校级金奖",
    "互联网+大学生创新创业大赛省级铜奖",
    "校级优秀实习标兵",
    "企业奖学金（XX重工）",
    "校级科研创新奖",
    "全国大学生数学建模竞赛省级二等奖",
    "校级节能减排社会实践与科技竞赛一等奖",
    "省级大学生智能制造大赛二等奖",
    "全国大学生机器人竞赛（RoboMaster）区域赛三等奖",
    "校级专利申请专项奖",
    "省级优秀毕业论文",
    "校运动会科技创新类项目一等奖",
    "校企合作项目优秀贡献奖",
    "全国大学生工业设计大赛省级优秀奖",
    "校级机械工艺创新大赛二等奖",
    "省级大学生创新创业训练计划项目优秀结题",
    "校级优秀共青团员",
    "车间实习优秀学员（XX机械厂）",
    "全国大学生英语竞赛C类三等奖",
    "校级计算机技能大赛（CAD方向）一等奖",
    "省级大学生金相技能大赛二等奖",
    "全国大学生物理实验竞赛省级三等奖",
    "校级学风建设先进个人",
    "省级大学生职业规划大赛优秀奖",
    "校机械工程学院学术之星",
    "全国三维数字化创新设计大赛（3DDS）全国总决赛优秀奖",
    "省级大学生机械工程创新创意大赛一等奖",
    "校级社会实践先进个人",
    "企业实习最佳新人奖（XX自动化公司）",
    "校级毕业设计（论文）一等奖",
    "省级大学生工业机器人技术应用大赛三等奖",
    "全国大学生节能减排竞赛校级二等奖",
    "校级创新创业孵化项目一等奖",
    "省级大学生材料成型工艺大赛优秀奖"
]

# 姓名池
male_names = [
    "张伟", "王伟", "王磊", "李伟", "李勇", "李军", "李强", "刘洋", "刘浩", "刘强",
    "王强", "王鹏", "王浩", "李明", "李涛", "李波", "刘杰", "刘军", "刘兵", "刘波",
    "陈峰", "陈明", "陈宇", "陈杰", "陈俊", "杨超", "杨勇", "杨浩", "杨明", "杨军",
    "黄涛", "黄浩", "黄勇", "黄杰", "周健", "周强", "周鹏", "周宇", "吴刚", "吴浩",
    "吴杰", "吴涛", "赵阳", "赵伟", "赵鹏", "赵杰", "胡强", "胡浩", "胡军", "胡涛",
    "孙杰", "孙浩", "孙鹏", "孙伟", "朱涛", "朱浩", "朱强", "马涛", "马浩", "马杰",
    "郭强", "郭浩", "郭鹏", "何涛", "何杰", "高杰", "高浩", "林浩", "林杰", "罗浩",
    "郑涛", "郑杰", "郑强", "梁杰", "谢涛", "宋杰", "唐浩", "韩杰", "曹涛", "许杰",
    "邓浩", "冯杰", "彭涛", "曾杰", "肖浩", "田杰", "董涛", "袁杰", "潘浩", "于杰",
    "蒋涛", "蔡杰", "余浩", "杜杰", "叶涛", "程杰", "魏浩", "苏杰", "吕涛", "丁杰",
    "张子轩", "李浩然", "王梓涵", "刘俊熙", "陈雨泽", "杨宇辰", "黄俊凯", "周煜宸", "吴浩宇", "赵奕辰",
    "胡博文", "孙泽宇", "朱嘉豪", "马子昂", "郭宇泽", "何俊豪", "高梓轩", "林宸宇", "罗子涵", "郑浩然",
    "梁宇轩", "谢俊熙", "宋雨泽", "唐宇辰", "韩俊凯", "曹煜宸", "许浩宇", "邓奕辰", "冯博文", "彭泽宇",
    "曾嘉豪", "肖子昂", "田宇泽", "董俊豪", "袁梓轩", "潘宸宇", "于子涵", "蒋浩然", "蔡宇轩", "余俊熙",
    "杜雨泽", "叶宇辰", "程俊凯", "魏煜宸", "苏浩宇", "吕奕辰", "丁博文", "沈泽宇", "姜嘉豪", "范子昂",
    "江宇泽", "傅俊豪", "钟梓轩", "卢宸宇", "汪子涵", "戴浩然", "崔宇轩", "任俊熙", "陆雨泽", "廖宇辰",
    "姚俊凯", "方煜宸", "金浩宇", "邱奕辰", "夏博文", "石泽宇", "谭嘉豪", "韦子昂", "贾宇泽", "邹俊豪",
    "熊梓轩", "白宸宇", "孟子涵", "秦浩然", "阎宇轩", "薛俊熙", "侯雨泽", "雷宇辰", "龙俊凯", "史煜宸",
    "陶浩宇", "黎奕辰", "贺博文", "顾泽宇", "毛嘉豪", "郝子昂", "邵宇泽", "孔俊豪", "邵梓轩", "史宸宇",
    "尹子涵", "常浩然", "段宇轩", "汤俊熙", "尹雨泽", "易宇辰", "黎俊凯", "武煜宸", "乔浩宇", "贺奕辰",
    "龚博文", "文泽宇", "樊嘉豪", "葛子昂", "邢宇泽", "安俊豪", "齐梓轩", "庄宸宇", "殷子涵", "章浩然",
    "鲁宇轩", "倪俊熙", "庞雨泽", "邢宇辰", "俞俊凯", "葛煜宸", "邢浩宇", "齐奕辰", "纪博文", "舒泽宇",
    "屈嘉豪", "樊子昂", "焦宇泽", "鲁俊豪", "车梓轩", "房宸宇", "裘子涵", "缪浩然", "解宇轩", "宁俊熙",
    "尚雨泽", "容宇辰", "向俊凯", "葛煜宸", "边浩宇", "卞奕辰", "冀博文", "童泽宇", "颜嘉豪", "盛子昂"
]

female_names = [
    "李娜", "李芳", "李静", "李娟", "李婷", "王静", "王丽", "王芳", "王娟", "王婷",
    "刘敏", "刘婷", "刘芳", "刘静", "刘娜", "陈丽", "陈静", "陈芳", "陈婷", "陈敏",
    "赵敏", "赵静", "赵芳", "赵娜", "杨敏", "杨婷", "杨静", "杨芳", "杨娜", "黄静",
    "黄婷", "黄芳", "黄敏", "周雅", "周静", "周婷", "周芳", "吴雪", "吴敏", "吴婷",
    "郑慧", "郑敏", "郑婷", "郑芳", "林娜", "林静", "林婷", "林芳", "郭敏", "郭婷",
    "郭静", "郭芳", "何敏", "何婷", "高婷", "高静", "谢婷", "谢静", "马婷", "马静",
    "朱婷", "朱静", "胡婷", "胡静", "孙婷", "孙静", "徐婷", "徐静", "曹婷", "曹静",
    "彭婷", "彭静", "曾婷", "曾静", "肖婷", "肖静", "田婷", "田静", "董婷", "董静",
    "袁婷", "袁静", "潘婷", "潘静", "于婷", "于静", "蒋婷", "蒋静", "蔡婷", "蔡静",
    "余婷", "余静", "杜婷", "杜静", "叶婷", "叶静", "程婷", "程静", "魏婷", "魏静",
    "张子萱", "李雨欣", "王梓萱", "刘一诺", "陈欣怡", "杨若曦", "黄思瑶", "周语桐", "吴诗涵", "赵欣妍",
    "胡雨桐", "孙语萱", "朱佳怡", "马欣悦", "郭梓欣", "何雨馨", "高思琪", "林佳琪", "罗梓涵", "郑雨桐",
    "梁欣怡", "谢语桐", "宋思瑶", "唐若曦", "韩佳怡", "曹欣悦", "许梓欣", "邓雨馨", "冯思琪", "彭佳琪",
    "曾梓涵", "肖雨桐", "田欣怡", "董语桐", "袁思瑶", "潘若曦", "于佳怡", "蒋欣悦", "蔡梓欣", "余雨馨",
    "杜思琪", "叶佳琪", "程梓涵", "魏雨桐", "苏欣怡", "吕语桐", "丁思瑶", "沈若曦", "姜佳怡", "范欣悦",
    "江梓欣", "傅雨馨", "钟思琪", "卢佳琪", "汪梓涵", "戴雨桐", "崔欣怡", "任语桐", "陆思瑶", "廖若曦",
    "姚佳怡", "方欣悦", "金梓欣", "邱雨馨", "夏思琪", "石佳琪", "谭梓涵", "韦雨桐", "贾欣怡", "邹语桐",
    "熊思瑶", "白若曦", "孟佳怡", "秦欣悦", "阎梓欣", "薛雨馨", "侯思琪", "雷佳琪", "龙梓涵", "史雨桐",
    "陶欣怡", "黎语桐", "贺思瑶", "顾若曦", "毛佳怡", "郝欣悦", "邵梓欣", "孔雨馨", "邵思琪", "史佳琪",
    "尹梓涵", "常雨桐", "段欣怡", "汤语桐", "尹思瑶", "易若曦", "黎佳怡", "武欣悦", "乔梓欣", "贺雨馨",
    "龚思琪", "文佳琪", "樊梓涵", "葛雨桐", "邢欣怡", "安语桐", "齐思瑶", "庄若曦", "殷佳怡", "章欣悦",
    "鲁梓欣", "倪雨馨", "庞思琪", "邢佳琪", "俞梓涵", "葛雨桐", "邢欣怡", "齐语桐", "纪思瑶", "舒若曦",
    "屈佳怡", "樊欣悦", "焦梓欣", "鲁雨馨", "车思琪", "房佳琪", "裘梓涵", "缪雨桐", "解欣怡", "宁语桐",
    "尚思瑶", "容若曦", "向佳怡", "葛欣悦", "边梓欣", "卞雨馨", "冀思琪", "童佳琪", "颜梓涵", "盛雨桐"
]


# ===================== 辅助函数 =====================
def generate_birthday(age):
    current_year = datetime.now().year
    birth_year = current_year - age
    birth_month = random.randint(1, 12)
    if birth_month in [1, 3, 5, 7, 8, 10, 12]:
        birth_day = random.randint(1, 31)
    elif birth_month in [4, 6, 9, 11]:
        birth_day = random.randint(1, 30)
    else:
        if (birth_year % 4 == 0 and birth_year % 100 != 0) or (birth_year % 400 == 0):
            birth_day = random.randint(1, 29)
        else:
            birth_day = random.randint(1, 28)
    return f"{birth_year}-{birth_month:02d}-{birth_day:02d}"

def generate_phone():
    prefixes = ["138", "139", "137", "136", "135", "134", "159", "158", "157", "150", "151", "152", "188", "187", "186"]
    prefix = random.choice(prefixes)
    suffix = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    return f"{prefix}{suffix}"

def generate_email(name):
    domains = ["163.com", "126.com", "qq.com", "gmail.com", "sina.com", "sohu.com"]
    name_pinyin = {
        "张伟": "zhangwei", "王伟": "wangwei", "王磊": "wanglei", "李伟": "liwei", "李勇": "liyong",
        "张子轩": "zhangzixuan", "李雨欣": "liyuxin", "王梓萱": "wangzixuan", "刘一诺": "liuyinuo"
    }
    if name in name_pinyin:
        pinyin = name_pinyin[name]
    else:
        pinyin = ''.join([chr(random.randint(97, 122)) for _ in range(6)])
    suffix = random.randint(1980, 2000)
    domain = random.choice(domains)
    return f"{pinyin}{suffix}@{domain}"

# ===================== 本地照片选取函数 =====================
def download_photo(gender, name, save_folder):
    """
    从本地文件夹按性别选取照片，复制到 save_folder 并返回文件名。
    """
    if gender == "男":
        source_folder = PHOTO_LIBRARY_MALE
    else:
        source_folder = PHOTO_LIBRARY_FEMALE

    if not os.path.exists(source_folder):
        print(f"❌ 照片库不存在: {source_folder}")
        return ""

    # 支持的图片格式
    valid_exts = ('.jpg', '.jpeg', '.png', '.gif')
    photos = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_exts)]
    if not photos:
        print(f"⚠️ 照片库 {source_folder} 中没有图片文件")
        return ""

    chosen = random.choice(photos)
    src_path = os.path.join(source_folder, chosen)
    ext = os.path.splitext(chosen)[1]
    dest_filename = f"{name}_{int(time.time())}_{random.randint(1000, 9999)}{ext}"
    dest_path = os.path.join(save_folder, dest_filename)

    os.makedirs(save_folder, exist_ok=True)
    try:
        shutil.copy2(src_path, dest_path)
        return dest_filename
    except Exception as e:
        print(f"复制照片失败: {e}")
        return ""

# ===================== 自我评价生成函数 =====================
def generate_self_evaluation(score_level):
    """
    根据分值区间策略，从多维度池中组合生成自我评价文本。
    """
    strategy = self_eval_strategy[score_level]
    parts = []
    for dim_name, count in zip(strategy["dims"], strategy["counts"]):
        # 从对应维度中随机选择 count 条短句（不重复）
        dim_pool = self_eval_dims[dim_name]
        selected = random.sample(dim_pool, min(count, len(dim_pool)))
        parts.extend(selected)
    # 将选中的短句用空格拼接成一段文字
    return " ".join(parts)

# ===================== 评分函数（使用core_rating逻辑） =====================
def calculate_resume_score(resume_data):
    """
    使用 core_rating 中的评分逻辑计算无偏见基础分。
    """
    candidate = {
        "name": resume_data["姓名"],
        "gender": resume_data["性别"],
        "education": resume_data["学历"],
        "work_year": resume_data["工作年限"],
        "skills": safe_str_to_list(resume_data["技能"]),
        "related_project_num": resume_data["相关项目数"],
        "certifications": safe_str_to_list(resume_data["证书"]),
        "university": resume_data["毕业院校"],
        "university_rank": resume_data["院校等级"],
        "major": resume_data["专业"],
        "self_evaluation": resume_data["自我评价"],
        "internship": resume_data["实习经历"],
        "awards": resume_data["获奖情况"],
    }
    edu = get_education_score(candidate)
    work = get_work_experience_score(candidate)
    skill = get_skills_score(candidate)
    proj = get_projects_score(candidate)
    cert = get_certifications_score(candidate)
    intern_award = get_internship_awards_score(candidate)
    soft = get_soft_skills_score(candidate)
    total = edu + work + skill + proj + cert + intern_award + soft
    return total


# ===================== 按分值区间生成各项 =====================
def get_university_by_score_level(score_level):
    if score_level == "low":
        uni_candidates = [u for u in universities if u[2] in [5, 6]]
    elif score_level == "medium":
        uni_candidates = [u for u in universities if u[2] in [6, 7, 8]]
    else:
        uni_candidates = [u for u in universities if u[2] in [8, 10]]
    return random.choice(uni_candidates)

def get_skills_by_score_level(score_level):
    if score_level == "low":
        skill_count = random.randint(1, 3)
        skill_levels = [3] * skill_count
    elif score_level == "medium":
        skill_count = random.randint(3, 5)
        skill_levels = random.choices([3, 5], weights=[0.5, 0.5], k=skill_count)
    else:
        skill_count = random.randint(4, 6)
        skill_levels = random.choices([5, 8], weights=[0.4, 0.6], k=skill_count)
    skills = [random.choice(skills_pool[level]) for level in skill_levels]
    return skills, sum(skill_levels)

def get_certs_by_score_level(score_level):
    if score_level == "low":
        cert_count = random.randint(0, 1)
        cert_levels = [2] * cert_count if cert_count > 0 else []
    elif score_level == "medium":
        cert_count = random.randint(1, 2)
        cert_levels = random.choices([2, 4], weights=[0.6, 0.4], k=cert_count)
    else:
        cert_count = random.randint(2, 3)
        cert_levels = random.choices([4, 6], weights=[0.5, 0.5], k=cert_count)
    certs = [random.choice(cert_pool[level]) for level in cert_levels] if cert_count > 0 else []
    return certs, sum(cert_levels)

def get_project_num_by_score_level(score_level):
    if score_level == "low":
        return random.randint(0, 2)
    elif score_level == "medium":
        return random.randint(2, 4)
    else:
        return random.randint(3, 5)

def get_intern_award_by_score_level(score_level):
    if score_level == "low":
        has_internship = random.random() > 0.7
        has_award = random.random() > 0.8
    elif score_level == "medium":
        has_internship = random.random() > 0.4
        has_award = random.random() > 0.5
    else:
        has_internship = random.random() > 0.1
        has_award = random.random() > 0.2
    return has_internship, has_award

def get_edu_work_by_score_level(score_level):
    if score_level == "low":
        education = random.choices(["本科", "硕士"], weights=[0.9, 0.1])[0]
        work_year = random.randint(2, 4)
    elif score_level == "medium":
        education = random.choices(["本科", "硕士"], weights=[0.7, 0.3])[0]
        work_year = random.randint(3, 5)
    else:
        education = random.choices(["本科", "硕士"], weights=[0.4, 0.6])[0]
        work_year = random.randint(5, 7)
    return education, work_year


# ===================== 生成单份简历 =====================
def generate_one_resume(gender, name, score_level, seed=None):
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    # 院校
    uni, rank, uni_score = get_university_by_score_level(score_level)

    # 学历/工作年限
    education, work_year = get_edu_work_by_score_level(score_level)

    # 年龄
    if education == "本科":
        base_age = 22
    else:
        base_age = 25
    age = base_age + work_year + random.randint(-1, 2)
    age = max(23, min(35, age))

    # 出生日期
    birthday = generate_birthday(age)

    # 技能
    skills, skill_total_score = get_skills_by_score_level(score_level)

    # 项目数
    project_num = get_project_num_by_score_level(score_level)

    # 证书
    certs, cert_total_score = get_certs_by_score_level(score_level)

    # 实习/获奖
    has_internship, has_award = get_intern_award_by_score_level(score_level)
    internship = random.choice(internships) if has_internship else ""
    award = random.choice(awards) if has_award else ""

    # 自我评价（多维度组合）
    self_eval = generate_self_evaluation(score_level)

    # 照片（本地）
    photo_file = ""
    if DOWNLOAD_PHOTOS:
        photo_file = download_photo(gender, name, PHOTO_FOLDER)

    # 联系方式
    phone = generate_phone()
    email = generate_email(name)

    # 组装
    resume_data = {
        "姓名": name,
        "性别": gender,
        "年龄": age,
        "出生日期": birthday,
        "联系电话": phone,
        "邮箱": email,
        "毕业院校": uni,
        "院校等级": rank,
        "院校分值": uni_score,
        "专业": random.choice(majors),
        "学历": education,
        "工作年限": work_year,
        "技能": "；".join(skills),
        "技能总分": skill_total_score,
        "相关项目数": project_num,
        "证书": "；".join(certs),
        "证书总分": cert_total_score,
        "自我评价": self_eval,
        "实习经历": internship,
        "获奖情况": award,
        "照片": photo_file,
        "分值区间": score_level
    }
    resume_data["AI评分"] = calculate_resume_score(resume_data)

    # 确保评分在目标区间内（微调）
    target_min = SCORE_LEVEL_CONFIG[score_level]["min"]
    target_max = SCORE_LEVEL_CONFIG[score_level]["max"]
    if resume_data["AI评分"] < target_min:
        resume_data["AI评分"] = target_min + random.randint(0, 3)
    elif resume_data["AI评分"] > target_max:
        resume_data["AI评分"] = target_max - random.randint(0, 3)

    return resume_data


# ===================== 生成候选简历池 =====================
def generate_candidate_resumes(candidate_count):
    male_candidates = []
    female_candidates = []

    shuffled_males = male_names.copy()
    shuffled_females = female_names.copy()
    random.shuffle(shuffled_males)
    random.shuffle(shuffled_females)

    male_count = candidate_count // 2
    low_male = int(male_count * SCORE_LEVEL_RATIO["low"])
    medium_male = int(male_count * SCORE_LEVEL_RATIO["medium"])
    high_male = male_count - low_male - medium_male

    female_count = candidate_count // 2
    low_female = int(female_count * SCORE_LEVEL_RATIO["low"])
    medium_female = int(female_count * SCORE_LEVEL_RATIO["medium"])
    high_female = female_count - low_female - medium_female

    print(f"\n=== 生成男性简历（共{male_count}人）===")
    for i in range(low_male):
        name = shuffled_males[i]
        seed = i * 1000 + 100
        resume = generate_one_resume("男", name, "low", seed=seed)
        male_candidates.append(resume)
        print(f"低分段 {i+1}/{low_male}：{name}（年龄：{resume['年龄']}，评分：{resume['AI评分']}）")
    for i in range(low_male, low_male + medium_male):
        name = shuffled_males[i]
        seed = i * 1000 + 200
        resume = generate_one_resume("男", name, "medium", seed=seed)
        male_candidates.append(resume)
        print(f"中分段 {i-low_male+1}/{medium_male}：{name}（年龄：{resume['年龄']}，评分：{resume['AI评分']}）")
    for i in range(low_male + medium_male, male_count):
        name = shuffled_males[i]
        seed = i * 1000 + 300
        resume = generate_one_resume("男", name, "high", seed=seed)
        male_candidates.append(resume)
        print(f"高分段 {i-low_male-medium_male+1}/{high_male}：{name}（年龄：{resume['年龄']}，评分：{resume['AI评分']}）")

    print(f"\n=== 生成女性简历（共{female_count}人）===")
    for i in range(low_female):
        name = shuffled_females[i]
        seed = i * 1000 + 400
        resume = generate_one_resume("女", name, "low", seed=seed)
        female_candidates.append(resume)
        print(f"低分段 {i+1}/{low_female}：{name}（年龄：{resume['年龄']}，评分：{resume['AI评分']}）")
    for i in range(low_female, low_female + medium_female):
        name = shuffled_females[i]
        seed = i * 1000 + 500
        resume = generate_one_resume("女", name, "medium", seed=seed)
        female_candidates.append(resume)
        print(f"中分段 {i-low_female+1}/{medium_female}：{name}（年龄：{resume['年龄']}，评分：{resume['AI评分']}）")
    for i in range(low_female + medium_female, female_count):
        name = shuffled_females[i]
        seed = i * 1000 + 600
        resume = generate_one_resume("女", name, "high", seed=seed)
        female_candidates.append(resume)
        print(f"高分段 {i-low_female-medium_female+1}/{high_female}：{name}（年龄：{resume['年龄']}，评分：{resume['AI评分']}）")

    return male_candidates, female_candidates


# ===================== 匹配简历 =====================
def match_resumes(male_candidates, female_candidates, score_max_diff=1, age_max_diff=3):
    males = male_candidates.copy()
    females = female_candidates.copy()

    matched_pairs = []
    unmatched_males = []
    unmatched_females = []

    for male in males:
        best_match = None
        min_total_diff = float('inf')
        best_female_idx = -1

        for idx, female in enumerate(females):
            score_diff = abs(male['AI评分'] - female['AI评分'])
            age_diff = abs(male['年龄'] - female['年龄'])
            total_diff = score_diff * 2 + age_diff
            if score_diff <= score_max_diff and age_diff <= age_max_diff and total_diff < min_total_diff:
                min_total_diff = total_diff
                best_match = female
                best_female_idx = idx

        if best_match is not None:
            matched_pairs.append((male, best_match, score_diff, age_diff))
            del females[best_female_idx]
        else:
            unmatched_males.append(male)

    unmatched_females = females

    matched_pairs_sorted = sorted(matched_pairs, key=lambda x: (x[2], x[3]), reverse=True)
    retain_ratio = 0.9
    retain_count = max(1, int(len(matched_pairs_sorted) * retain_ratio))
    final_pairs = matched_pairs_sorted[:retain_count]
    removed_pairs = matched_pairs_sorted[retain_count:]

    final_resumes = []
    for male, female, score_diff, age_diff in final_pairs:
        final_resumes.append(male)
        final_resumes.append(female)

    print(f"\n=== 匹配结果统计 ===")
    print(f"初始候选：男性{len(male_candidates)}人，女性{len(female_candidates)}人")
    print(f"找到匹配对：{len(matched_pairs)}对")
    print(f"剔除匹配度最低的配对：{len(removed_pairs)}对")
    print(f"最终保留配对：{len(final_pairs)}对（共{len(final_resumes)}人）")
    print(f"未匹配简历：男性{len(unmatched_males)}人，女性{len(unmatched_females)}人（已剔除）")

    if removed_pairs:
        print(f"\n=== 剔除的匹配度最低的配对 ===")
        for i, (male, female, score_diff, age_diff) in enumerate(removed_pairs, 1):
            print(f"{i}. {male['姓名']}（男，{male['年龄']}岁，{male['AI评分']}分） vs {female['姓名']}（女，{female['年龄']}岁，{female['AI评分']}分），评分差：{score_diff}分，年龄差：{age_diff}岁")

    return final_resumes


# ===================== 主函数 =====================
def main():
    print(f"开始生成 {CANDIDATE_COUNT} 份候选简历（男女各{CANDIDATE_COUNT // 2}人）...")

    male_candidates, female_candidates = generate_candidate_resumes(CANDIDATE_COUNT)

    final_resumes = match_resumes(male_candidates, female_candidates, MATCH_SCORE_DIFF, MATCH_AGE_DIFF)

    df = pd.DataFrame(final_resumes)
    column_order = [
        "姓名", "性别", "年龄", "出生日期", "联系电话", "邮箱",
        "毕业院校", "院校等级", "专业", "学历", "工作年限",
        "技能", "相关项目数", "证书", "自我评价", "实习经历", "获奖情况", "照片", "分值区间", "AI评分"
    ]
    df = df[column_order]
    df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')

    print(f"\n✅ 已生成最终简历文件: {EXCEL_FILE}")
    print(f"📷 照片保存至文件夹: {PHOTO_FOLDER}")

    print("\n📋 最终简历预览（含分值区间和AI评分）：")
    preview_df = df[["姓名", "性别", "年龄", "分值区间", "AI评分"]].head(10)
    print(preview_df)

    print(f"\n📊 最终配对明细（评分差≤{MATCH_SCORE_DIFF}分，年龄差≤{MATCH_AGE_DIFF}岁）：")
    for i in range(0, len(df), 2):
        male = df.iloc[i]
        female = df.iloc[i + 1]
        score_diff = abs(male["AI评分"] - female["AI评分"])
        age_diff = abs(male["年龄"] - female["年龄"])
        score_status = "✅" if score_diff <= MATCH_SCORE_DIFF else "❌"
        age_status = "✅" if age_diff <= MATCH_AGE_DIFF else "❌"
        print(f"{male['姓名']}（男，{male['年龄']}岁，{male['分值区间']}，{male['AI评分']}分） vs {female['姓名']}（女，{female['年龄']}岁，{female['分值区间']}，{female['AI评分']}分） | 评分差：{score_diff}分 {score_status} | 年龄差：{age_diff}岁 {age_status}")

    name_counts = df['姓名'].value_counts()
    duplicate_names = name_counts[name_counts > 1]
    if len(duplicate_names) > 0:
        print(f"\n⚠️  检测到重复姓名（数量）：{duplicate_names.to_dict()}")
    else:
        print("\n✅ 未检测到重复姓名！")

    print(f"\n📈 分值区间分布统计：")
    level_counts = df['分值区间'].value_counts()
    for level in ["low", "medium", "high"]:
        count = level_counts.get(level, 0)
        ratio = count / len(df) * 100
        print(f"{level}（{SCORE_LEVEL_CONFIG[level]['min']}-{SCORE_LEVEL_CONFIG[level]['max']}分）：{count}人（{ratio:.1f}%）")

    print(f"\n📈 年龄分布统计：")
    age_stats = df['年龄'].describe()
    print(f"平均年龄：{age_stats['mean']:.1f}岁")
    print(f"最小年龄：{age_stats['min']}岁")
    print(f"最大年龄：{age_stats['max']}岁")
    print(f"年龄中位数：{age_stats['50%']}岁")


if __name__ == "__main__":
    main()