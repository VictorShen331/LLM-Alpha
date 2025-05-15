import datetime
import os
DEEPSEEK_API_KEY = ''
DS_BASE_URL = "https://api.deepseek.com"
GIT_URL = "https://github.com/VictorShen331/LLM-Alpha/blob/main/"
DATA_PATH = "E:/intern/llm行业基本面量化课题数据"
FUNDAMENTAL_FILE = "处理后的基础数据.parquet.gzip"
INDUSTRY_REPORT_FILE = "ReportIndustry.csv"
DEFINITION_FILE = "行业商业模式分类.xlsx"
OPERATOR_FILE = "operator_list.xlsx"
INDUSTRY_L1_CODE = '801120'
# INDUSTRY_L1_NAME = "食品饮料"
# 申万一级“消费”行业代码示例
CONSUMER_I1_CODES = ["801120", "801130", "801140"]
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
TRAINED_MODEL_DIR = DATA_PATH + f"/{now}"
os.makedirs(TRAINED_MODEL_DIR)