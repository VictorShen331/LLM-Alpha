
import os
DEEPSEEK_API_KEY = ''
DS_BASE_URL = "https://api.deepseek.com"
GIT_URL = "https://github.com/VictorShen331/LLM-Alpha/blob/main/"
DATA_PATH = "E:/intern/llm行业基本面量化课题数据"
FUNDAMENTAL_FILE = "处理后的基础数据.parquet.gzip"
INDUSTRY_REPORT_FILE = "ReportIndustry.csv"
DEFINITION_FILE = "行业商业模式分类.xlsx"
OPERATOR_FILE = "operator_list.xlsx"
INDUSTRY_L1_CODE_LIST = ['801040', '801120' ,'801200' ,'801140']
INDUSTRY_L1_NAME_LIST = ["钢铁" ,"食品饮料" ,"商贸零售" ,"轻工制造"]
INDUSTRY_IDX = 3
# INDUSTRY_L1_CODE = '801040' #'801120' #'801200' #'801140'
# INDUSTRY_L1_NAME = "钢铁" #"食品饮料" #"商贸零售" #"轻工制造"
INDUSTRY_L1_CODE = INDUSTRY_L1_CODE_LIST[INDUSTRY_IDX]
INDUSTRY_L1_NAME = INDUSTRY_L1_NAME_LIST[INDUSTRY_IDX]

FUNDAMENTAL_PROCESSED_PATH = DATA_PATH + '/fundamental_data'
INDUSTRY_PATH = DATA_PATH + '/combined' + f'/{INDUSTRY_L1_CODE}'
if not os.path.exists(FUNDAMENTAL_PROCESSED_PATH):
    os.makedirs(FUNDAMENTAL_PROCESSED_PATH)
if not os.path.exists(INDUSTRY_PATH):
    os.makedirs(INDUSTRY_PATH)