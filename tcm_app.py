import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 配置本地模型 — 使用中文嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
Settings.llm = Ollama(model="qwen2.5:7b-instruct-q5_K_M", request_timeout=300)

st.set_page_config(page_title="倪海厦中医经方助手", page_icon="🌿", layout="wide")

@st.cache_resource
def load_knowledge():
    storage_context = StorageContext.from_defaults(persist_dir="./nihaixia_index")
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"
    )

query_engine = load_knowledge()

SYSTEM_PROMPT = """
你现在完全按照倪海厦医师的经典中医经方体系进行辨证：
1. 先辨六经：太阳→阳明→少阳 → 太阴→少阴→厥阴，理清传变关系
2. 对照《伤寒论》《金匮要略》条文，匹配对应汤证
3. 明确给出方剂名称、药物组成、剂量原则、煎服法与禁忌
4. 参考倪氏健康六大标准：睡眠、饮食、二便、手足温、体力等
5. 引用原文出处，说明辨证思路

⚠️ 免责声明：本系统仅供学习交流使用，不构成医疗建议。身体不适请前往正规医疗机构就诊。
"""

st.title("🌿 倪海厦 — 经方AI辨证助手")
st.subheader("六经辨证 · 随证处方")

col1, col2 = st.columns(2)
with col1:
    chills = st.selectbox("恶寒发热", ["无明显异常", "恶寒重发热轻", "发热重恶寒轻", "寒热往来"])
    sweating = st.selectbox("出汗情况", ["正常", "无汗", "自汗", "盗汗"])
    pain = st.multiselect("疼痛部位", ["颈项", "头部", "胸胁", "脘腹", "四肢关节"])
with col2:
    thirst = st.selectbox("口渴饮水", ["正常", "不渴", "喜冷饮", "喜温饮"])
    stool = st.selectbox("大便情况", ["正常", "干结难下", "稀溏", "黎明泻"])
    urine = st.selectbox("小便情况", ["正常", "清长量多", "黄赤量少", "排尿不适"])

tongue = st.text_input("舌象（舌质/舌苔）")
pulse = st.text_input("脉象")
symptoms_extra = st.text_area("其他症状与不适")

if st.button("🔍 辨证分析与推荐方剂", type="primary"):
    with st.spinner("正在运用六经体系辨证，请稍候……"):
        query = f"""
患者症状：
- 恶寒发热：{chills}
- 出汗：{sweating}
- 疼痛部位：{', '.join(pain) or '无'}
- 口渴饮水：{thirst}
- 大便：{stool}
- 小便：{urine}
- 舌象：{tongue or '未提供'}
- 脉象：{pulse or '未提供'}
- 补充：{symptoms_extra or '无'}

{SYSTEM_PROMPT}
"""
        result = query_engine.query(query)
        
        st.markdown("---")
        st.markdown("### 📋 辨证结果与处方建议")
        st.markdown(result.response)
        
        with st.expander("📖 知识来源原文"):
            for node in result.source_nodes:
                st.write(f"来源文件：{node.metadata.get('file_name', '未知')}")
                st.caption(node.text[:300] + "……")