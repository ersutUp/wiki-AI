"""从维基百科加载数据，支持 JSON 持久化"""
import json
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document

# JSON 文件默认路径
DEFAULT_JSON_PATH = Path(__file__).parent / "wikipedia_docs.json"


def fetch_wikipedia(query: str, lang: str = "zh", load_max_docs: int = 5) -> List[Document]:
    """从维基百科实时获取文档"""
    return WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs).load()


def save_to_json(docs: List[Document], file_path: str | Path = DEFAULT_JSON_PATH) -> Path:
    """将文档列表保存为 JSON 文件"""
    file_path = Path(file_path)
    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {len(docs)} 篇文档到 {file_path}")
    return file_path


def load_from_json(file_path: str | Path = DEFAULT_JSON_PATH) -> List[Document]:
    """从 JSON 文件读取文档列表"""
    file_path = Path(file_path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    docs = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in data]
    print(f"已从 {file_path} 读取 {len(docs)} 篇文档")
    return docs

def load():
    # ===== 模块级：首次导入时自动加载（兼容写入图数据库.py 的 import） =====
    if DEFAULT_JSON_PATH.exists():
        loaded_docs = load_from_json()
    else:
        # 国内网络环境需要代理才能访问维基百科 API
        # NOTE: 必须在 import langchain_community 之前设置，否则 requests 库读不到
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

        loaded_docs = fetch_wikipedia(query="马云", lang="zh", load_max_docs=5)
        save_to_json(loaded_docs)
        print(f"共加载 {len(loaded_docs)} 篇文档：")
    return loaded_docs


if __name__ == "__main__":
    loaded_docs = load()
    for doc in loaded_docs:
        print(f"  标题: {doc.metadata.get('title')}")
        print(f"  摘要: {doc.metadata.get('summary', '')[:80]}...")
        print(f"  来源: {doc.metadata.get('source')}")
        print(f"  内容长度: {len(doc.page_content)} 字符")
        print()