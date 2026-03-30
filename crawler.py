from pathlib import Path
import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

from utils import load_config

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+(?=[，。！？；：、）】》”])", "", text)
    text = re.sub(r"(?<=[（【《“])\s+", "", text)
    text = re.sub(r"\s*/\s*", "/", text)

    useless_phrases = [
        "（如下图）",
        "(如下图)",
        "如下图",
        "如图所示",
        "见下图",
    ]
    for phrase in useless_phrases:
        text = text.replace(phrase, "")

    return text.strip()


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "_", name).strip("_")
    return name[:80] if name else "untitled"

def get_url_suffix(url: str) -> str:

    path = urlparse(url).path.rstrip("/")

    suffix = path.split("/")[-1]

    suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", suffix).strip("_")
    return suffix if suffix else "page"

def text_density_score(node) -> float:

    text = clean_text(node.get_text(" ", strip=True))
    if not text:
        return -1

    text_len = len(text)
    p_count = len(node.find_all("p"))
    li_count = len(node.find_all("li"))
    a_count = len(node.find_all("a"))
    punct_count = sum(text.count(ch) for ch in "。！？；：，")

    score = (
        text_len
        + p_count * 80
        + punct_count * 20
        - a_count * 8
        - li_count * 5
    )

    return score


def find_main_content_node(soup):

    candidate_selectors = [
        "article",
        "main",
        ".article",
        ".content",
        ".article-content",
        ".post-content",
        ".help-content",
        ".detail",
        ".detail-content",
        ".problem-content",
        ".solution",
        "[class*='content']",
        "[class*='article']",
        "[class*='detail']",
        "[class*='help']",
        "[id*='content']",
        "[id*='article']",
        "[id*='detail']",
    ]

    best_node = None
    best_score = -1

    for selector in candidate_selectors:
        for node in soup.select(selector):
            score = text_density_score(node)
            if score > best_score:
                best_score = score
                best_node = node

    if best_node is None:
        for node in soup.find_all(["div", "section", "article", "main"]):
            score = text_density_score(node)
            if score > best_score:
                best_score = score
                best_node = node

    return best_node

def extract_main_text(html: str, url: str) -> tuple[str, str]:

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg", "img", "footer", "nav"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)

    main_node = find_main_content_node(soup)
    if main_node is None:
        raise ValueError("未找到可用的正文区域")

    blocks = []
    seen = set()


    for tag in main_node.find_all(["h1", "h2", "h3", "p"]):

        text = clean_text(tag.get_text(" ", strip=True))

        if len(text) < 12:
            continue


        if text in seen:
            continue
        seen.add(text)


        if "这条帮助是否解决了您的问题" in text:
            continue
        if "猜你感兴趣的问题" in text:
            continue
        if "人参与投票" in text:
            continue

        if tag.name == "h1":
            blocks.append(f"# {text}")
        elif tag.name == "h2":
            blocks.append(f"## {text}")
        elif tag.name == "h3":
            blocks.append(f"### {text}")
        else:
            blocks.append(text)

    if not blocks:
        raise ValueError("正文区域存在，但没有提取到有效文本")
    # content = [
    #     f"# 页面标题\n{title or '未提取到标题'}",
    #     f"\n# 页面来源\n{url}",
    #     "\n# 正文\n",
    #     "\n\n".join(blocks)
    # ]
    # return title or url, "\n".join(content)
    content = "\n\n".join(blocks)
    return title or url, content


def crawl_one(url: str, headers: dict, timeout: int, save_dir: Path) -> None:
    print(f"[抓取] {url}")

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    title, content = extract_main_text(resp.text, url)

    file_path = save_dir / f"{safe_filename(title)}_{get_url_suffix(url)}.md"
    file_path.write_text(content, encoding="utf-8")
    print(f"[保存] {file_path}")


def main():

    cfg = load_config()

    headers = cfg["crawler"]["headers"]
    timeout = cfg["crawler"]["timeout"]
    save_dir = Path(cfg["crawler"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    seed_urls = [u for u in cfg["seed_urls"] if u.strip()]

    if not seed_urls:
        raise ValueError("config/config.yaml 里的 seed_urls 还没有填写网址。")

    success, failed = 0, 0

    for url in seed_urls:
        try:
            crawl_one(url, headers, timeout, save_dir)
            success += 1
        except Exception as e:
            failed += 1
            print(f"[失败] {url} -> {e}")

    print(f"\n完成：成功 {success}，失败 {failed}")


if __name__ == "__main__":
    main()