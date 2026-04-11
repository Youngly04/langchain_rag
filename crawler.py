from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup

from utils import load_config
def strip_md_prefix(text: str) -> str:
    """去掉 markdown 标题前缀"""
    return re.sub(r"^#{1,6}\s+", "", text).strip()


def is_heading_block(text: str) -> bool:
    raw = strip_md_prefix(text)

    if not raw:
        return False

    # markdown 标题
    if re.match(r"^#{1,6}\s+", text):
        return True

    # 中文章节标题：一、服务介绍 / 二、服务流程
    if re.fullmatch(r"[一二三四五六七八九十]+、.{1,15}", raw):
        return True

    # 阿拉伯数字章节标题：1、下单流程 / 2、服务时间
    if re.fullmatch(r"\d+、.{1,15}", raw):
        return True

    return False


def is_short_label_block(text: str) -> bool:
    """
    判断是不是“只有标签、没有正文”的短条目
    例如：
    ①退换货运费收取规则：
    电脑端：
    手机端：
    """
    raw = strip_md_prefix(text)

    if not raw:
        return False

    # 短标签：以冒号结尾，整体较短，且不像完整句子
    if raw.endswith(("：", ":")) and len(raw) <= 25:
        return True

    return False


def next_nonempty_block(blocks: list[str], start: int):
    """找后面第一个非空块"""
    for j in range(start + 1, len(blocks)):
        if blocks[j].strip():
            return blocks[j].strip()
    return None


def prune_blocks(blocks: list[str]) -> list[str]:
    """
    删除：
    1. 空标题：后面没有正文，或后面直接又是另一个标题
    2. 空条目：只有“电脑端：/①xxx：”这种标签，但后面没有正文支撑
    """
    pruned = []

    for i, block in enumerate(blocks):
        cur = block.strip()
        if not cur:
            continue

        nxt = next_nonempty_block(blocks, i)

        # 1) 删除空标题
        if is_heading_block(cur):
            if nxt is None or is_heading_block(nxt):
                continue

        # 2) 删除空条目
        if is_short_label_block(cur):
            if nxt is None or is_heading_block(nxt) or is_short_label_block(nxt):
                continue

        pruned.append(cur)

    # 再压一次连续重复空白
    text = "\n\n".join(pruned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return [x for x in text.split("\n\n") if x.strip()]
def clean_text(text: str) -> str:
    # 1. 统一特殊空白字符
    text = re.sub(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]+", " ", text)

    # 2. 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. 项目符号转为换行，保留列表结构
    text = re.sub(r"[ \t]*[·•●▪▶][ \t]*", "\n", text)

    # 4. 去掉每行首尾空白
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 5. 压缩非换行空白
    text = re.sub(r"[^\S\n]+", " ", text)

    # 6. 去掉中文之间的空格
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)

    # 7. 去掉中文与数字之间的空格
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=\d)", "", text)
    text = re.sub(r"(?<=\d) (?=[\u4e00-\u9fff])", "", text)

    # 8. 去掉中文与中文标点之间的空格
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[，。！？；：、）】》”])", "", text)
    text = re.sub(r"(?<=[，。！？；：、（【《“]) (?=[\u4e00-\u9fff])", "", text)

    # 9. 去掉括号内侧多余空格
    text = re.sub(r"(?<=[（【《“]) +", "", text)
    text = re.sub(r" +(?=[）】》”])", "", text)

    # 10. 统一连接符周围空格
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*:\s*", ":", text)

    # 11. 统一 APP 写法
    text = re.sub(r"京东app", "京东APP", text, flags=re.IGNORECASE)

    # 12. 多个横线压成一个
    text = re.sub(r"-{2,}", "-", text)

    # 13. 修正常见路径引号混乱
    text = text.replace('“我的"-“我的订单”-"检测维修服务订单”-“联系客服”',
                        '“我的”-“我的订单”-“检测维修服务订单”-“联系客服”')
    text = text.replace('“我的"-“我的订单”', '“我的”-“我的订单”')
    text = text.replace('”-"', '”-“')

    # 14. 清理无意义短语
    useless_phrases = [
        "（如下图）",
        "(如下图)",
        "如下图",
        "如图所示",
        "见下图",
    ]
    for phrase in useless_phrases:
        text = text.replace(phrase, "")

    # 15. 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def text_density_score(node) -> float:

    text = clean_text(node.get_text("", strip=True))
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


    for tag in main_node.find_all(["h1", "h2", "h3", "p", "li"]):

        text = clean_text(tag.get_text("", strip=True))

        if text.endswith("？") or text.endswith("?"):
            if len(text) <= 25:
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
        if "help.jd.com" in text:
            continue
        # 过滤空标题/无意义占位行
        if text in {
            "服务标识查看方式如下："
        }:
            continue

        # 过滤纯日期行，如 2025年9月24日
        if re.fullmatch(r"\d{4}年\d{1,2}月\d{1,2}日", text):
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
    blocks = prune_blocks(blocks)
    content = "\n\n".join(blocks)
    content = clean_text(content)
    return title or url, content


def crawl_one(name: str, url: str, headers: dict, timeout: int, save_dir: Path) -> None:
    print(f"[抓取] {url}")

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    title, content = extract_main_text(resp.text, url)

    file_path = save_dir / f"{name}.md"
    file_path.write_text(content, encoding="utf-8")
    print(f"[保存] {file_path}")


def main():

    cfg = load_config()

    headers = cfg["crawler"]["headers"]
    timeout = cfg["crawler"]["timeout"]
    save_dir = Path(cfg["crawler"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    seed_pages = cfg["seed_pages"]

    if not seed_pages:
        raise ValueError("config/config.yaml 里的 seed_pages 还没有填写内容。")

    success, failed = 0, 0

    for item in seed_pages:
        try:
            crawl_one(item["name"], item["url"], headers, timeout, save_dir)
            success += 1
        except Exception as e:
            failed += 1
            print(f"[失败] {item['url']} -> {e}")

    print(f"\n完成：成功 {success}，失败 {failed}")


if __name__ == "__main__":
    main()