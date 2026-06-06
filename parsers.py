"""
文档解析器 — 把各种格式的文件统一转换成纯文本
支持：TXT / Markdown / PDF / Word / Excel
"""

import logging
import os
from typing import List

logger = logging.getLogger("kb")


def parse_file(file_path: str) -> str:
    """
    根据文件扩展名自动选择解析器，返回纯文本内容
    """
    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"parse_file: {os.path.basename(file_path)} (ext={ext})")

    if ext in (".txt", ".md", ".markdown"):
        return _parse_text(file_path)
    elif ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    elif ext == ".doc":
        return _parse_doc(file_path)
    elif ext in (".xlsx", ".xls"):
        return _parse_excel(file_path)
    else:
        logger.error(f"不支持的文件格式: {ext}")
        raise ValueError(f"不支持的文件格式: {ext}")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    语义分块：按段落 → 句子 → 字符逐级切分，尊重文档结构
    1. 优先按段落（双换行）切分
    2. 超长段落按句子切分
    3. 超长句子按字符数切分（兜底）
    4. 合并相邻小段落，直到接近 chunk_size
    5. chunk_overlap 以段落/句子为粒度实现重叠
    """
    # ==================== 参数校验 ====================
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须大于 0，当前值: {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap 不能为负数，当前值: {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})，"
            "否则分块效率极低或产生死循环"
        )

    text = text.strip()
    if not text:
        return []

    # ==================== 第一步：按段落切分 ====================
    raw_paragraphs = text.split("\n\n")
    # 去除每段首尾空白，过滤空段落
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # ==================== 第二步：超长段落按句子切分 ====================
    import re
    # 句子分隔符：中文句末标点和英文句末标点
    sentence_pattern = re.compile(r'(?<=[。！？.!?])')

    def split_to_sentences(para: str) -> List[str]:
        """将段落按句子分隔符切分，保留分隔符在句尾"""
        parts = sentence_pattern.split(para)
        # 过滤空白部分
        return [s.strip() for s in parts if s.strip()]

    # 将所有段落展开为「语义单元」列表（每个单元是一段或一句）
    semantic_units: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            semantic_units.append(para)
        else:
            sentences = split_to_sentences(para)
            for sentence in sentences:
                if len(sentence) <= chunk_size:
                    semantic_units.append(sentence)
                else:
                    # ==================== 第三步：超长句子按字符切分（兜底） ====================
                    sub_start = 0
                    while sub_start < len(sentence):
                        sub_end = min(sub_start + chunk_size, len(sentence))
                        sub_chunk = sentence[sub_start:sub_end]
                        # 尝试在最近的标点处断开
                        if sub_end < len(sentence):
                            for punct in ("，", "；", "、", ",", ";", " "):
                                last_p = sub_chunk.rfind(punct, chunk_size // 2)
                                if last_p > 0:
                                    sub_chunk = sub_chunk[:last_p + 1]
                                    sub_end = sub_start + len(sub_chunk)
                                    break
                        sub_chunk = sub_chunk.strip()
                        if sub_chunk:
                            semantic_units.append(sub_chunk)
                        # 确保始终前进
                        next_sub = sub_end
                        if next_sub <= sub_start:
                            next_sub = sub_start + chunk_size
                        sub_start = next_sub

    if not semantic_units:
        return []

    # ==================== 第四步：合并相邻小单元 ====================
    chunks: List[str] = []
    current = semantic_units[0]

    for unit in semantic_units[1:]:
        # 如果合并后不超过 chunk_size，则合并
        if len(current) + len(unit) + 1 <= chunk_size:  # +1 是换行符
            current = current + "\n" + unit
        else:
            chunks.append(current)
            current = unit
    # 别忘了最后一个累积单元
    if current:
        chunks.append(current)

    # ==================== 第五步：段落/句子级别的重叠 ====================
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            # 从前一个 chunk 的尾部取约 chunk_overlap 个字符的内容作为重叠前缀
            prev = chunks[i - 1]
            # 尝试在句子/段落边界处截取重叠内容，而非硬切字符
            overlap_text = _extract_tail_overlap(prev, chunk_overlap)
            if overlap_text:
                overlapped.append(overlap_text + "\n" + chunks[i])
            else:
                overlapped.append(chunks[i])
        chunks = overlapped

    return chunks


def _extract_tail_overlap(text: str, target_len: int) -> str:
    """
    从文本尾部提取约 target_len 个字符的重叠内容
    优先在句子边界处截断，其次在段落边界处，最后按字符截断
    """
    if target_len <= 0 or len(text) <= target_len:
        return ""

    tail = text[-target_len:]

    # 尝试在句子起始处截断（找第一个句末标点之后的位置）
    import re
    cut_match = re.search(r'[。！？.!?]\s*', tail)
    if cut_match and cut_match.start() < len(tail) // 2:
        return tail[cut_match.end():]

    # 尝试在换行处截断
    newline_pos = tail.find("\n")
    if 0 < newline_pos < len(tail) // 2:
        return tail[newline_pos + 1:]

    # 兜底：直接截取
    return tail


# ==================== 以下为各格式的具体解析实现 ====================


def _parse_text(file_path: str) -> str:
    """解析 TXT / Markdown 文件（支持大文件流式读取）"""
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB，超过则报错
    encodings = ["utf-8", "gbk", "gb2312", "utf-16", "latin-1"]

    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(
            f"文件过大（{file_size / 1024 / 1024:.1f}MB），最大支持 {MAX_FILE_SIZE / 1024 / 1024:.0f}MB。"
            "请拆分文件后重试。"
        )

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法识别文件编码: {file_path}")


def _parse_pdf(file_path: str) -> str:
    """解析 PDF 文件（提取文字，忽略图片）"""
    import fitz  # PyMuPDF

    MAX_PAGES = 100  # 限制最大页数，防止内存耗尽
    MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB，限制 PDF 文件大小

    try:
        # 先检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size > MAX_PDF_SIZE:
            raise ValueError(
                f"PDF 文件过大（{file_size / 1024 / 1024:.1f}MB），最大支持 {MAX_PDF_SIZE / 1024 / 1024:.0f}MB。"
                "请拆分后重试。"
            )

        with fitz.open(file_path) as doc:
            if len(doc) > MAX_PAGES:
                raise ValueError(
                    f"PDF 页数过多（{len(doc)} 页），最多支持 {MAX_PAGES} 页。"
                    "请拆分后再上传。"
                )
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
        # with 语句自动关闭 doc，无需手动 close
    except fitz.FileDataError:
        raise ValueError(f"PDF 文件无法打开，可能已损坏或加密: {file_path}")
    except Exception as e:
        raise ValueError(f"PDF 文件解析失败: {file_path}，错误: {e}")

    return "\n".join(text_parts)


def _parse_docx(file_path: str) -> str:
    """解析 Word (.docx) 文件"""
    from docx import Document

    doc = Document(file_path)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # 也提取表格内容
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    return "\n".join(paragraphs)


def _parse_doc(file_path: str) -> str:
    """
    解析旧版 Word (.doc) 文件
    通过 Word/WPS COM 自动转换为 .docx 后解析
    """
    import tempfile

    abs_path = os.path.abspath(file_path)
    tmp_path = None

    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()

        # 尝试 Word，再尝试 WPS
        app = None
        for prog_id in ("Word.Application", "KWPS.Application"):
            try:
                app = win32com.client.Dispatch(prog_id)
                break
            except Exception:
                continue

        if app is None:
            raise RuntimeError("未找到 Word 或 WPS")

        app.Visible = False
        app.DisplayAlerts = False

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = tmp.name

        doc = app.Documents.Open(abs_path)
        try:
            doc.SaveAs2(os.path.abspath(tmp_path), FileFormat=16)  # 16 = docx
        finally:
            doc.Close(False)

        result = _parse_docx(tmp_path)
        return result

    except Exception as e:
        raise ValueError(
            f"无法解析 .doc 文件: {os.path.basename(file_path)}\n"
            f"错误: {e}\n"
            "请用 Word/WPS 另存为 .docx 格式后重新上传"
        )
    finally:
        # 确保 Word/WPS 进程被关闭
        try:
            app.Quit()
        except Exception:
            pass
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass


def _parse_excel(file_path: str) -> str:
    """解析 Excel 文件（逐行转成文本）"""
    import pandas as pd

    MAX_ROWS = 100000  # 限制最大行数，防止内存溢出

    try:
        # 先获取所有 sheet 名称
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        all_rows = []

        for sheet in sheet_names:
            try:
                # 使用 nrows 限制读取行数，防止大文件 OOM
                df = pd.read_excel(xl, sheet_name=sheet, nrows=MAX_ROWS)
            except Exception as e:
                raise ValueError(f"Excel 文件读取 Sheet [{sheet}] 失败: {file_path}，错误: {e}")

            if df.empty:
                continue

            # 逐行处理，防止 iterrows 效率问题，同时消毒公式注入
            for _, row in df.iterrows():
                parts = []
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        continue
                    val_str = str(val).strip()
                    if not val_str:
                        continue
                    # 消毒：移除以 =、+、-、@ 开头的危险公式注入字符
                    if val_str.startswith(("=", "+", "-", "@")):
                        val_str = "'" + val_str  # 加前缀防止公式执行
                    parts.append(f"{col}: {val_str}")
                if parts:
                    all_rows.append("；".join(parts))

        if not all_rows:
            return ""

        # 如果有多个 sheet，拼接时标注来源
        if len(sheet_names) > 1:
            return "\n".join(all_rows) + f"\n（共读取 {len(sheet_names)} 个 Sheet）"
        else:
            return "\n".join(all_rows)

    except FileNotFoundError:
        raise ValueError(f"Excel 文件不存在: {file_path}")
    except Exception as e:
        raise ValueError(f"Excel 文件解析失败: {file_path}，错误: {e}")
