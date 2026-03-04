#!/usr/bin/env python3
"""
EPUB Novel Summarizer using Ollama
===================================
Rút gọn bộ truyện siêu dài định dạng EPUB xuống còn ~1/10 độ dài gốc.
Sử dụng AI local Ollama để tóm tắt từng chương, giữ nguyên tên chương.

Cài đặt:
    pip install ebooklib beautifulsoup4 requests tqdm

Sử dụng:
    python epub_summarizer.py input.epub output.epub
    python epub_summarizer.py input.epub output.epub --ratio 0.1 --model llama3
    python epub_summarizer.py input.epub output.epub --start-chapter 50 --end-chapter 100
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ebooklib
import requests
from bs4 import BeautifulSoup
from ebooklib import epub
from tqdm import tqdm

# ─────────────────────────────────────────────
# Cấu hình mặc định
# ─────────────────────────────────────────────
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_RATIO = 0.20  # Rút gọn còn 20% bản gốc
OLLAMA_URL = "http://localhost:11434/api/generate"
CONTEXT_KEEP = 0  # Số chương gần nhất giữ lại làm context
MAX_CONTEXT_WORDS = 600  # Giới hạn context (words) gửi kèm mỗi lần
RETRY_LIMIT = 3  # Số lần thử lại nếu Ollama lỗi
RETRY_DELAY = 5  # Giây chờ giữa các lần retry
PROGRESS_FILE_SUFFIX = ".progress.json"  # File lưu tiến trình


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class Chapter:
    index: int
    title: str
    raw_html: str
    text: str = ""
    word_count: int = 0
    summary: str = ""
    summary_word_count: int = 0


@dataclass
class SummaryContext:
    """Lưu context tóm tắt của các chương gần nhất."""

    recent: list[dict] = field(default_factory=list)

    def add(self, title: str, summary: str):
        self.recent.append({"title": title, "summary": summary})
        if len(self.recent) > CONTEXT_KEEP:
            self.recent.pop(0)

    def build_context_text(self, max_words: int = MAX_CONTEXT_WORDS) -> str:
        if not self.recent:
            return ""
        parts = []
        for item in self.recent:
            parts.append(f"[{item['title']}]: {item['summary']}")
        context = "\n".join(parts)
        words = context.split()
        if len(words) > max_words:
            context = " ".join(words[-max_words:])
        return context


# ─────────────────────────────────────────────
# Tiện ích văn bản
# ─────────────────────────────────────────────
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def count_words(text: str) -> int:
    return len(text.split())


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


# ─────────────────────────────────────────────
# Ollama API
# ─────────────────────────────────────────────
def call_ollama(prompt: str, model: str, temperature: float = 0.3) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 2048},
    }
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            print(
                f"\n⚠️  Không kết nối được Ollama (thử {attempt}/{RETRY_LIMIT}). "
                f"Đảm bảo `ollama serve` đang chạy."
            )
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"\n⚠️  Lỗi Ollama (thử {attempt}/{RETRY_LIMIT}): {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
    raise RuntimeError("Ollama không phản hồi sau nhiều lần thử. Dừng chương trình.")


# Model context limit (words). Chapters longer than this are chunked.
# Most 7B models handle ~6000–8000 words safely in one call.
MODEL_CONTEXT_WORD_LIMIT = 7000


def build_prompt(
    chapter_title: str,
    chapter_text: str,
    original_word_count: int,
    target_words: int,
    context_text: str,
    language: str,
    chunk_info: str = "",
) -> str:
    lang_instruction = {
        "vi": "Hãy rút gọn truyện bằng tiếng Việt.",
        "en": "Please summarize in English.",
        "auto": "Summarize in the same language as the chapter text.",
    }.get(language, "Summarize in the same language as the chapter text.")

    context_block = ""
    if context_text:
        context_block = f"""CONTEXT (tóm tắt các chương trước để duy trì mạch truyện):
{context_text}

"""

    chunk_note = f"\n({chunk_info})" if chunk_info else ""

    prompt = f"""\
Bạn là một nhà văn chuyên biên tập và chuyển thể tiểu thuyết tiên hiệp. Nhiệm vụ của bạn là viết lại một chương truyện dài thành một phiên bản tinh gọn nhưng vẫn giữ được phong thái, nhịp điệu và cảm xúc của một chương truyện đầy đủ.

{context_block}CHƯƠNG CẦN XỬ LÝ: {chapter_title}{chunk_note}
Độ dài yêu cầu: Bắt buộc nằm trong khoảng {int(target_words * 4.5 * 0.7)}-{int(target_words * 4.5 * 1.3)} ký tự. (Đây là yêu cầu nghiêm ngặt, không được viết dưới {int(target_words * 4.5 * 0.7)} ký tự).

CẤU TRÚC VÀ PHONG CÁCH:
1. Văn phong: Viết dưới dạng chương truyện hoàn chỉnh, có dẫn dắt, có lời thoại và miêu tả tâm lý. Tuyệt đối không viết theo kiểu liệt kê sự kiện hay tóm tắt mục lục.
2. Giữ lại linh hồn: Phải có đủ các nhân vật, sự kiện then chốt và đặc biệt là cảm xúc của nhân vật chính trong cao trào.
3. Kỹ thuật tinh gọn:
  * Thay vì miêu tả chiêu thức 10 dòng, hãy dùng 1-2 câu súc tích nhưng đầy uy lực.
  * Lược bỏ các đoạn hội thoại sáo rỗng, chỉ giữ lại những câu thoại mang tính quyết định hoặc thể hiện thần thái nhân vật.
  * Biến các đoạn nội tâm dài dòng thành những suy nghĩ sắc bén, gãy gọn.
4. Nhịp độ: Đẩy nhanh tốc độ diễn tiến nhưng vẫn phải có những khoảng lặng cảm xúc để người đọc không cảm thấy bị "ngộp" thông tin.

QUY TẮC CẤM:
- Không thêm tình tiết mới ngoài bản gốc.
- Không ghi chú "Dưới đây là bản rút gọn...", không tiêu đề, không lời dẫn. Chỉ trả về nội dung chương truyện.
- Không sử dụng các cụm từ tóm tắt như "Tóm lại...", "Chương này kể về...".

NỘI DUNG CHƯƠNG:
{chapter_text}

TÓM TẮT:"""
    print(prompt[:5000] + "\n...")  # In phần đầu prompt để debug
    return prompt


def summarize_chapter(
    chapter: Chapter, target_words: int, context_text: str, language: str, model: str
) -> str:
    """
    Tóm tắt một chương. Nếu chương quá dài so với context window của model,
    tự động chia thành các phần nhỏ, tóm tắt từng phần rồi gộp lại.
    """
    words = chapter.text.split()
    total_words = len(words)

    if total_words <= MODEL_CONTEXT_WORD_LIMIT:
        # Chương vừa tầm — gửi toàn bộ một lần
        prompt = build_prompt(
            chapter_title=chapter.title,
            chapter_text=chapter.text,
            original_word_count=chapter.word_count,
            target_words=target_words,
            context_text=context_text,
            language=language,
        )
        return call_ollama(prompt, model)

    # Chương rất dài — chia chunk, tóm tắt từng phần rồi gộp
    chunk_size = MODEL_CONTEXT_WORD_LIMIT
    chunks = [words[i : i + chunk_size] for i in range(0, total_words, chunk_size)]
    num_chunks = len(chunks)
    target_per_chunk = max(30, target_words // num_chunks)

    chunk_summaries = []
    for i, chunk_words in enumerate(chunks):
        chunk_text = " ".join(chunk_words)
        chunk_info = f"Phần {i+1}/{num_chunks}"
        prompt = build_prompt(
            chapter_title=chapter.title,
            chapter_text=chunk_text,
            original_word_count=len(chunk_words),
            target_words=target_per_chunk,
            context_text=context_text if i == 0 else "",  # context chỉ cần ở phần đầu
            language=language,
            chunk_info=chunk_info,
        )
        chunk_summary = call_ollama(prompt, model)
        chunk_summaries.append(chunk_summary)

    # Nếu chỉ có 1 chunk dư thì không cần merge
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Gộp các chunk summaries lại thành bản tóm tắt cuối
    merged_text = "\n".join(chunk_summaries)
    merge_prompt = f"""Dưới đây là các đoạn tóm tắt của từng phần trong chương "{chapter.title}".
Hãy gộp chúng thành một đoạn tóm tắt duy nhất, liền mạch, khoảng {target_words} từ.
Chỉ trả về nội dung tóm tắt, không thêm tiêu đề hay ghi chú.

CÁC ĐOẠN TÓM TẮT:
{merged_text}

TÓM TẮT HOÀN CHỈNH:"""
    return call_ollama(merge_prompt, model)


# ─────────────────────────────────────────────
# Đọc EPUB
# ─────────────────────────────────────────────
def load_epub_chapters(epub_path: str) -> tuple[epub.EpubBook, list[Chapter]]:
    book = epub.read_epub(epub_path)
    chapters: list[Chapter] = []
    idx = 0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            raw_html = item.get_content().decode("utf-8", errors="replace")
            soup = BeautifulSoup(raw_html, "html.parser")

            # Tìm tiêu đề chương
            title_tag = soup.find(["h1", "h2", "h3", "h4"])
            if title_tag:
                title = clean_title(title_tag.get_text())
            else:
                title = item.get_name() or f"Chương {idx + 1}"

            text = html_to_text(raw_html)
            wc = count_words(text)

            # Bỏ qua trang quá ngắn (trang bìa, mục lục...)
            if wc < 50:
                continue

            chapters.append(
                Chapter(
                    index=idx,
                    title=title,
                    raw_html=raw_html,
                    text=text,
                    word_count=wc,
                )
            )
            idx += 1

    return book, chapters


# ─────────────────────────────────────────────
# Lưu / tải tiến trình
# ─────────────────────────────────────────────
def progress_path(output_path: str) -> str:
    return output_path + PROGRESS_FILE_SUFFIX


def save_progress(output_path: str, data: dict):
    with open(progress_path(output_path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_progress(output_path: str) -> dict:
    p = progress_path(output_path)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# Tạo EPUB đầu ra
# ─────────────────────────────────────────────
def build_output_epub(
    original_book: epub.EpubBook,
    chapters: list[Chapter],
    output_path: str,
    ratio: float,
):
    new_book = epub.EpubBook()

    # Sao chép metadata
    new_book.set_identifier(original_book.uid or "summarized-epub")
    titles = original_book.title
    new_book.set_title(f"{titles} [Tóm tắt {int(ratio*100)}%]" if titles else "Tóm tắt")

    for lang in original_book.language or ["vi"]:
        new_book.set_language(lang if isinstance(lang, str) else "vi")
        break

    new_book.add_author("AI Summarizer (Ollama)")

    # CSS đơn giản
    css_content = """
body { font-family: serif; line-height: 1.8; margin: 2em; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.3em; margin-top: 2em; color: #333; }
p  { margin: 0.8em 0; text-align: justify; }
.meta { color: #888; font-size: 0.85em; font-style: italic; margin-bottom: 1.5em; }
"""
    css_item = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=css_content.encode("utf-8"),
    )
    new_book.add_item(css_item)

    spine = ["nav"]
    toc_items = []

    for ch in chapters:
        if not ch.summary:
            continue  # bỏ qua chương chưa được tóm tắt

        safe_name = re.sub(r"[^\w]", "_", ch.title)[:40]
        file_name = f"chap_{ch.index:04d}_{safe_name}.xhtml"

        html_content = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <title>{ch.title}</title>
  <link rel="stylesheet" type="text/css" href="../style/main.css"/>
</head>
<body>
  <h2>{ch.title}</h2>
  <p class="meta">📖 Gốc: {ch.word_count} từ → Tóm tắt: {ch.summary_word_count} từ</p>
  {''.join(f'<p>{para.strip()}</p>' for para in ch.summary.split('\n') if para.strip())}
</body>
</html>"""

        epub_ch = epub.EpubHtml(
            title=ch.title,
            file_name=file_name,
            lang="vi",
        )
        epub_ch.content = html_content.encode("utf-8")

        epub_ch.add_item(css_item)
        new_book.add_item(epub_ch)
        spine.append(epub_ch)
        toc_items.append(epub.Link(file_name, ch.title, f"ch{ch.index}"))

    new_book.toc = toc_items
    new_book.spine = spine
    new_book.add_item(epub.EpubNcx())
    new_book.add_item(epub.EpubNav())

    epub.write_epub(output_path, new_book)
    print(f"\n✅ Đã lưu EPUB tóm tắt: {output_path}")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def summarize_epub(
    input_path: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    ratio: float = DEFAULT_RATIO,
    language: str = "auto",
    start_chapter: int = 0,
    end_chapter: Optional[int] = None,
    dry_run: bool = False,
):
    print(f"\n{'='*60}")
    print(f"  EPUB Summarizer — Ollama ({model})")
    print(f"{'='*60}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Tỉ lệ : {ratio*100:.0f}% (1/{round(1/ratio)})")
    print(f"  Model : {model}")
    print(f"{'='*60}\n")

    # Kiểm tra Ollama
    if not dry_run:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            models_available = [m["name"] for m in r.json().get("models", [])]
            print(
                f"✅ Ollama OK — Models: {', '.join(models_available) or '(không có)'}"
            )
            if model not in models_available and not any(
                model in m for m in models_available
            ):
                print(f"⚠️  Model '{model}' chưa pull. Chạy: ollama pull {model}")
        except Exception:
            print("❌ Không kết nối được Ollama. Hãy chạy: ollama serve")
            sys.exit(1)

    # Đọc EPUB
    print(f"\n📚 Đọc EPUB: {input_path}")
    book, chapters = load_epub_chapters(input_path)
    total = len(chapters)
    print(f"   Tổng số chương hợp lệ: {total}")

    if total == 0:
        print("❌ Không tìm thấy chương nào. Kiểm tra lại file EPUB.")
        sys.exit(1)

    # Áp dụng bộ lọc chương
    end_idx = end_chapter if end_chapter is not None else total
    chapters_to_process = chapters[start_chapter:end_idx]
    total_words = sum(c.word_count for c in chapters_to_process)
    print(
        f"   Xử lý chương {start_chapter}–{end_idx-1} ({len(chapters_to_process)} chương)"
    )
    print(f"   Tổng số từ: {total_words:,} → mục tiêu: {int(total_words * ratio):,} từ")

    if dry_run:
        print("\n[DRY RUN] Không gọi Ollama. Hiển thị danh sách chương:")
        for ch in chapters_to_process[:20]:
            tw = int(ch.word_count * ratio)
            print(
                f"  [{ch.index:4d}] {ch.title[:60]:60s} | {ch.word_count:5d} từ → {tw:4d} từ"
            )
        if len(chapters_to_process) > 20:
            print(f"  ... và {len(chapters_to_process)-20} chương nữa")
        return

    # Tải tiến trình đã lưu
    progress = load_progress(output_path)
    context = SummaryContext()

    # Khôi phục context từ tiến trình
    if progress:
        done_count = sum(1 for ch in chapters_to_process if str(ch.index) in progress)
        print(
            f"   ♻️  Tìm thấy tiến trình cũ: {done_count}/{len(chapters_to_process)} chương đã xử lý"
        )
        for ch in chapters_to_process:
            key = str(ch.index)
            if key in progress:
                ch.summary = progress[key]["summary"]
                ch.summary_word_count = progress[key]["word_count"]
                context.add(ch.title, ch.summary)

    # Tóm tắt từng chương
    pbar = tqdm(
        chapters_to_process,
        desc="Tóm tắt",
        unit="chương",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for ch in pbar:
        key = str(ch.index)
        if key in progress:
            pbar.set_postfix_str(f"⏭ skip: {ch.title[:30]}")
            continue

        target_words = max(50, int(ch.word_count * ratio))
        ctx_text = context.build_context_text()

        pbar.set_postfix_str(f"✍ {ch.title[:35]}")

        try:
            summary = summarize_chapter(ch, target_words, ctx_text, language, model)
        except RuntimeError as e:
            print(f"\n❌ {e}")
            print("💾 Đang lưu tiến trình trước khi dừng...")
            save_progress(output_path, progress)
            # Vẫn tạo EPUB với những gì đã xong
            build_output_epub(book, chapters, output_path, ratio)
            sys.exit(1)

        ch.summary = summary
        ch.summary_word_count = count_words(summary)

        progress[key] = {
            "title": ch.title,
            "original_words": ch.word_count,
            "word_count": ch.summary_word_count,
            "summary": summary,
        }

        context.add(ch.title, summary)

        # Lưu tiến trình sau mỗi chương
        save_progress(output_path, progress)

    # Thống kê
    done_chs = [ch for ch in chapters_to_process if ch.summary]
    orig_total = sum(ch.word_count for ch in done_chs)
    summ_total = sum(ch.summary_word_count for ch in done_chs)
    actual_ratio = summ_total / orig_total if orig_total else 0

    print(f"\n📊 Thống kê:")
    print(f"   Chương đã tóm tắt : {len(done_chs)}/{len(chapters_to_process)}")
    print(f"   Từ gốc            : {orig_total:,}")
    print(f"   Từ sau tóm tắt    : {summ_total:,}")
    print(f"   Tỉ lệ thực tế     : {actual_ratio*100:.1f}%")

    # Ghi EPUB
    build_output_epub(book, chapters, output_path, ratio)

    # Xoá file tiến trình nếu hoàn thành toàn bộ
    if len(done_chs) == len(chapters_to_process):
        p = progress_path(output_path)
        if os.path.exists(p):
            os.remove(p)
            print("🗑️  Đã xoá file tiến trình (hoàn thành).")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Rút gọn bộ truyện EPUB siêu dài xuống còn ~1/10 bằng Ollama AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tóm tắt toàn bộ, dùng llama3
  python epub_summarizer.py "Phong Thần Ký.epub" "Phong Thần Ký_tomtat.epub"

  # Tóm tắt 1/5 độ dài, dùng mistral
  python epub_summarizer.py input.epub output.epub --ratio 0.2 --model mistral

  # Chỉ xử lý chương 100-200, output tiếng Việt
  python epub_summarizer.py input.epub output.epub --start-chapter 100 --end-chapter 200 --lang vi

  # Xem danh sách chương không gọi AI
  python epub_summarizer.py input.epub output.epub --dry-run
        """,
    )
    parser.add_argument("input", help="Đường dẫn file EPUB đầu vào")
    parser.add_argument("output", help="Đường dẫn file EPUB đầu ra")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Tên model Ollama (mặc định: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=f"Tỉ lệ tóm tắt 0.0–1.0 (mặc định: {DEFAULT_RATIO} = 10%%)",
    )
    parser.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "vi", "en"],
        help="Ngôn ngữ đầu ra: auto/vi/en (mặc định: auto)",
    )
    parser.add_argument(
        "--start-chapter",
        type=int,
        default=0,
        help="Chương bắt đầu (0-indexed, mặc định: 0)",
    )
    parser.add_argument(
        "--end-chapter",
        type=int,
        default=None,
        help="Chương kết thúc (exclusive, mặc định: hết truyện)",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help=f"URL Ollama API (mặc định: {OLLAMA_URL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Chỉ liệt kê chương, không gọi AI"
    )

    args = parser.parse_args()

    # global OLLAMA_URL
    # OLLAMA_URL = args.ollama_url

    if not os.path.exists(args.input):
        print(f"❌ File không tồn tại: {args.input}")
        sys.exit(1)

    if not 0 < args.ratio <= 1:
        print("❌ --ratio phải trong khoảng (0, 1]")
        sys.exit(1)

    # OLLAMA_URL = args.ollama_url

    summarize_epub(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        ratio=args.ratio,
        language=args.lang,
        start_chapter=args.start_chapter,
        end_chapter=args.end_chapter,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
