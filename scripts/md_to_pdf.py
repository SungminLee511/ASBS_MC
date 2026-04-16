#!/usr/bin/env python
"""Convert RESULT.md to RESULT.pdf using fpdf2 with inline PNGs."""
import re
from pathlib import Path
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
md_path = ROOT / "RESULT.md"
pdf_path = ROOT / "RESULT.pdf"


class ResultPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Mode Concentration in ASBS: Experimental Results", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font("Helvetica", "B", 18)
            self.ln(6)
        elif level == 2:
            self.set_font("Helvetica", "B", 14)
            self.ln(4)
        else:
            self.set_font("Helvetica", "B", 11)
            self.ln(3)
        self.multi_cell(0, 8 if level <= 2 else 6, title)
        if level <= 2:
            self.set_draw_color(200, 200, 200)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)
        else:
            self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        # Handle bold markers
        text = text.strip()
        if not text:
            return
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_image_row(self, images, captions=None):
        """Add a row of images side by side."""
        valid = [(img, cap) for img, cap in zip(images, captions or [""]*len(images))
                 if (ROOT / img).exists()]
        if not valid:
            return
        n = len(valid)
        avail_w = self.w - self.l_margin - self.r_margin - (n - 1) * 3
        img_w = avail_w / n

        # Check if we need a page break (estimate ~60mm per image row)
        if self.get_y() + 65 > self.h - 20:
            self.add_page()

        start_x = self.l_margin
        start_y = self.get_y()
        max_h = 0

        for i, (img, cap) in enumerate(valid):
            x = start_x + i * (img_w + 3)
            self.image(str(ROOT / img), x=x, y=start_y, w=img_w)
            # Estimate image height from aspect ratio
            from PIL import Image as PILImage
            pil = PILImage.open(ROOT / img)
            aspect = pil.height / pil.width
            h = img_w * aspect
            if h > max_h:
                max_h = h

        self.set_y(start_y + max_h + 2)
        # Add captions if provided
        if captions and any(c for c in captions):
            self.set_font("Helvetica", "I", 7)
            for i, (_, cap) in enumerate(valid):
                if cap:
                    x = start_x + i * (img_w + 3)
                    self.set_x(x)
                    self.cell(img_w, 4, cap, align="C")
            self.ln(6)
        else:
            self.ln(3)

    def add_table(self, headers, rows):
        """Render a simple table."""
        if self.get_y() + 10 + len(rows) * 6 > self.h - 20:
            self.add_page()
        n_cols = len(headers)
        avail_w = self.w - self.l_margin - self.r_margin
        col_w = avail_w / n_cols

        # Header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(240, 242, 245)
        for h in headers:
            self.cell(col_w, 6, h.strip(), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows):
            if i % 2 == 1:
                self.set_fill_color(249, 250, 251)
                fill = True
            else:
                fill = False
            for cell in row:
                self.cell(col_w, 5.5, cell.strip(), border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)


def parse_and_render(pdf, md_text):
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip TOC
        if line.strip().startswith("1. [") or line.strip().startswith("2. [") or \
           (line.strip().startswith(("3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")) and "[" in line):
            i += 1
            continue

        # Headers
        if line.startswith("# ") and not line.startswith("## "):
            pdf.chapter_title(line[2:].strip(), level=1)
            i += 1
            continue
        if line.startswith("## "):
            pdf.chapter_title(line[3:].strip(), level=2)
            i += 1
            continue
        if line.startswith("### "):
            pdf.chapter_title(line[4:].strip(), level=3)
            i += 1
            continue

        # Image tables: |...|...|...| where cells contain ![](...)
        if line.strip().startswith("|") and "![" in line:
            # This is an image row — parse it
            imgs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if imgs:
                captions = [cap for cap, _ in imgs]
                paths = [p for _, p in imgs]
                # Skip the separator line
                i += 1
                if i < len(lines) and lines[i].strip().startswith("|") and "---" in lines[i]:
                    i += 1
                # Check if next line is also images
                if i < len(lines) and "![" in lines[i]:
                    imgs2 = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', lines[i])
                    if imgs2:
                        pdf.add_image_row(paths, captions)
                        captions2 = [cap for cap, _ in imgs2]
                        paths2 = [p for _, p in imgs2]
                        pdf.add_image_row(paths2, captions2)
                        i += 1
                        continue
                pdf.add_image_row(paths, captions)
                continue
            i += 1
            continue

        # Standalone images: ![...](...)
        img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if img_match:
            cap, path = img_match.groups()
            full = ROOT / path
            if full.exists():
                pdf.add_image_row([path], [cap])
            i += 1
            continue

        # Tables: | header | header |
        if line.strip().startswith("|") and "|" in line[1:]:
            headers = [c.strip() for c in line.strip().strip("|").split("|")]
            # Skip separator
            i += 1
            if i < len(lines) and "---" in lines[i]:
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if "![" in lines[i]:
                    # Image table row — handle separately
                    imgs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', lines[i])
                    if imgs:
                        captions = [cap for cap, _ in imgs]
                        paths = [p for _, p in imgs]
                        pdf.add_table(headers, rows) if rows else None
                        rows = []
                        pdf.add_image_row(paths, captions)
                    i += 1
                    continue
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                # Clean markdown bold
                cells = [re.sub(r'\*\*([^*]+)\*\*', r'\1', c) for c in cells]
                rows.append(cells)
                i += 1
            if rows:
                pdf.add_table(headers, rows)
            continue

        # Horizontal rule
        if line.strip() == "---":
            i += 1
            continue

        # Bullet/numbered lists and body text
        if line.strip().startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 10)
            # Clean bold markers
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', line.strip()[2:])
            pdf.cell(5)
            pdf.multi_cell(0, 5, f"  * {text}")
            pdf.ln(1)
            i += 1
            continue

        numbered = re.match(r'^(\d+)\.\s+(.+)', line.strip())
        if numbered:
            pdf.set_font("Helvetica", "", 10)
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', numbered.group(2))
            pdf.cell(5)
            pdf.multi_cell(0, 5, f"  {numbered.group(1)}. {text}")
            pdf.ln(1)
            i += 1
            continue

        # Bold-only lines (like **Setup:**)
        if line.strip().startswith("**") and line.strip().endswith("**"):
            pdf.set_font("Helvetica", "B", 10)
            text = line.strip().strip("*")
            pdf.multi_cell(0, 5, text)
            pdf.ln(2)
            i += 1
            continue

        # Regular text
        text = line.strip()
        if text:
            # Clean markdown
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'`([^`]+)`', r'\1', text)
            pdf.body_text(text)

        i += 1


def main():
    pdf = ResultPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    md_text = md_path.read_text(encoding="utf-8")
    # Replace special unicode chars that fpdf can't handle
    md_text = md_text.replace("\u2014", " -- ")
    md_text = md_text.replace("\u2013", " - ")
    md_text = md_text.replace("\u2265", ">=")
    md_text = md_text.replace("\u2264", "<=")
    md_text = md_text.replace("\u2248", "~")
    md_text = md_text.replace("\u03b1", "alpha")
    md_text = md_text.replace("\u03b2", "beta")
    md_text = md_text.replace("\u03b3", "gamma")
    md_text = md_text.replace("\u03ba", "kappa")
    md_text = md_text.replace("\u03c1", "rho")
    md_text = md_text.replace("\u03c3", "sigma")
    md_text = md_text.replace("\u2016", "||")
    md_text = md_text.replace("\u2192", "->")
    md_text = md_text.replace("\u00b1", "+/-")
    md_text = md_text.replace("\u00d7", "x")
    md_text = md_text.replace("\u0134", "J")
    md_text = md_text.replace("\u03b7", "eta")
    md_text = md_text.replace("\u2208", " in ")

    # Subscripts
    for c in "0123456789":
        md_text = md_text.replace(f"\u2080"[0], "0")  # just in case
    md_text = md_text.replace("\u2080", "0")
    md_text = md_text.replace("\u2081", "1")
    md_text = md_text.replace("\u2082", "2")
    md_text = md_text.replace("\u2083", "3")
    md_text = md_text.replace("\u2084", "4")
    md_text = md_text.replace("\u2085", "5")
    md_text = md_text.replace("\u2086", "6")
    md_text = md_text.replace("\u2087", "7")
    md_text = md_text.replace("\u2088", "8")
    md_text = md_text.replace("\u2089", "9")
    md_text = md_text.replace("\u2093", "x")
    md_text = md_text.replace("\u2096", "k")

    parse_and_render(pdf, md_text)
    pdf.output(str(pdf_path))
    print(f"Done! PDF saved to {pdf_path}")
    print(f"Size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
