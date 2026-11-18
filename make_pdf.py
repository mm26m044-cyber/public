"""
make_pdf.py
-----------
Self-contained, pure Python script to generate a publication-quality, multi-page PDF document
summarizing the Band Explorer & Ewald Sphere Diffraction project for Claude / Resume Building.
Zero external dependencies.
"""

import sys
import os

# Standard character widths for Helvetica (per 1000 units)
HELVETICA_WIDTHS = {
    ' ': 278, '!': 278, '"': 355, '#': 556, '$': 556, '%': 889, '&': 667, "'": 191,
    '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333, '.': 278, '/': 278,
    '0': 556, '1': 556, '2': 556, '3': 556, '4': 556, '5': 556, '6': 556, '7': 556,
    '8': 556, '9': 556, ':': 278, ';': 278, '<': 584, '=': 584, '>': 584, '?': 556,
    '@': 1015, 'A': 667, 'B': 667, 'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778,
    'H': 722, 'I': 278, 'J': 500, 'K': 667, 'L': 556, 'M': 833, 'N': 722, 'O': 778,
    'P': 667, 'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722, 'V': 667, 'W': 944,
    'X': 667, 'Y': 667, 'Z': 611, '[': 278, '\\': 278, ']': 278, '^': 469, '_': 556,
    '`': 333, 'a': 556, 'b': 556, 'c': 500, 'd': 556, 'e': 556, 'f': 278, 'g': 556,
    'h': 556, 'i': 222, 'j': 222, 'k': 500, 'l': 222, 'm': 833, 'n': 556, 'o': 556,
    'p': 556, 'q': 556, 'r': 333, 's': 500, 't': 278, 'u': 556, 'v': 500, 'w': 722,
    'x': 500, 'y': 500, 'z': 500, '{': 334, '|': 260, '}': 334, '~': 584
}

HELVETICA_BOLD_WIDTHS = {
    ' ': 278, '!': 333, '"': 474, '#': 556, '$': 556, '%': 889, '&': 722, "'": 238,
    '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333, '.': 278, '/': 278,
    '0': 556, '1': 556, '2': 556, '3': 556, '4': 556, '5': 556, '6': 556, '7': 556,
    '8': 556, '9': 556, ':': 333, ';': 333, '<': 584, '=': 584, '>': 584, '?': 611,
    '@': 975, 'A': 722, 'B': 722, 'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778,
    'H': 722, 'I': 278, 'J': 556, 'K': 722, 'L': 611, 'M': 833, 'N': 722, 'O': 778,
    'P': 667, 'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722, 'V': 667, 'W': 944,
    'X': 667, 'Y': 667, 'Z': 611, '[': 333, '\\': 278, ']': 333, '^': 584, '_': 556,
    '`': 333, 'a': 556, 'b': 611, 'c': 556, 'd': 611, 'e': 556, 'f': 333, 'g': 611,
    'h': 611, 'i': 278, 'j': 278, 'k': 556, 'l': 278, 'm': 889, 'n': 611, 'o': 611,
    'p': 611, 'q': 611, 'r': 389, 's': 556, 't': 333, 'u': 611, 'v': 556, 'w': 778,
    'x': 556, 'y': 556, 'z': 500, '{': 389, '|': 280, '}': 389, '~': 584
}


def get_string_width(text: str, font: str = 'Helvetica', font_size: float = 10.0) -> float:
    widths = HELVETICA_BOLD_WIDTHS if 'Bold' in font else HELVETICA_WIDTHS
    total = 0
    for ch in text:
        total += widths.get(ch, 550)
    return total * font_size / 1000.0


def escape_pdf_text(text: str) -> str:
    # Replace unicode special characters with clean ASCII equivalents
    replacements = {
        '—': '--', '–': '-', '’': "'", '‘': "'", '“': '"', '”': '"',
        '•': '*', '·': '*', '≈': '~=', '≤': '<=', '≥': '>=', '≠': '!=',
        '±': '+/-', 'µ': 'u', 'μ': 'u', 'τ': 'tau', 'λ': 'lambda',
        'Γ': 'Gamma', 'Δ': 'Delta', 'Å': 'Angstrom', 'ħ': 'hbar',
        'θ': 'theta', '₀': '0', '₁': '1', '₂': '2', '₃': '3',
        '²': '^2', '⁴': '^4', '⁻': '^-', 'π': 'pi', 'Σ': 'Sum',
        '→': '->', '←': '<-', '⇒': '=>', '°': ' deg'
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    
    # Escape PDF parentheses and backslashes
    res = []
    for c in text:
        if c in ('\\', '(', ')'):
            res.append('\\' + c)
        elif ord(c) < 32 or ord(c) > 126:
            res.append('?')
        else:
            res.append(c)
    return ''.join(res)


class PDFBuilder:
    def __init__(self, filename: str, page_width: float = 612.0, page_height: float = 792.0):
        self.filename = filename
        self.width = page_width
        self.height = page_height
        self.margin_x = 48.0
        self.margin_top = 54.0
        self.margin_bottom = 50.0
        self.content_width = self.width - 2 * self.margin_x
        
        self.pages = []
        self.current_stream = []
        self.current_y = self.height - self.margin_top
        
        # Color palette (RGB floats 0-1)
        self.c_navy = (0.06, 0.09, 0.16)       # #0f172a
        self.c_royal = (0.11, 0.31, 0.85)      # #1d4ed8
        self.c_slate = (0.28, 0.33, 0.41)      # #475569
        self.c_dark = (0.12, 0.16, 0.23)       # #1e293b
        self.c_emerald = (0.02, 0.59, 0.41)    # #059669
        self.c_amber = (0.85, 0.45, 0.05)      # #d97706
        self.c_light_bg = (0.96, 0.97, 0.99)   # #f8fafc
        self.c_card_border = (0.88, 0.91, 0.94)

        self._start_new_page()

    def _start_new_page(self):
        if self.pages:
            self.pages.append(b"".join(self.current_stream))
            self.current_stream = []
        else:
            self.pages.append(b"")  # placeholder for page 0
            self.current_stream = []
        self.current_y = self.height - self.margin_top

    def check_space(self, needed_height: float):
        if self.current_y - needed_height < self.margin_bottom:
            self._start_new_page()

    def draw_rect(self, x: float, y: float, w: float, h: float, fill_color=None, stroke_color=None, line_width=1.0):
        s = []
        if stroke_color:
            s.append(f"{stroke_color[0]:.3f} {stroke_color[1]:.3f} {stroke_color[2]:.3f} RG\n")
            s.append(f"{line_width:.2f} w\n")
        if fill_color:
            s.append(f"{fill_color[0]:.3f} {fill_color[1]:.3f} {fill_color[2]:.3f} rg\n")
        
        s.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re\n")
        if fill_color and stroke_color:
            s.append("B\n")
        elif fill_color:
            s.append("f\n")
        elif stroke_color:
            s.append("S\n")
        self.current_stream.append("".join(s).encode('latin1'))

    def draw_text(self, text: str, x: float, y: float, font: str = 'Helvetica', size: float = 10.0, color=(0, 0, 0)):
        font_tag = '/F2' if 'Bold' in font else ('/F3' if 'Courier' in font else '/F1')
        escaped = escape_pdf_text(text)
        stream_cmd = (
            f"BT\n"
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg\n"
            f"{font_tag} {size:.2f} Tf\n"
            f"1 0 0 1 {x:.2f} {y:.2f} Tm\n"
            f"({escaped}) Tj\n"
            f"ET\n"
        )
        self.current_stream.append(stream_cmd.encode('latin1'))

    def add_page_header(self, page_num: int):
        # Top banner line
        y = self.height - 28.0
        self.draw_rect(self.margin_x, y + 2, self.content_width, 1.5, fill_color=self.c_royal)
        self.draw_text("BAND EXPLORER -- COMPUTATIONAL MATERIALS PHYSICS & INFORMATICS PLATFORM", 
                       self.margin_x, y + 7, font='Helvetica-Bold', size=7.5, color=self.c_slate)
        self.draw_text("PROJECT BRIEFING FOR RESUME / CLAUDE", 
                       self.width - self.margin_x - 170.0, y + 7, font='Helvetica-Bold', size=7.5, color=self.c_royal)

    def add_page_footer(self, page_num: int, total_pages: int):
        y = 28.0
        self.draw_rect(self.margin_x, y + 14, self.content_width, 0.75, fill_color=self.c_card_border)
        footer_text = f"Page {page_num} of {total_pages}  |  Materials Project API * pymatgen * Plotly.js * Python Solid-State Simulation"
        self.draw_text(footer_text, self.margin_x, y, font='Helvetica', size=7.5, color=self.c_slate)
        self.draw_text("CONFIDENTIAL / PORTFOLIO DOCUMENT", self.width - self.margin_x - 145.0, y, font='Helvetica-Bold', size=7.0, color=self.c_royal)

    def add_title_block(self, title: str, subtitle: str, meta_badges: list):
        self.check_space(110.0)
        # Main background card for title
        box_h = 95.0
        box_y = self.current_y - box_h
        self.draw_rect(self.margin_x, box_y, self.content_width, box_h, fill_color=self.c_light_bg, stroke_color=self.c_royal, line_width=1.5)
        # Left blue accent bar
        self.draw_rect(self.margin_x, box_y, 5.0, box_h, fill_color=self.c_royal)

        # Title
        self.draw_text(title, self.margin_x + 18.0, box_y + 65.0, font='Helvetica-Bold', size=18.0, color=self.c_navy)
        # Subtitle
        self.draw_text(subtitle, self.margin_x + 18.0, box_y + 47.0, font='Helvetica', size=10.0, color=self.c_slate)

        # Badges row
        cur_x = self.margin_x + 18.0
        badge_y = box_y + 16.0
        for badge_text, bg_col, text_col in meta_badges:
            w = get_string_width(badge_text, 'Helvetica-Bold', 8.0) + 14.0
            self.draw_rect(cur_x, badge_y - 2, w, 16.0, fill_color=bg_col, stroke_color=text_col, line_width=0.75)
            self.draw_text(badge_text, cur_x + 7.0, badge_y + 2.5, font='Helvetica-Bold', size=8.0, color=text_col)
            cur_x += w + 8.0

        self.current_y = box_y - 18.0

    def add_section_header(self, title: str, icon_tag: str = ""):
        self.check_space(38.0)
        full_title = f"{icon_tag}  {title}".strip()
        y = self.current_y - 18.0
        # Accent left box
        self.draw_rect(self.margin_x, y - 2, 4.0, 18.0, fill_color=self.c_royal)
        self.draw_rect(self.margin_x + 6.0, y - 2, self.content_width - 6.0, 18.0, fill_color=(0.95, 0.96, 0.98))
        self.draw_text(full_title, self.margin_x + 14.0, y + 2.5, font='Helvetica-Bold', size=11.5, color=self.c_navy)
        self.current_y = y - 10.0

    def add_subsection_header(self, title: str):
        self.check_space(24.0)
        y = self.current_y - 14.0
        self.draw_text(title, self.margin_x, y, font='Helvetica-Bold', size=10.5, color=self.c_royal)
        self.current_y = y - 4.0

    def add_paragraph(self, text: str, font_size: float = 9.0, line_spacing: float = 12.5, color=None):
        if color is None:
            color = self.c_dark
        words = text.split()
        lines = []
        current_line = []
        
        for w in words:
            test_line = " ".join(current_line + [w])
            if get_string_width(test_line, 'Helvetica', font_size) <= self.content_width:
                current_line.append(w)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [w]
        if current_line:
            lines.append(" ".join(current_line))

        for l in lines:
            self.check_space(line_spacing)
            self.draw_text(l, self.margin_x, self.current_y - font_size, font='Helvetica', size=font_size, color=color)
            self.current_y -= line_spacing
        self.current_y -= 4.0

    def add_bullet(self, title: str, body: str, font_size: float = 8.5, line_spacing: float = 11.5):
        prefix = "-- "
        title_w = get_string_width(prefix + title + ": ", 'Helvetica-Bold', font_size)
        indent = self.margin_x + 14.0
        available_first = self.content_width - 14.0 - title_w

        # Wrap body text
        words = body.split()
        lines = []
        cur = []
        first_line = True
        
        for w in words:
            cur_try = " ".join(cur + [w])
            max_w = available_first if first_line else (self.content_width - 14.0)
            if get_string_width(cur_try, 'Helvetica', font_size) <= max_w:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                    first_line = False
                cur = [w]
        if cur:
            lines.append(" ".join(cur))

        self.check_space((len(lines) + 1) * line_spacing)
        
        # Draw bullet mark and title
        y = self.current_y - font_size
        self.draw_text("*", self.margin_x + 4.0, y, font='Helvetica-Bold', size=font_size, color=self.c_royal)
        self.draw_text(title + ":", indent, y, font='Helvetica-Bold', size=font_size, color=self.c_navy)
        
        if lines:
            self.draw_text(lines[0], indent + get_string_width(title + ": ", 'Helvetica-Bold', font_size), y, font='Helvetica', size=font_size, color=self.c_dark)
            self.current_y -= line_spacing
            for rem in lines[1:]:
                self.check_space(line_spacing)
                self.draw_text(rem, indent, self.current_y - font_size, font='Helvetica', size=font_size, color=self.c_dark)
                self.current_y -= line_spacing
        else:
            self.current_y -= line_spacing
        self.current_y -= 2.0

    def add_resume_bullet(self, action_verb: str, content: str, font_size: float = 8.5, line_spacing: float = 11.5):
        full_text = f"**{action_verb}** {content}"
        # Bullet layout
        indent = self.margin_x + 14.0
        words = content.split()
        prefix = f"{action_verb} "
        prefix_w = get_string_width(prefix, 'Helvetica-Bold', font_size)

        lines = []
        cur = []
        first = True
        for w in words:
            cur_try = " ".join(cur + [w])
            max_w = (self.content_width - 14.0 - prefix_w) if first else (self.content_width - 14.0)
            if get_string_width(cur_try, 'Helvetica', font_size) <= max_w:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                    first = False
                cur = [w]
        if cur:
            lines.append(" ".join(cur))

        needed = max(1, len(lines)) * line_spacing + 3.0
        self.check_space(needed)
        y = self.current_y - font_size
        self.draw_text(">", self.margin_x + 4.0, y, font='Helvetica-Bold', size=font_size + 1.0, color=self.c_royal)
        self.draw_text(prefix, indent, y, font='Helvetica-Bold', size=font_size, color=self.c_navy)

        if lines:
            self.draw_text(lines[0], indent + prefix_w, y, font='Helvetica', size=font_size, color=self.c_dark)
            self.current_y -= line_spacing
            for rem in lines[1:]:
                self.check_space(line_spacing)
                self.draw_text(rem, indent, self.current_y - font_size, font='Helvetica', size=font_size, color=self.c_dark)
                self.current_y -= line_spacing
        else:
            self.current_y -= line_spacing
        self.current_y -= 3.0

    def add_callout_box(self, title: str, text: str, border_color, bg_color):
        font_size = 8.5
        spacing = 11.5
        words = text.split()
        lines = []
        cur = []
        for w in words:
            if get_string_width(" ".join(cur + [w]), 'Helvetica', font_size) <= self.content_width - 24.0:
                cur.append(w)
            else:
                lines.append(" ".join(cur))
                cur = [w]
        if cur:
            lines.append(" ".join(cur))

        box_h = 24.0 + len(lines) * spacing + 8.0
        self.check_space(box_h + 8.0)
        box_y = self.current_y - box_h

        self.draw_rect(self.margin_x, box_y, self.content_width, box_h, fill_color=bg_color, stroke_color=border_color, line_width=1.0)
        self.draw_rect(self.margin_x, box_y, 4.0, box_h, fill_color=border_color)

        self.draw_text(title, self.margin_x + 12.0, box_y + box_h - 16.0, font='Helvetica-Bold', size=9.0, color=border_color)
        cy = box_y + box_h - 28.0
        for l in lines:
            self.draw_text(l, self.margin_x + 12.0, cy, font='Helvetica', size=font_size, color=self.c_dark)
            cy -= spacing

        self.current_y = box_y - 8.0

    def add_table(self, headers: list, rows: list, col_widths: list):
        row_h = 16.0
        header_h = 18.0
        total_h = header_h + len(rows) * row_h
        self.check_space(total_h + 10.0)

        y = self.current_y - header_h
        # Header background
        self.draw_rect(self.margin_x, y, self.content_width, header_h, fill_color=self.c_navy)
        
        # Header text
        cx = self.margin_x
        for h, w in zip(headers, col_widths):
            self.draw_text(h, cx + 5.0, y + 5.0, font='Helvetica-Bold', size=8.0, color=(1, 1, 1))
            cx += w

        # Rows
        for i, row in enumerate(rows):
            y -= row_h
            bg = self.c_light_bg if (i % 2 == 0) else (1, 1, 1)
            self.draw_rect(self.margin_x, y, self.content_width, row_h, fill_color=bg, stroke_color=self.c_card_border, line_width=0.5)
            
            cx = self.margin_x
            for val, w in zip(row, col_widths):
                self.draw_text(str(val), cx + 5.0, y + 4.5, font='Helvetica', size=8.0, color=self.c_dark)
                cx += w

        self.current_y = y - 8.0

    def save(self):
        # Finalize pages and append footer/header
        if self.current_stream:
            self.pages.append(b"".join(self.current_stream))
        
        actual_pages = [p for p in self.pages if len(p) > 0]
        total_p = len(actual_pages)

        # Build complete PDF object graph
        pdf_objs = []
        
        # Object 1: Catalog
        catalog_str = b"<< /Type /Catalog /Pages 2 0 R >>"
        
        # Object 2: Pages
        kids_str = " ".join(f"{3 + i * 2} 0 R" for i in range(total_p))
        pages_str = f"<< /Type /Pages /Kids [{kids_str}] /Count {total_p} >>".encode('latin1')

        # Fonts (4, 5, 6)
        f_helv = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        f_bold = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
        f_cour = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"

        obj_list = [catalog_str, pages_str]
        
        # For each page, create page object and content stream object
        font_base_idx = 3 + total_p * 2
        f1_ref = f"{font_base_idx} 0 R"
        f2_ref = f"{font_base_idx + 1} 0 R"
        f3_ref = f"{font_base_idx + 2} 0 R"

        page_objs = []
        stream_objs = []

        for idx, page_content in enumerate(actual_pages):
            page_num = idx + 1
            # Add running header and footer directly into page stream
            hf_stream = []
            
            # Header
            y_h = self.height - 28.0
            hf_stream.append(f"{self.c_royal[0]:.3f} {self.c_royal[1]:.3f} {self.c_royal[2]:.3f} rg\n{self.margin_x:.2f} {y_h + 2:.2f} {self.content_width:.2f} 1.50 re\nf\n")
            hf_stream.append(f"BT\n{self.c_slate[0]:.3f} {self.c_slate[1]:.3f} {self.c_slate[2]:.3f} rg\n/F2 7.50 Tf\n1 0 0 1 {self.margin_x:.2f} {y_h + 6:.2f} Tm\n(BAND EXPLORER -- COMPUTATIONAL MATERIALS PHYSICS & INFORMATICS PLATFORM) Tj\nET\n")
            hf_stream.append(f"BT\n{self.c_royal[0]:.3f} {self.c_royal[1]:.3f} {self.c_royal[2]:.3f} rg\n/F2 7.50 Tf\n1 0 0 1 {self.width - self.margin_x - 175.0:.2f} {y_h + 6:.2f} Tm\n(PROJECT BRIEFING FOR RESUME / CLAUDE) Tj\nET\n")

            # Footer
            y_f = 28.0
            hf_stream.append(f"{self.c_card_border[0]:.3f} {self.c_card_border[1]:.3f} {self.c_card_border[2]:.3f} rg\n{self.margin_x:.2f} {y_f + 14:.2f} {self.content_width:.2f} 0.75 re\nf\n")
            foot_str = f"Page {page_num} of {total_p}  |  Materials Project API * pymatgen * Plotly.js * Python Solid-State Simulation"
            hf_stream.append(f"BT\n{self.c_slate[0]:.3f} {self.c_slate[1]:.3f} {self.c_slate[2]:.3f} rg\n/F1 7.50 Tf\n1 0 0 1 {self.margin_x:.2f} {y_f:.2f} Tm\n({foot_str}) Tj\nET\n")
            hf_stream.append(f"BT\n{self.c_royal[0]:.3f} {self.c_royal[1]:.3f} {self.c_royal[2]:.3f} rg\n/F2 7.00 Tf\n1 0 0 1 {self.width - self.margin_x - 145.0:.2f} {y_f:.2f} Tm\n(CONFIDENTIAL / PORTFOLIO BRIEFING) Tj\nET\n")

            full_stream_bytes = b"".join(s.encode('latin1') if isinstance(s, str) else s for s in hf_stream) + page_content
            
            page_obj_num = 3 + idx * 2
            stream_obj_num = page_obj_num + 1

            p_obj = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
                f"/Resources << /Font << /F1 {f1_ref} /F2 {f2_ref} /F3 {f3_ref} >> >> "
                f"/Contents {stream_obj_num} 0 R >>"
            ).encode('latin1')

            s_obj = (
                f"<< /Length {len(full_stream_bytes)} >>\nstream\n".encode('latin1') +
                full_stream_bytes +
                b"\nendstream"
            )

            page_objs.append(p_obj)
            stream_objs.append(s_obj)

        for p_o, s_o in zip(page_objs, stream_objs):
            obj_list.append(p_o)
            obj_list.append(s_o)

        # Append fonts
        obj_list.extend([f_helv, f_bold, f_cour])

        # Write PDF to file
        with open(self.filename, 'wb') as f:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
            offsets = []
            
            for idx, o in enumerate(obj_list):
                offsets.append(f.tell())
                obj_num = idx + 1
                f.write(f"{obj_num} 0 obj\n".encode('latin1'))
                f.write(o)
                f.write(b"\nendobj\n")

            xref_offset = f.tell()
            f.write(b"xref\n")
            f.write(f"0 {len(obj_list) + 1}\n".encode('latin1'))
            f.write(b"0000000000 65535 f \n")
            for off in offsets:
                f.write(f"{off:010d} 00000 n \n".encode('latin1'))

            f.write(b"trailer\n")
            f.write(f"<< /Size {len(obj_list) + 1} /Root 1 0 R >>\n".encode('latin1'))
            f.write(b"startxref\n")
            f.write(f"{xref_offset}\n".encode('latin1'))
            f.write(b"%%EOF\n")

        print(f"Successfully generated PDF: {self.filename} ({len(actual_pages)} pages)")


def build_full_documentation():
    target_path = "Band_Explorer_Project_Briefing_for_Claude.pdf"
    doc = PDFBuilder(target_path)

    # PAGE 1: TITLE, PROJECT SUMMARY & CORE HIGHLIGHTS
    doc.add_title_block(
        title="BAND EXPLORER & EWALD SPHERE PLATFORM",
        subtitle="Computational Solid-State Physics, Materials Informatics & Interactive 3D Diffraction Simulator",
        meta_badges=[
            ("Python 3.14", (0.9, 0.95, 1.0), (0.1, 0.3, 0.85)),
            ("Materials Project API", (0.9, 0.98, 0.93), (0.02, 0.55, 0.35)),
            ("pymatgen", (0.98, 0.95, 0.9), (0.8, 0.4, 0.05)),
            ("Plotly.js 3D/2D", (0.95, 0.92, 1.0), (0.45, 0.15, 0.75)),
            ("Crystallography & DFT", (0.92, 0.96, 1.0), (0.08, 0.45, 0.75)),
        ]
    )

    doc.add_section_header("1. Executive Overview & Problem Statement", "[OVERVIEW]")
    doc.add_paragraph(
        "Band Explorer is an end-to-end computational materials science and electronic structure analysis platform. "
        "Modern density functional theory (DFT) databases like the Materials Project host petabytes of valuable electronic "
        "and crystallographic data, but extracting device-relevant physical observables (effective mass tensors, conduction valley "
        "minima, carrier mobility, and multi-wavelength crystallographic diffraction patterns) typically requires fragmented, "
        "fragile scripting. Band Explorer solves this by integrating an automated API-driven ingestion engine, an offline-first "
        "caching pipeline, a unified analytical physics core, and a responsive web dashboard with publication-quality exports."
    )

    doc.add_callout_box(
        "CORE VALUE PROPOSITION FOR RESUME / PORTFOLIO",
        "Demonstrates mastery across computational condensed matter physics, materials informatics, numerical finite-difference "
        "algorithms, full-stack scientific application development, and 3D/2D interactive visualization with zero external web dependencies.",
        doc.c_royal, doc.c_light_bg
    )

    doc.add_section_header("2. Key Impact Metrics & System Capabilities", "[METRICS]")
    doc.add_table(
        headers=["Functional Dimension", "Technical Capability", "Physical Scope / Formula / Standard"],
        rows=[
            ["Electronic Structure", "Band Gap & Valley Analysis", "Direct/Indirect gap, VBM/CBM, Gamma, L, Delta, X, K valleys"],
            ["Transport Physics", "Effective Mass & Mobility", "Non-uniform 3-point d2E/dk2 curvature & Drude mobility (mu = e*tau/m*)"],
            ["State Classification", "Metals vs. Semiconductors", "Auto-detection of Fermi crossings, zero gap, and DOS N(E_F)"],
            ["Crystallography", "3D Ewald Sphere Simulator", "Reciprocal lattice (b1,b2,b3), Laue condition, dynamic mosaicity"],
            ["Diffraction Optics", "Kinematical Structure Factor", "F_hkl = Sum_j Z_j exp(i G.r_j); systematic extinction filtering"],
            ["Multi-Radiation", "X-ray & Relativistic Electrons", "Cu/Mo/Cr/Co Ka lines and 100-300 keV TEM electron beams"],
            ["Visualization", "Interactive 3D/2D Dashboard", "Textbook-style BS+DOS, 3D Ewald mesh, and 2D detector spot patterns"],
            ["Export & Pipeline", "Automated Persistent Storage", "Auto-saves JSON/CSV to data/processed/; 1400x800px white PNG export"]
        ],
        col_widths=[110.0, 160.0, 246.0]
    )

    # PAGE 2: PHYSICAL FORMULATIONS & MATHEMATICAL ENGINES
    doc.check_space(600.0)  # Move to next page cleanly
    doc.add_section_header("3. Mathematical Formulations & Theoretical Rigor", "[PHYSICS]")
    
    doc.add_subsection_header("A. Non-Uniform Finite-Difference Valley Curvature & Effective Mass (m*)")
    doc.add_paragraph(
        "The electronic effective mass m* reflects the dispersion curvature near local conduction band minima. Because high-symmetry "
        "k-point paths from DFT calculations often feature non-uniform path segmentation, standard uniform finite-difference approximations fail. "
        "Band Explorer implements a generalized non-uniform 3-point central finite-difference scheme around the valley index i0:"
    )
    doc.add_bullet("Left / Right Step Spacings", "h1 = k(i0) - k(i0 - step),   h2 = k(i0 + step) - k(i0)  [in Angstrom^-1]")
    doc.add_bullet("Second Derivative Curvature", "d2E/dk2 = [2 / (h1 + h2)] * [ (E(i0 + step) - E(i0))/h2 - (E(i0) - E(i0 - step))/h1 ]  [in eV*Angstrom^2]")
    doc.add_bullet("Longitudinal Mass Ratio", "m* / m0 = (hbar^2 / m0) / (d2E/dk2),   where hbar^2/m0 = 7.61996 eV*Angstrom^2")
    doc.add_bullet("Drude Electron Mobility", "mu = (e * tau) / (m* * m0) * 10^4  [in cm^2 / (V*s)],   with real-time interactive tau slider")

    doc.add_subsection_header("B. Density of States (DOS) at the Fermi Level: N(E_F)")
    doc.add_paragraph(
        "For metallic materials (e.g. Al, Cu, Fe, Ga), band gaps and semiconductor valley extrema are undefined. The engine automatically "
        "switches modes, verifying Fermi level crossings and computing the continuous density of states at E_F via numerical interpolation: "
        "N(E_F) = interp(E_F, energies, densities) [states/eV/unit cell]. This directly governs Sommerfeld electronic heat capacity (gamma_e), "
        "Pauli paramagnetic susceptibility, and metallic electrical conductivity."
    )

    doc.add_subsection_header("C. Crystallographic Reciprocal Lattice & Ewald Construction")
    doc.add_paragraph(
        "Given direct lattice vectors a1, a2, a3 with volume V = a1 . (a2 x a3), reciprocal basis vectors b1, b2, b3 are derived via the 2*pi crystallography convention:"
    )
    doc.add_bullet("Reciprocal Vectors", "b1 = 2*pi*(a2 x a3)/V,   b2 = 2*pi*(a3 x a1)/V,   b3 = 2*pi*(a1 x a2)/V  [in Angstrom^-1]")
    doc.add_bullet("Wavevector Magnitude", "k0 = 2*pi / lambda,   where lambda is radiation wavelength in Angstroms")
    doc.add_bullet("Relativistic Electron Lambda", "lambda = 12.2643 / sqrt( V_volts * (1 + 0.978466e-6 * V_volts) )  [for 100-300 keV TEM]")
    doc.add_bullet("Ewald Center & Radius", "Sphere of radius k0 centered at C = -k_i; passes through the reciprocal origin (0,0,0) by construction")
    doc.add_bullet("Geometric Laue Condition", "||k_i + G_hkl| - k0| <= epsilon = tolerance * k0, modeling crystal mosaicity and beam divergence")

    doc.add_subsection_header("D. Universal Structure Factor (F_hkl) & Systematic Absences")
    doc.add_paragraph(
        "To ensure the simulator functions universally across single-atom metals and complex semiconductors, the kinematical structure factor "
        "is evaluated analytically across all basis atoms j with fractional coordinates (x_j, y_j, z_j) and atomic numbers Z_j:"
    )
    doc.add_bullet("Structure Factor Formula", "F_hkl = Sum_j Z_j * exp[ 2*pi*i * (h*x_j + k*y_j + l*z_j) ],   Diffracted Intensity I_hkl proportional to |F_hkl|^2")
    doc.add_bullet("Systematic Absences", "Reflections with |F_hkl|^2 / max(|F|^2) < 10^-3 are classified as forbidden extinctions (e.g. Diamond Si {200} forbidden, {111}/{220} allowed; FCC unmixed parity rules).")

    # PAGE 3: SOFTWARE ARCHITECTURE & MODULAR DESIGN
    doc.check_space(600.0)
    doc.add_section_header("4. Software Engineering & Modular Architecture", "[ARCHITECTURE]")
    
    doc.add_paragraph(
        "The project follows strict clean-code and separation-of-concerns principles. Each module has an isolated responsibility and "
        "interacts via standard Python dictionaries and serialized JSON files, enabling independent unit testing and high maintainability."
    )

    doc.add_table(
        headers=["Module Name", "Primary Responsibility", "Key Functions / Algorithms"],
        rows=[
            ["fetch_data.py", "Data Ingestion & Caching", "get_or_fetch(), search_materials_by_formula(), ground-state fallback, MontyEncoder/Decoder"],
            ["analyzer.py", "Physics & Valuation Engine", "analyze_material(), extract_conduction_band(), find_conduction_valleys(), calculate_dos_at_fermi()"],
            ["ewald_sphere.py", "Diffraction Simulator", "simulate_ewald_diffraction(), build_reciprocal_lattice(), compute_structure_factor(), relativistic lambda"],
            ["visualizer.py", "Interactive Plot Generator", "generate_plotly_combined_dict(), generate_plotly_ewald_3d(), generate_plotly_diffraction_2d()"],
            ["app.py", "Web Application & REST API", "ReusableHTTPServer (auto-port fallback, address reuse), single-page dashboard, export endpoints"],
            ["data/raw/", "Offline Immutable Storage", "Raw DFT calculation files (*.json) cached once from Materials Project API"],
            ["data/processed/", "Analytical Output Data", "Automatically generated {id}_summary.json, {id}_valleys.csv, and {id}_diffraction.csv"]
        ],
        col_widths=[105.0, 165.0, 246.0]
    )

    doc.add_subsection_header("Architectural Highlights")
    doc.add_bullet("Offline-First Computing", "Downloads heavy DFT wavefunctions once; all downstream dispersion, curvature, valley, and Ewald calculations run 100% offline.")
    doc.add_bullet("Zero-Dependency Local Server", "Built on Python's native http.server with dynamic ReusableHTTPServer; includes automatic port incrementation (8050-8060) if ports are occupied.")
    doc.add_bullet("Publication-Grade PNG Export", "Generates high-contrast, pure-white-background 1400x800px PNG diagrams with deep royal blue band lines and dark typography, ready for IEEE/APS journal submission.")
    doc.add_bullet("Interactive 3D Visualizer", "Renders translucent Ewald sphere with 3 orthogonal guide rings, glowing gold/emerald Bragg nodes, and toggleable visual layers (surface, beams, lattice, extinctions).")

    # PAGE 4: READY-TO-USE RESUME CONTENT FOR CLAUDE
    doc.check_space(600.0)
    doc.add_section_header("5. Tailored Resume Bullets & Portfolio Impact for Claude", "[RESUME GUIDE]")
    
    doc.add_paragraph(
        "When prompting Claude to draft or refine your resume, use the following structured, quantified bullet points. "
        "Each bullet demonstrates strong software engineering practices, physical modeling, and measurable business/academic outcomes."
    )

    doc.add_subsection_header("Role / Project Title: Lead Developer -- Band Explorer (Materials Informatics Platform)")

    doc.add_resume_bullet(
        "Architected",
        "an end-to-end computational materials informatics web application using Python, pymatgen, and the Materials Project REST API to analyze electronic band structures and simulate 3D crystallographic diffraction for semiconductors and metals."
    )

    doc.add_resume_bullet(
        "Engineered",
        "an automated physical classification engine that differentiates metals from semiconductors, extracts VBM/CBM transitions, locates Gamma, L, Delta, X, K conduction band valleys, and computes Density of States at the Fermi level N(E_F)."
    )

    doc.add_resume_bullet(
        "Formulated",
        "a generalized non-uniform 3-point central finite-difference numerical engine to calculate conduction band curvature (d2E/dk2), extracting electron effective mass tensors (m*/m0) and estimating Drude carrier mobility with dynamic scattering time modeling."
    )

    doc.add_resume_bullet(
        "Developed",
        "a full-featured 3D Ewald Sphere and 2D diffraction simulator supporting X-ray characteristic lines (Cu, Mo, Cr, Co Ka) and relativistic de Broglie TEM electron beams (100-300 keV), computing universal kinematical structure factors F_hkl to enforce systematic extinction rules."
    )

    doc.add_resume_bullet(
        "Implemented",
        "an interactive high-performance dashboard featuring dual-mode Plotly.js visualizations, automatic offline caching with MontyEncoder/Decoder, dynamic port-fallback HTTP server, and automated CSV/JSON export pipelines."
    )

    doc.add_resume_bullet(
        "Added",
        "a 1-click publication-quality white-background export module generating high-resolution (1400x800 px) side-by-side Band Structure + DOS graphics optimized for academic publications and technical reports."
    )

    doc.add_section_header("6. Key Technical Skills & ATS Keywords", "[KEYWORDS]")
    doc.add_bullet("Programming & Languages", "Python 3.14, JavaScript (ES6+), HTML5, Tailwind CSS, SQL/JSON schema design")
    doc.add_bullet("Materials & Physics Libraries", "pymatgen, mp-api (Materials Project API), NumPy, SciPy, Matplotlib, Plotly.js, monty")
    doc.add_bullet("Domains & Core Competencies", "Computational Materials Science, Materials Informatics, Density Functional Theory (DFT), Solid-State Physics, Crystallography, Reciprocal Space, Band Structure Analysis, Ewald Sphere Construction, Carrier Mobility Modeling")
    doc.add_bullet("Software Architecture", "RESTful API Integration, Offline Caching, Finite-Difference Methods, Kinematical Scattering, Clean Architecture, High-DPI Data Visualization")

    # Save to disk
    doc.save()


if __name__ == "__main__":
    build_full_documentation()
