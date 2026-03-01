#!/usr/bin/env python3
"""
Convert Markdown to PDF using fpdf2 (pure Python, no system dependencies)
Usage: python md_to_pdf_simple.py <input.md> [output.pdf]
"""

import sys
import os
import re
from pathlib import Path
from fpdf import FPDF


class MarkdownPDF(FPDF):
    """Custom PDF class for rendering Markdown content."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        
    def footer(self):
        """Add footer with page number."""
        self.set_y(-15)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')


def strip_markdown(text):
    """Remove markdown syntax for plain text rendering."""
    if not text:
        return ""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '[Code Block]', text)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove bold/italic markers
    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'___([^_]+)___', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', text)
    return text


def parse_markdown(md_content):
    """Parse markdown into structured elements."""
    lines = md_content.split('\n')
    elements = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.startswith('```'):
            lang = line[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            elements.append(('code', '\n'.join(code_lines), lang))
            i += 1
            continue
            
        # Tables
        if '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
            table_lines = [line]
            i += 1
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            elements.append(('table', table_lines))
            continue
            
        # Headers
        if line.startswith('# '):
            elements.append(('h1', line[2:]))
        elif line.startswith('## '):
            elements.append(('h2', line[3:]))
        elif line.startswith('### '):
            elements.append(('h3', line[4:]))
        elif line.startswith('#### '):
            elements.append(('h4', line[5:]))
        # Horizontal rule
        elif line.strip() == '---' or line.strip() == '***':
            elements.append(('hr', ''))
        # Blockquote
        elif line.startswith('> '):
            quote_lines = [line[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith('> '):
                quote_lines.append(lines[i][2:])
                i += 1
            elements.append(('blockquote', '\n'.join(quote_lines)))
            continue
        # List items
        elif re.match(r'^[\s]*[-*+]\s', line):
            elements.append(('ul_item', re.sub(r'^[\s]*[-*+]\s', '', line)))
        elif re.match(r'^[\s]*\d+\.\s', line):
            elements.append(('ol_item', re.sub(r'^[\s]*\d+\.\s', '', line)))
        # Empty line
        elif line.strip() == '':
            elements.append(('empty', ''))
        # Regular paragraph
        else:
            elements.append(('paragraph', line))
            
        i += 1
    
    return elements


def convert_md_to_pdf(md_path, pdf_path=None):
    """Convert markdown file to PDF."""
    
    if not os.path.exists(md_path):
        print(f"Error: File not found: {md_path}")
        return False
    
    # Default output path
    if pdf_path is None:
        pdf_path = str(Path(md_path).with_suffix('.pdf'))
    
    # Read markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Parse markdown
    elements = parse_markdown(md_content)
    
    # Create PDF
    pdf = MarkdownPDF()
    pdf.add_page()
    
    # Track list state
    in_ul = False
    in_ol = False
    ol_number = 1
    
    for elem in elements:
        elem_type = elem[0]
        content = elem[1] if len(elem) > 1 else ""
        
        if elem_type == 'h1':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            pdf.set_font('Helvetica', 'B', 20)
            pdf.set_text_color(26, 54, 93)
            text = strip_markdown(content)
            pdf.cell(0, 12, text, new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(44, 82, 130)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(8)
            
        elif elem_type == 'h2':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(44, 82, 130)
            text = strip_markdown(content)
            pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(226, 232, 240)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(6)
            
        elif elem_type == 'h3':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(45, 55, 72)
            text = strip_markdown(content)
            pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            
        elif elem_type == 'h4':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(74, 85, 104)
            text = strip_markdown(content)
            pdf.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            
        elif elem_type == 'paragraph':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(51, 51, 51)
            text = strip_markdown(content)
            pdf.multi_cell(0, 6, text)
            pdf.ln(3)
            
        elif elem_type == 'code':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            pdf.set_fill_color(45, 55, 72)
            pdf.set_text_color(226, 232, 240)
            pdf.set_font('Courier', '', 8)
            pdf.cell(0, 6, '[Code Block]', new_x="LMARGIN", new_y="NEXT")
            code_text = content[:500] + '...' if len(content) > 500 else content
            pdf.multi_cell(0, 5, code_text, fill=True)
            pdf.set_text_color(51, 51, 51)
            pdf.ln(5)
            
        elif elem_type == 'blockquote':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            pdf.set_fill_color(235, 248, 255)
            pdf.set_draw_color(49, 130, 206)
            pdf.set_line_width(0.5)
            pdf.set_font('Helvetica', 'I', 10)
            pdf.set_text_color(74, 85, 104)
            pdf.line(18, pdf.get_y(), 18, pdf.get_y() + 20)
            pdf.cell(3)
            text = strip_markdown(content)
            pdf.multi_cell(0, 6, text, fill=True)
            pdf.ln(5)
            
        elif elem_type == 'ul_item':
            if not in_ul:
                in_ul = True
                pdf.ln(2)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(51, 51, 51)
            pdf.cell(5)
            pdf.cell(5, 6, chr(149))
            text = strip_markdown(content)
            pdf.multi_cell(0, 6, text)
            
        elif elem_type == 'ol_item':
            if not in_ol:
                in_ol = True
                ol_number = 1
                pdf.ln(2)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(51, 51, 51)
            pdf.cell(5)
            pdf.cell(8, 6, f'{ol_number}.')
            text = strip_markdown(content)
            pdf.multi_cell(0, 6, text)
            ol_number += 1
            
        elif elem_type == 'table':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            render_table(pdf, content)
            pdf.ln(5)
            
        elif elem_type == 'hr':
            if in_ul or in_ol:
                pdf.ln(5)
                in_ul = in_ol = False
            pdf.set_draw_color(226, 232, 240)
            pdf.line(15, pdf.get_y() + 5, 195, pdf.get_y() + 5)
            pdf.ln(10)
            
        elif elem_type == 'empty':
            pass
    
    # Save PDF
    pdf.output(pdf_path)
    
    print(f"✅ PDF created: {pdf_path}")
    print(f"   File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    print(f"   Pages: {pdf.page_no()}")
    return True


def render_table(pdf, table_lines):
    """Render a markdown table in PDF."""
    if len(table_lines) < 2:
        return
    
    # Parse table
    headers = [cell.strip() for cell in table_lines[0].split('|') if cell.strip()]
    rows = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if cells:
            rows.append(cells)
    
    if not headers:
        return
    
    # Calculate column widths
    col_count = len(headers)
    col_width = 180 / col_count
    
    # Header row
    pdf.set_fill_color(44, 82, 130)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_line_width(0.3)
    
    for header in headers:
        text = strip_markdown(header[:20])
        pdf.cell(col_width, 8, text, border=1, fill=True)
    pdf.ln()
    
    # Data rows
    pdf.set_fill_color(247, 250, 252)
    pdf.set_text_color(51, 51, 51)
    pdf.set_font('Helvetica', '', 8)
    
    fill = False
    for row in rows[:20]:
        if len(row) < col_count:
            row.extend([''] * (col_count - len(row)))
        for i, cell in enumerate(row[:col_count]):
            text = strip_markdown(cell[:25])
            pdf.cell(col_width, 7, text, border=1, fill=fill)
        pdf.ln()
        fill = not fill


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_pdf_simple.py <input.md> [output.pdf]")
        print("Example: python md_to_pdf_simple.py LOCAL_LLM_1000_POC_ANALYSIS.md")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_md_to_pdf(md_file, pdf_file)
    sys.exit(0 if success else 1)
