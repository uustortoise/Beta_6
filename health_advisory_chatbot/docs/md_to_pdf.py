#!/usr/bin/env python3
"""
Convert Markdown to PDF with nice formatting
Usage: python md_to_pdf.py <input.md> [output.pdf]
"""

import sys
import os
from pathlib import Path

def convert_md_to_pdf(md_path, pdf_path=None):
    """Convert markdown file to PDF using WeasyPrint."""
    
    if not os.path.exists(md_path):
        print(f"Error: File not found: {md_path}")
        return False
    
    # Default output path
    if pdf_path is None:
        pdf_path = str(Path(md_path).with_suffix('.pdf'))
    
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: python3 -m pip install markdown weasyprint")
        return False
    
    # Read markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'toc', 'nl2br']
    )
    
    # Add professional styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{Path(md_path).stem}</title>
        <style>
            @page {{
                size: A4;
                margin: 2.5cm 2cm;
                @bottom-center {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 9pt;
                    color: #666;
                }}
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}
            
            h1 {{
                font-size: 24pt;
                color: #1a365d;
                border-bottom: 3px solid #2c5282;
                padding-bottom: 10px;
                margin-top: 0;
                page-break-after: avoid;
            }}
            
            h2 {{
                font-size: 16pt;
                color: #2c5282;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 8px;
                margin-top: 30px;
                page-break-after: avoid;
            }}
            
            h3 {{
                font-size: 13pt;
                color: #2d3748;
                margin-top: 25px;
                page-break-after: avoid;
            }}
            
            h4 {{
                font-size: 11pt;
                color: #4a5568;
                margin-top: 20px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 10pt;
                page-break-inside: avoid;
            }}
            
            th {{
                background-color: #2c5282;
                color: white;
                padding: 10px 8px;
                text-align: left;
                font-weight: 600;
            }}
            
            td {{
                padding: 8px;
                border-bottom: 1px solid #e2e8f0;
            }}
            
            tr:nth-child(even) {{
                background-color: #f7fafc;
            }}
            
            code {{
                background-color: #edf2f7;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 9pt;
            }}
            
            pre {{
                background-color: #2d3748;
                color: #e2e8f0;
                padding: 15px;
                border-radius: 6px;
                overflow-x: auto;
                font-size: 9pt;
                line-height: 1.4;
                page-break-inside: avoid;
            }}
            
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            
            blockquote {{
                border-left: 4px solid #2c5282;
                margin: 20px 0;
                padding: 10px 20px;
                background-color: #f7fafc;
                font-style: italic;
            }}
            
            ul, ol {{
                margin: 15px 0;
                padding-left: 25px;
            }}
            
            li {{
                margin: 5px 0;
            }}
            
            hr {{
                border: none;
                border-top: 2px solid #e2e8f0;
                margin: 30px 0;
            }}
            
            .toc {{
                background-color: #f7fafc;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            
            .toc ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            
            .toc li {{
                margin: 8px 0;
            }}
            
            .toc a {{
                color: #2c5282;
                text-decoration: none;
            }}
            
            strong {{
                color: #2d3748;
            }}
            
            em {{
                color: #4a5568;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert to PDF
    font_config = FontConfiguration()
    HTML(string=styled_html).write_pdf(
        pdf_path,
        font_config=font_config
    )
    
    print(f"✅ PDF created: {pdf_path}")
    print(f"   File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_pdf.py <input.md> [output.pdf]")
        print("Example: python md_to_pdf.py LOCAL_LLM_1000_POC_ANALYSIS.md")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_md_to_pdf(md_file, pdf_file)
    sys.exit(0 if success else 1)
