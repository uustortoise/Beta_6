#!/usr/bin/env python3
"""
Convert Markdown to styled HTML for browser viewing / PDF printing
Usage: python md_to_html.py <input.md> [output.html]
"""

import sys
import os
from pathlib import Path

def convert_md_to_html(md_path, html_path=None):
    """Convert markdown file to styled HTML."""
    
    if not os.path.exists(md_path):
        print(f"Error: File not found: {md_path}")
        return False
    
    # Default output path
    if html_path is None:
        html_path = str(Path(md_path).with_suffix('.html'))
    
    try:
        import markdown
    except ImportError:
        print("Error: markdown library not installed")
        print("Install with: python3 -m pip install markdown")
        return False
    
    # Read markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'toc']
    )
    
    # Add professional styling
    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{Path(md_path).stem}</title>
    <style>
        :root {{
            --primary-color: #1a365d;
            --secondary-color: #2c5282;
            --accent-color: #3182ce;
            --bg-color: #ffffff;
            --text-color: #333;
            --border-color: #e2e8f0;
            --code-bg: #f7fafc;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.7;
            color: var(--text-color);
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
            background: var(--bg-color);
        }}
        
        h1 {{
            font-size: 26pt;
            color: var(--primary-color);
            border-bottom: 4px solid var(--secondary-color);
            padding-bottom: 15px;
            margin-top: 0;
            margin-bottom: 30px;
        }}
        
        h2 {{
            font-size: 16pt;
            color: var(--secondary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        
        h3 {{
            font-size: 13pt;
            color: #2d3748;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        h4 {{
            font-size: 11pt;
            color: #4a5568;
            margin-top: 25px;
            margin-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 10pt;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: linear-gradient(135deg, var(--secondary-color) 0%, var(--accent-color) 100%);
            color: white;
            padding: 12px 10px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        tr:nth-child(even) {{
            background-color: #f7fafc;
        }}
        
        tr:hover {{
            background-color: #edf2f7;
        }}
        
        code {{
            background-color: var(--code-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: 9pt;
            color: #2d3748;
        }}
        
        pre {{
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.5;
            margin: 20px 0;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            color: inherit;
        }}
        
        blockquote {{
            border-left: 4px solid var(--accent-color);
            margin: 25px 0;
            padding: 15px 25px;
            background: linear-gradient(90deg, #ebf8ff 0%, #f7fafc 100%);
            border-radius: 0 8px 8px 0;
        }}
        
        ul, ol {{
            margin: 20px 0;
            padding-left: 30px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        hr {{
            border: none;
            height: 2px;
            background: linear-gradient(90deg, var(--border-color) 0%, var(--secondary-color) 50%, var(--border-color) 100%);
            margin: 40px 0;
        }}
        
        strong {{
            color: var(--primary-color);
            font-weight: 600;
        }}
        
        em {{
            color: #4a5568;
        }}
        
        a {{
            color: var(--accent-color);
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        /* Print styles */
        @media print {{
            body {{
                padding: 0;
                max-width: 100%;
            }}
            
            h1, h2, h3 {{
                page-break-after: avoid;
            }}
            
            table, pre, blockquote {{
                page-break-inside: avoid;
            }}
            
            tr {{
                page-break-inside: avoid;
            }}
        }}
        
        /* Screen styles */
        @media screen {{
            body {{
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                margin: 20px auto;
            }}
        }}
        
        /* Status indicators */
        .status-yes, .status-good, .status-check {{
            color: #38a169;
            font-weight: bold;
        }}
        
        .status-no, .status-bad, .status-error {{
            color: #e53e3e;
            font-weight: bold;
        }}
        
        .status-warning, .status-maybe {{
            color: #d69e2e;
            font-weight: bold;
        }}
    </style>
    <!-- Mermaid JS for diagrams -->
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        
        // Initialize with manual start
        mermaid.initialize({{ startOnLoad: false }});

        document.addEventListener('DOMContentLoaded', async function() {{
            const mermaidBlocks = document.querySelectorAll('code.language-mermaid');
            
            // Transform markdown code blocks to mermaid divs
            mermaidBlocks.forEach(block => {{
                const pre = block.parentElement;
                const div = document.createElement('div');
                div.className = 'mermaid';
                div.textContent = block.textContent;
                pre.replaceWith(div);
            }});
            
            // Explicitly run mermaid on the new divs
            await mermaid.run({{
                nodes: document.querySelectorAll('.mermaid')
            }});
        }});
    </script>
</head>
<body>
{html_body}
</body>
</html>"""
    
    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    print(f"✅ HTML created: {html_path}")
    print(f"   File size: {os.path.getsize(html_path) / 1024:.1f} KB")
    print(f"\n📖 To view: Open this file in your browser")
    print(f"🖨️  To save as PDF: Print → Save as PDF (Cmd+P)")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_html.py <input.md> [output.html]")
        print("Example: python md_to_html.py LOCAL_LLM_1000_POC_ANALYSIS.md")
        sys.exit(1)
    
    md_file = sys.argv[1]
    html_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_md_to_html(md_file, html_file)
    sys.exit(0 if success else 1)
