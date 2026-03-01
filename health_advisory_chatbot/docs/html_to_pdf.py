#!/usr/bin/env python3
"""
Convert HTML to PDF using Playwright (headless Chrome)
Usage: python html_to_pdf.py <input.html> [output.pdf]
"""

import sys
import os
import asyncio
from pathlib import Path

async def convert_html_to_pdf(html_path, pdf_path=None):
    """Convert HTML file to PDF using Playwright."""
    
    if not os.path.exists(html_path):
        print(f"Error: File not found: {html_path}")
        return False
    
    # Default output path
    if pdf_path is None:
        pdf_path = str(Path(html_path).with_suffix('.pdf'))
    
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Error: playwright not installed")
        print("Install with: python3 -m pip install playwright")
        print("Then: python3 -m playwright install chromium")
        return False
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load HTML file
        file_url = f"file://{os.path.abspath(html_path)}"
        await page.goto(file_url, wait_until="networkidle")
        
        # Generate PDF
        await page.pdf(
            path=pdf_path,
            format="A4",
            margin={
                "top": "20mm",
                "right": "15mm", 
                "bottom": "20mm",
                "left": "15mm"
            },
            print_background=True
        )
        
        await browser.close()
    
    print(f"✅ PDF created: {pdf_path}")
    print(f"   File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python html_to_pdf.py <input.html> [output.pdf]")
        print("Example: python html_to_pdf.py LOCAL_LLM_1000_POC_ANALYSIS.html")
        sys.exit(1)
    
    html_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = asyncio.run(convert_html_to_pdf(html_file, pdf_file))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
