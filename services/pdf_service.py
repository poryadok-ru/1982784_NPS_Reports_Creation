import markdown
import pdfkit


def md_to_pdf(md_text: str, pdf_path: str) -> None:
    """Сконвертировать markdown-отчет в PDF."""
    html_body = markdown.markdown(md_text, extensions=['fenced_code', 'tables'])
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, Verdana, Helvetica, sans-serif; margin: 2em; font-size: 14px; }}
        h1, h2, h3, h4 {{ color: #1a588b; }}
        code, pre {{ background: #f8f8f8; border-radius: 4px; padding: 5px 8px; font-size:13px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ccc; padding: 6px; }}
        th {{ background-color: #f0f0f5; }}
        ul, ol {{ margin-left: 1.5em; }}
        blockquote {{ border-left: 4px solid #ccc; margin: 1em; padding-left: 1em; color: #555; font-style: italic; background: #f9f9fd; }}
    </style>
    </head>
    <body>
    {html_body}
    </body>
    </html>
    """
    try:
        config = pdfkit.configuration()
    except Exception:
        config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
    pdfkit.from_string(html, pdf_path, configuration=config)