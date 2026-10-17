"""Render a saved Markdown audit; retain literal $$ display mathematics."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re

import markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source = args.report.read_text(encoding="utf-8")
    equations = []

    def protect(match):
        index = len(equations)
        equations.append(match.group())
        return f"\n\nDISPLAYEQUATIONTOKEN{index}END\n\n"

    protected = re.sub(r"\$\$[\s\S]*?\$\$", protect, source)
    body = markdown.markdown(protected, extensions=["tables", "fenced_code", "toc"],
                             output_format="html5")
    for i, equation in enumerate(equations):
        body = body.replace(f"<p>DISPLAYEQUATIONTOKEN{i}END</p>",
                            '<div class="equation">' + html.escape(equation) + "</div>")
    title = html.escape(source.splitlines()[0].lstrip("# "))
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TITLE</title><style>
body{margin:0;background:#f4f7fa;color:#183247;font:17px/1.8 'Microsoft YaHei',sans-serif}
main{max-width:1120px;margin:30px auto;padding:44px 56px;background:white;border-radius:12px}
h1{font-size:32px;line-height:1.45}h2{font-size:26px;border-bottom:2px solid #16818a;padding-bottom:8px;margin-top:52px}
h3{font-size:21px;margin-top:34px}a{color:#096aa4}strong{color:#133f50}
img{display:block;width:100%;height:auto;margin:24px auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65;margin:24px 0}
th{background:#edf4f8}td,th{border:1px solid #ccdae2;padding:10px 12px;text-align:left}tr:nth-child(even){background:#fafcfd}
pre,.equation{overflow-x:auto;background:#f5f8fa;padding:14px 18px;border-radius:6px;white-space:pre-wrap}
code{font-size:14px}li{margin:10px 0}.nav{font-size:14px;color:#526879}
@media(max-width:760px){main{margin:0;padding:22px 16px}body{font-size:16px}h1{font-size:26px}table{display:block;overflow-x:auto;white-space:normal}}
@media print{main{margin:0;padding:0;max-width:none}img,table{break-inside:avoid}}
</style><script>window.MathJax={tex:{displayMath:[['$$','$$']]}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
<main><p class="nav">V8机制诊断 · <a href="report.md">Markdown</a> · 图像与统计均为实际已完成实验</p>BODY</main></html>"""
    output = args.report.with_suffix(".html")
    output.write_text(page.replace("TITLE", title).replace("BODY", body), encoding="utf-8")
    print(f"Rendered {len(equations)} display equations and {len(re.findall(r'<img ', body))} figures")


if __name__ == "__main__":
    main()
