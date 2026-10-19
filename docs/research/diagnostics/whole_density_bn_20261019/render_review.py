"""Render report58 Markdown with protected display math and validate local links."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_dir.resolve()
    for svg in (root / "figures").glob("*.svg"):
        content = svg.read_text(encoding="utf-8")
        normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
        if normalized != content:
            svg.write_text(normalized, encoding="utf-8", newline="\n")
    source = root / "report.md"
    text = source.read_text(encoding="utf-8-sig")
    equations = []

    def protect(match):
        equations.append(match.group(1))
        return "\n\nMATHPLACEHOLDER%04d\n\n" % (len(equations) - 1)

    protected = re.sub(r"(?ms)^\$\$\s*\n(.*?)^\$\$\s*$", protect, text)
    body = markdown.markdown(protected, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    for index, equation in enumerate(equations):
        token = "<p>MATHPLACEHOLDER%04d</p>" % index
        if token not in body:
            raise ValueError("Display-math token was not preserved")
        body = body.replace(token, '<div class="equation">$$\n' + html.escape(equation.strip()) + '\n$$</div>')
    title = text.splitlines()[0].lstrip("# ").strip()
    template = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{margin:0;background:#f4f7fa;color:#183247;font:17px/1.85 "Microsoft YaHei",sans-serif}
main{max-width:1120px;margin:28px auto;padding:44px 54px;background:white;border-radius:12px}
h1{font-size:32px;line-height:1.4}h2{font-size:25px;margin-top:48px;border-bottom:2px solid #16818a;padding-bottom:7px}
h3{font-size:20px;margin-top:30px}a{color:#096aa4}strong{color:#133f50}img{display:block;width:100%;height:auto;margin:24px auto}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65;margin:22px 0}
th{background:#edf4f8}td,th{border:1px solid #ccdae2;padding:9px 11px;text-align:left}tr:nth-child(even){background:#fafcfd}
.equation,pre{overflow-x:auto;background:#f5f8fa;padding:14px 18px;border-radius:6px}.nav{font-size:14px;color:#526879}
blockquote{border-left:4px solid #16818a;padding-left:18px;margin-left:0}li{margin:7px 0}
@media(max-width:760px){main{margin:0;padding:22px 16px}body{font-size:16px}h1{font-size:25px}table{display:block;overflow-x:auto}}
</style><script>window.MathJax={tex:{displayMath:[['$$','$$']]}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
<main><p class="nav">冻结训练集审查 · <a href="report.md">Markdown互动版</a> · <a href="tables/whole_depth_distribution.csv">完整密度统计</a></p>
__BODY__</main></html>"""
    (root / "report.html").write_text(template.replace("__TITLE__", html.escape(title)).replace("__BODY__", body), encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    local, missing = [], []
    for value in links:
        if re.match(r"^[a-zA-Z]+://", value) or value.startswith("#"):
            continue
        path = (root / value.split("#", 1)[0]).resolve()
        local.append(value)
        if not path.exists() and path != root / "validation.json":
            missing.append(value)
    if missing:
        raise FileNotFoundError("Missing local report targets: " + repr(missing))
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|(?i:[a-z]:\\Users\\)|/mnt/[^\s]+", text):
        raise ValueError("Private runtime path leaked into report")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "validation.json":
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    validation = dict(passed=True, local_links=len(local), missing_links=missing,
                      display_math_blocks=len(equations), scientific_png=len(list((root / "figures").glob("*.png"))),
                      scientific_svg=len(list((root / "figures").glob("*.svg"))),
                      no_private_runtime_paths_in_report=True, source_markdown_unchanged=True,
                      files_sha256=files)
    (root / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in validation.items() if key != "files_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
