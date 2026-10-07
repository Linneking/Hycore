"""Self-contained offline HTML delivery for the HIER post-training audit."""
from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path

from .plots import _slug, save_figures, usage_transition


_MONITORING_REQUIREMENTS = [
    ("whole depth", "Training-forward whole depth and radius"),
    ("proxy depth", "Mapped proxy depth and saved tangent parameters"),
    ("projection", "Numerical projection and parameter-constraint events"),
    ("sample usage", "Actual sample hard-selected ancestor proxy IDs"),
    ("noncollision", "Noncollision and live-hinge activation definitions"),
    ("gradient activation", "Per-proxy nonzero-gradient activation"),
    ("fixed snapshots", "Identity-aligned fixed-pool geometry snapshots"),
    ("retrieval", "Raw and direction-controlled top-k object retrieval"),
    ("controlled stability", "Same-input augmentation / Gumbel stability baseline"),
    ("independent structure", "Independent geometric or semantic structure signal"),
    ("object thumbnails", "Actual point-cloud/object thumbnails"),
]


def _public_text(value):
    """Redact private filesystem / connection identifiers in free-form notices."""
    text = str(value)
    text = re.sub(r"\b[A-Za-z]:[\\/][^\s<>\"']+", "<private-path>", text)
    text = re.sub(r"/(?:mnt|home|tmp|root|Users|media|srv|opt)/[^\s<>\"']+", "<private-path>", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<private-host>", text)
    text = re.sub(r"\b(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "<private-id>", text)
    text = re.sub(r"(?:ssh|sftp)://[^\s<>]+", "<private-connection>", text)
    return text


def _escape(value):
    return html.escape(_public_text(value), quote=True)


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _format(value):
    if value is None:
        return "—"
    if _numeric(value):
        return f"{value:.5g}" if not isinstance(value, int) else str(value)
    return _escape(value)


def _availability(analysis, snapshots, figures=None):
    entries = []
    for record in analysis.get("availability", []):
        sources = record.get("source", "")
        if isinstance(sources, (list, tuple)):
            sources = "; ".join(_public_text(source) for source in sources)
        else:
            sources = _public_text(sources)
        entries.append({"metric": record.get("metric", "unknown"), "status": record.get("status", "missing"),
                        "source": sources,
                        "reason": _public_text(record.get("reason") or "")})
    keys = {str(k) for row in analysis.get("epochs", []) for k, value in row.get("metrics", {}).items() if _numeric(value)}
    usage = analysis.get("proxy_usage", [])
    config = analysis.get("identity", {}).get("config", {})
    proxy_disabled = "proxy_optimizer" in config and config.get("proxy_optimizer") in (False, None)
    no_proxy_observations = not usage and not any(k.startswith("proxy_depth") or k.startswith("proxy_radius") for k in keys)
    baseline = proxy_disabled and no_proxy_observations
    run_id = analysis.get("run_id")
    snapshot_rows = [row for row in (snapshots or {}).get("snapshots", []) if row.get("run_key", row.get("run_id")) == run_id]
    observed = {
        "whole depth": any(k.startswith("whole_depth") for k in keys),
        "proxy depth": any(k.startswith("proxy_depth") for k in keys),
        "projection": any(any(token in k for token in ("projection", "overcap", "over_cap", "cap_hit", "numerical_saturation")) for k in keys),
        "sample usage": any(r.get("component") == "sample" for r in usage),
        "noncollision": any(r.get("domain") in ("noncollision", "active_noncollision") for r in usage),
        "gradient activation": any("grad_active" in k or "gradient_active" in k for k in keys),
        "fixed snapshots": bool(snapshot_rows),
        "retrieval": any(row.get("retrieval", {}).get("available") for row in snapshot_rows),
        "controlled stability": bool(analysis.get("controlled_stability")),
        "independent structure": bool(analysis.get("independent_structure")),
        "object thumbnails": any(figure.get("kind") == "object_gallery" and figure.get("run_id") == run_id for figure in (figures or [])),
    }
    existing = {str(entry["metric"]).lower() for entry in entries}
    for name, label in _MONITORING_REQUIREMENTS:
        if name not in existing:
            not_applicable = baseline and name in ("proxy depth", "projection", "sample usage", "noncollision", "gradient activation")
            entries.append({"metric": label, "status": "not_applicable" if not_applicable else "available" if observed[name] else "missing", "source": "derived availability",
                            "reason": "Proxy optimizer explicitly disabled; no proxy observations expected in this baseline." if not_applicable else "Saved data present; see its stated definition." if observed[name] else "No matching saved field or supplied snapshot/control. Not inferred, interpolated or replaced with zero."})
    if baseline:
        for entry in entries:
            if entry["status"] == "missing" and str(entry["metric"]).startswith(("proxy_", "sample_")):
                entry.update(status="not_applicable", reason="Proxy optimizer explicitly disabled; hierarchy metrics are not applicable to this baseline.")
    return entries


def _safe_analysis(analysis):
    """Report data exports scalar observations, never raw configuration paths."""
    identity = analysis.get("identity", {})
    return {"run_id": _public_text(analysis.get("run_id", "unknown")),
            "display_name": _public_text(analysis.get("display_name", analysis.get("run_id", "unknown"))),
            "identity": {key: _public_text(identity[key]) for key in ("version", "seed", "commit", "split_sha256", "status") if key in identity},
            "metadata": {key: value for key, value in analysis.get("metadata", {}).items() if key in
                         ("warmup_epochs", "steps_per_epoch", "curvature", "n_proxy", "inherited_prefix_epochs") and isinstance(value, (str, int, float, bool, type(None)))},
            "epochs": [{key: row.get(key) for key in ("epoch", "model_updates", "logged_model_updates", "proxy_updates", "logged_proxy_updates", "phase")}
                       | {"metrics": {key: value if _numeric(value) else None for key, value in row.get("metrics", {}).items() if isinstance(value, (int, float, type(None)))}}
                       for row in analysis.get("epochs", [])]}


def render_report(analyses, snapshots, output_dir):
    """Render PNG/SVG plus a local HTML report, without network dependencies.

    Existing output files are rejected so this entry point cannot silently
    replace an earlier report. The caller owns run normalization/snapshot I/O.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "report.html").exists():
        raise FileExistsError("Report output already contains report.html; choose a new immutable audit directory.")
    rendered = save_figures(analyses, snapshots, output_dir)
    figures = rendered["figures"]
    warnings = [_public_text(w) for analysis in analyses for w in analysis.get("warnings", [])]
    warnings.extend(_public_text(w) for w in rendered.get("warnings", []))
    warnings = list(dict.fromkeys(warnings))
    table_dir = output_dir / "tables"
    table_dir.mkdir(exist_ok=True)
    table_paths, epoch_tables, availability_tables, identities = [], [], [], []
    safe_data = [_safe_analysis(a) for a in analyses]
    for analysis, public in zip(analyses, safe_data):
        run_id, display = public["run_id"], public["display_name"]
        metric_keys = sorted({key for row in public["epochs"] for key in row.get("metrics", {})})
        table_file = table_dir / (_slug(run_id) + "_epochs.csv")
        with table_file.open("x", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "model_updates", "logged_model_updates", "proxy_updates", "phase"] + metric_keys)
            writer.writeheader()
            for row in public["epochs"]:
                writer.writerow({key: row.get(key) for key in ("epoch", "model_updates", "logged_model_updates", "proxy_updates", "phase")} | row.get("metrics", {}))
        table_paths.append(str(table_file))
        temporal = []
        usage_groups = {}
        for item in analysis.get("proxy_usage", []):
            if item.get("role") == "combined":
                usage_groups.setdefault((item.get("component"), item.get("domain")), []).append(item)
        for (component, domain), records in sorted(usage_groups.items()):
            records.sort(key=lambda r: r.get("epoch", 0))
            for previous, current in zip(records, records[1:]):
                result = usage_transition(previous, current)
                temporal.append({"component": component, "domain": domain,
                                 "from_epoch": previous.get("epoch"), "to_epoch": current.get("epoch"),
                                 "epoch_span": current.get("epoch", 0) - previous.get("epoch", 0), **result})
        temporal_link = ""
        if temporal:
            temporal_file = table_dir / (_slug(run_id) + "_usage_transitions.csv")
            fieldnames = ["component", "domain", "from_epoch", "to_epoch", "epoch_span", "available", "reason", "jsd_bits", "rank_spearman", "used_set_jaccard", "newly_used_count", "became_inactive_count", "previous_used_count", "current_used_count"]
            with temporal_file.open("x", encoding="utf8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(temporal)
            table_paths.append(str(temporal_file))
            temporal_link = f' · <a href="tables/{html.escape(temporal_file.name)}">Within-run proxy usage transitions (CSV)</a>'
        primary = [key for key in ("val_oa_pct", "clean_train_oa_pct", "loss", "whole_depth_median", "whole_depth_mean", "proxy_depth_median", "sample_used_proxy_count", "sample_effective_proxy_count", "proxy_tangent_norm_max") if key in metric_keys]
        heads = ["Epoch", "Model updates", "Phase"] + primary
        rows = []
        for row in public["epochs"]:
            cells = [_format(row.get("epoch")), _format(row.get("model_updates")), _format(row.get("phase"))]
            cells.extend(_format(row.get("metrics", {}).get(key)) for key in primary)
            rows.append(f'<tr data-epoch="{_escape(row.get("epoch", ""))}">' + "".join("<td>" + cell + "</td>" for cell in cells) + "</tr>")
        epoch_tables.append(f'<section class="run-section" data-run="{_escape(run_id)}"><h3>{_escape(display)}: saved epoch table</h3><p><a href="tables/{html.escape(table_file.name)}">All scalar fields (CSV)</a>{temporal_link}</p><div class="table-scroll"><table><thead><tr>' + "".join("<th>" + _escape(h) + "</th>" for h in heads) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>")
        entries = _availability(analysis, snapshots, figures)
        availability_tables.append(f'<section class="run-section" data-run="{_escape(run_id)}"><h3>{_escape(display)}: availability</h3><div class="table-scroll"><table><thead><tr><th>Observation</th><th>Status</th><th>Source</th><th>Reason / limits</th></tr></thead><tbody>' + "".join(f'<tr><td>{_escape(e["metric"])}</td><td class="{_escape(e["status"])}">{_escape(e["status"])}</td><td>{_escape(e["source"])}</td><td>{_escape(e["reason"])}</td></tr>' for e in entries) + "</tbody></table></div></section>")
        identity_text = "; ".join(key + "=" + str(value) for key, value in public["identity"].items()) or "Identity metadata missing"
        metadata_text = "; ".join(key + "=" + str(value) for key, value in public["metadata"].items())
        identities.append(f'<li data-run="{_escape(run_id)}"><strong>{_escape(display)}</strong>: {_escape(identity_text)}<br>{_escape(metadata_text)}</li>')
    data_path = output_dir / "report_data.json"
    data_path.write_text(json.dumps({"schema_version": 1, "runs": safe_data, "warnings": warnings}, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf8")
    figure_blocks = []
    for figure in figures:
        figure_blocks.append(f'<figure class="run-section" data-run="{_escape(figure["run_id"])}"><h3>{_escape(figure["title"])}</h3><a class="image-link" href="{html.escape(figure["png"])}" target="_blank" rel="noopener"><img loading="lazy" src="{html.escape(figure["png"])}" alt="{_escape(figure["title"])}"></a><figcaption>{_escape(figure["note"])} <a href="{html.escape(figure["svg"])}">Vector SVG</a></figcaption></figure>')
    object_tables = []
    for snapshot in (snapshots or {}).get("snapshots", []):
        run_id = snapshot.get("run_key", snapshot.get("run_id", "snapshot"))
        retrieval = snapshot.get("retrieval", {})
        for method in ("hyperbolic", "direction"):
            method_data = retrieval.get(method, {})
            topk_ids = method_data.get("topk_sample_ids", [])
            if not topk_ids:
                continue
            proxy_ids = retrieval.get("proxy_ids", [])
            identity_label = "Proxy ID" if snapshot.get("proxy", {}).get("proxy_ids_stable") else "Proxy row (identity unverified)"
            if len(proxy_ids) != len(topk_ids) and not isinstance(topk_ids, dict):
                proxy_ids = list(range(len(topk_ids)))
            if isinstance(topk_ids, dict):
                records = list(topk_ids.items())
            else:
                records = list(zip(proxy_ids, topk_ids))
            # The report is a readable preview; full relation IDs remain in the
            # snapshot artifact rather than a fabricated point-cloud image.
            records = records[:32]
            cells = "".join("<tr><td>" + _escape(proxy_id) + "</td><td>" + _escape(", ".join(map(str, ids))) + "</td></tr>" for proxy_id, ids in records)
            object_tables.append(f'<section class="run-section" data-run="{_escape(run_id)}"><h3>e{_escape(snapshot.get("epoch"))} {method}: actual object IDs</h3><p>First 32 proxy rows. Object images are not reconstructed from embedding coordinates.</p><div class="table-scroll"><table><thead><tr><th>{identity_label}</th><th>Nearest object IDs</th></tr></thead><tbody>{cells}</tbody></table></div></section>')
    options = '<option value="all">All runs</option>' + "".join(f'<option value="{_escape(a["run_id"])}">{_escape(a["display_name"])}</option>' for a in safe_data)
    warning_html = "<ul>" + "".join("<li>" + _escape(w) + "</li>" for w in warnings) + "</ul>" if warnings else "<p>No additional adapter warnings. Availability limits still apply.</p>"
    incomplete = [a["display_name"] for a in safe_data if not str(a["identity"].get("status", "unknown")).lower().startswith("completed")]
    incomplete_notice = ('<div class="limit"><strong>INCOMPLETE / COMPLETENESS UNVERIFIED:</strong> ' + _escape(", ".join(incomplete)) + '. This report describes the saved prefix only; it does not assert completed training.</div>') if incomplete else ""
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HIER post-training audit</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#182033;background:#f3f5f8;margin:0}main{max-width:1160px;margin:auto;padding:24px}h1{font-size:27px}h2{font-size:22px;margin-top:30px}h3{font-size:17px}p{max-width:1000px}.controls{position:sticky;top:0;background:#fff;z-index:2;display:flex;gap:14px;flex-wrap:wrap;padding:13px;border:1px solid #d8deea;border-radius:8px}.controls input{width:85px}.controls select,.controls input{padding:5px}.run-section,figure{background:#fff;border:1px solid #d8deea;border-radius:8px;margin:14px 0;padding:16px}figure img{display:block;width:100%;height:auto}figcaption{color:#445269;font-size:13px}.table-scroll{overflow:auto;max-height:560px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #e1e6ef;vertical-align:top}th{background:#f8fafc;white-space:nowrap;position:sticky;top:0}.available{color:#087f5b}.partial{color:#9a6700}.missing{color:#b42318}a{color:#1759b8}.limit{padding:14px;background:#fff8e5;border-left:4px solid #c78b17}[hidden]{display:none!important}@media(max-width:600px){main{padding:12px}.run-section,figure{padding:9px}h1{font-size:23px}}
</style></head><body><main><h1>HIER post-training audit</h1><p>Saved evidence across training versions, with explicit identity, definitions and missing-data limits. This report is fully offline. Click any figure to view its original PNG; SVG files support publication-quality reuse.</p><div class="controls"><label>Run <select id="run">__OPTIONS__</select></label><label>Epoch from <input id="epoch-min" type="number" placeholder="All"></label><label>to <input id="epoch-max" type="number" placeholder="All"></label><span>Epoch filter applies to table rows; plots preserve the full saved timeline.</span></div>
__INCOMPLETE__<h2>Identity and interpretation</h2><ul>__IDENTITIES__</ul><div class="limit">Training-forward sampled geometry and frozen clean/eval fixed-pool geometry are separate observations. Hard-selected ancestors, noncollision ancestors, live-hinge proxies and gradient-active proxies are separate activation definitions. Nearest top-k objects are retrieval, not HIER training descendants. Cross-version comparisons require matching pools, update counts and protocol; stochastic usage changes need same-input noise controls before a mechanistic claim. No low-dimensional projection is represented as the original hyperbolic geometry.</div>
<h2>Data availability</h2>__AVAILABILITY__<h2>Observed trajectories</h2>__FIGURES__<h2>Saved epoch records</h2>__EPOCH_TABLES__<h2>Object relation previews</h2>__OBJECT_TABLES__<h2>Warnings</h2>__WARNINGS__<p><a href="report_data.json">Public scalar data JSON</a>. Empty table cells mean unavailable, never zero. Existing reports are not overwritten.</p></main><script>
(function(){const run=document.getElementById('run'), lo=document.getElementById('epoch-min'), hi=document.getElementById('epoch-max');function update(){const id=run.value,min=lo.value===''?-Infinity:Number(lo.value),max=hi.value===''?Infinity:Number(hi.value);document.querySelectorAll('[data-run]').forEach(el=>{el.hidden=id!=='all'&&el.dataset.run!==id});document.querySelectorAll('tr[data-epoch]').forEach(el=>{const e=Number(el.dataset.epoch);el.hidden=e<min||e>max})}run.addEventListener('change',update);lo.addEventListener('input',update);hi.addEventListener('input',update);update()})();
</script></body></html>"""
    replacements = {"__OPTIONS__": options, "__INCOMPLETE__": incomplete_notice, "__IDENTITIES__": "".join(identities), "__AVAILABILITY__": "".join(availability_tables),
                    "__FIGURES__": "".join(figure_blocks) or "<p>No eligible saved figure series. See availability reasons.</p>",
                    "__EPOCH_TABLES__": "".join(epoch_tables), "__OBJECT_TABLES__": "".join(object_tables) or "<p>No supplied retrieval snapshots or object thumbnails; none were fabricated.</p>", "__WARNINGS__": warning_html}
    for key, value in replacements.items():
        document = document.replace(key, value)
    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf8")
    return {"report_path": str(report_path), "figures": figures, "table_paths": table_paths, "warnings": warnings,
            "paths": {"report": str(report_path), "data": str(data_path), "tables": table_paths,
                      "figures": [str(output_dir / figure["png"]) for figure in figures]}}
