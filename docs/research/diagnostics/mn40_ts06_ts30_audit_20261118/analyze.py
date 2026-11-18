"""Join actual object exposures and frozen predictions; no causal attribution."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from audit import LATE, NAMES, load_json, write


def csv_file(path, rows):
    rows = list(rows)
    if not rows: return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def group(sh, cb):
    return np.where(sh, np.where(cb, "both_correct", "sh_only"), np.where(cb, "cb_only", "both_wrong"))


def margin(logits, y):
    z = logits.copy(); z[np.arange(len(y)), y] = -np.inf
    return logits[np.arange(len(y)), y] - z.max(1)


def corr(x, y):
    return None if np.std(x) == 0 or np.std(y) == 0 else float(spearmanr(x, y).statistic)


def arrays(path):
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    meta = load_json(args.inputs / "manifest.json"); identity = arrays(args.inputs / "identity.npz")
    sampling = {a: arrays(args.inputs / f"{a}_sampling.npz") for a in ("TS06", "TS30", "TS31")}
    assert load_json(args.predictions / "manifest.json")["status"] == "completed"
    ytrain = identity["train_labels"]; nclass = np.bincount(ytrain, minlength=40)
    results, groups = {}, {}

    def pred(arm, tag, split):
        z = arrays(args.predictions / f"{arm}_{tag}_{split}.npz")
        assert np.array_equal(z["ids"], identity[f"{split}_ids"])
        assert np.array_equal(z["labels"], identity[f"{split}_labels"])
        return z

    late = {}
    ytest = identity["test_labels"]
    for a in sampling:
        late[a] = np.stack([pred(a, f"e{e}", "test")["predictions"] for e in LATE])
    sc = late["TS06"] == ytest; cc = late["TS30"] == ytest; bc = late["TS31"] == ytest
    sh_count = (sc & ~cc).sum(0); cb_count = (~sc & cc).sum(0)
    stable_sh = sh_count >= 5; stable_cb = cb_count >= 5
    stable_wrong = (~sc & ~cc).sum(0) >= 5
    np.savez_compressed(args.out / "stability.npz", ids=identity["test_ids"],
        late_epochs=LATE, TS06=late["TS06"], TS30=late["TS30"], TS31=late["TS31"],
        sh_only_count=sh_count, cb_only_count=cb_count, persistent_sh=stable_sh,
        persistent_cb=stable_cb, persistent_both_wrong=stable_wrong)
    csv_file(args.out / "test_stability.csv", (
        dict(id=str(identity["test_ids"][i]), label=NAMES[int(ytest[i])], sh_only_epochs=int(sh_count[i]),
             cb_only_epochs=int(cb_count[i]), persistent_sh=bool(stable_sh[i]), persistent_cb=bool(stable_cb[i]),
             persistent_both_wrong=bool(stable_wrong[i]), cb31_correct_epochs=int(bc[:, i].sum()),
             sh_predictions="|".join(NAMES[int(x)] for x in late["TS06"][:, i]),
             cb_predictions="|".join(NAMES[int(x)] for x in late["TS30"][:, i])) for i in range(len(ytest))))

    train_rows = []
    for i, identifier in enumerate(identity["train_ids"]):
        row = dict(id=str(identifier), label=NAMES[int(ytrain[i])])
        for a, s in sampling.items():
            row.update({f"{a}_best_count": int(s["best_counts"][i]), f"{a}_final_count": int(s["final_counts"][i]),
                        f"{a}_seen_epochs": int(s["seen_epochs"][i]), f"{a}_longest_absence": int(s["longest_absence"][i]),
                        f"{a}_final_sameclass_partners": int(s["distinct_same_class_peers"][i])})
            for j in range(3): row[f"{a}_e{j*100+1}_{(j+1)*100}"] = int(s["epoch_counts"][j*100:(j+1)*100, i].sum())
        train_rows.append(row)
    csv_file(args.out / "train_exposures.csv", train_rows)

    for tag, count_key in (("best", "best_counts"), ("e300", "final_counts")):
        for split in ("test", "train"):
            z = {a: pred(a, tag, split) for a in sampling}
            y = identity[f"{split}_labels"]; ids = identity[f"{split}_ids"]
            good = {a: v["predictions"] == y for a, v in z.items()}
            g = group(good["TS06"], good["TS30"]); groups[(tag, split)] = g
            counts = {name: int((g == name).sum()) for name in np.unique(g)}
            allwrong = ~good["TS06"] & ~good["TS30"] & ~good["TS31"]
            counts["all_three_h0_wrong"] = int(allwrong.sum())
            counts["sh_correct"] = int(good["TS06"].sum()); counts["cb_correct"] = int(good["TS30"].sum())
            assert counts.get("sh_only", 0) - counts.get("cb_only", 0) == counts["sh_correct"] - counts["cb_correct"]
            if tag == "best" and split == "test": assert counts["sh_correct"] - counts["cb_correct"] == 12
            results[f"{tag}_{split}"] = counts
            rows = []
            margins = {a: margin(v["logits"], y) for a,v in z.items()}
            for i in range(len(y)):
                row = dict(id=str(ids[i]), label=NAMES[int(y[i])], group=str(g[i]), all_three_wrong=bool(allwrong[i]))
                for a in sampling:
                    row[f"{a}_prediction"] = NAMES[int(z[a]["predictions"][i])]
                    row[f"{a}_truth_margin"] = float(margins[a][i])
                    if split == "train": row[f"{a}_exposure"] = int(sampling[a][count_key][i])
                if split == "test":
                    row.update(sh_only_late=int(sh_count[i]), cb_only_late=int(cb_count[i]),
                               persistent_sh=bool(stable_sh[i]), persistent_cb=bool(stable_cb[i]))
                rows.append(row)
            csv_file(args.out / f"{tag}_{split}_objects.csv", rows)
            for name in ("sh_only", "cb_only", "both_wrong", "both_correct"):
                csv_file(args.out / f"{tag}_{split}_{name}.csv", (row for row in rows if row["group"] == name))
            csv_file(args.out / f"{tag}_{split}_all_three_wrong.csv", (row for row in rows if row["all_three_wrong"]))

        g = groups[(tag, "test")]; nt = np.bincount(ytest, minlength=40)
        scount, ccount = sampling["TS06"][count_key], sampling["TS30"][count_key]
        per_class = []
        for c, name in enumerate(NAMES):
            mask = ytest == c; tr = ytrain == c
            shn, cbn = int(((g == "sh_only") & mask).sum()), int(((g == "cb_only") & mask).sum())
            row = dict(class_id=c, name=name, train_size=int(nclass[c]), test_size=int(nt[c]),
                sh_draws=int(scount[tr].sum()), cb_draws=int(ccount[tr].sum()),
                sh_per_train_object=float(scount[tr].mean()), cb_per_train_object=float(ccount[tr].mean()),
                cb_to_sh_exposure_ratio=float(ccount[tr].sum()/scount[tr].sum()),
                sh_only=shn, cb_only=cbn, net_sh_advantage=shn-cbn, delta_oa_class=(shn-cbn)/nt[c]*100,
                persistent_sh=int((stable_sh & mask).sum()), persistent_cb=int((stable_cb & mask).sum()))
            per_class.append(row)
        csv_file(args.out / f"{tag}_classes.csv", per_class)
        x = np.log2([r["cb_to_sh_exposure_ratio"] for r in per_class]); y = np.array([r["delta_oa_class"] for r in per_class])
        rho = corr(x, y); rng = np.random.default_rng(4780)
        boot = []
        for _ in range(4000):
            ix = rng.integers(0, 40, 40); value = corr(x[ix], y[ix])
            if value is not None: boot.append(value)
        results[f"{tag}_class_association"] = dict(spearman=rho, bootstrap_class_ci95=np.quantile(boot,[.025,.975]).tolist(),
            high_loss_classes=sorted(per_class, key=lambda r:-r["net_sh_advantage"])[:10],
            high_gain_classes=sorted(per_class, key=lambda r:r["net_sh_advantage"])[:6])

        # Within-class count differences remove the deterministic class-prior shift.
        cz = np.zeros(9840)
        for c in range(40):
            mask = ytrain == c; values = ccount[mask]
            cz[mask] = (values-values.mean())/max(values.std(),1e-12)
        results[f"{tag}_within_train"] = {name:dict(n=int((groups[(tag,"train")]==name).sum()),
            cb_exposure_z_mean=float(cz[groups[(tag,"train")]==name].mean()) if np.any(groups[(tag,"train")]==name) else None)
            for name in ("both_correct","sh_only","cb_only","both_wrong")}
        results[f"{tag}_within_train"]["spearman_exposure_z_cb_margin"] = corr(cz, margin(pred("TS30",tag,"train")["logits"], ytrain))

    results["stability"] = dict(persistent_sh=int(stable_sh.sum()), persistent_cb=int(stable_cb.sum()),
        unanimous_sh=int((sh_count==6).sum()), unanimous_cb=int((cb_count==6).sum()),
        persistent_both_wrong=int(stable_wrong.sum()),
        best_sh_and_persistent=int((stable_sh & (groups[("best","test")]=="sh_only")).sum()),
        best_cb_and_persistent=int((stable_cb & (groups[("best","test")]=="cb_only")).sum()),
        cb30_persistently_wrong_cb31_recovered=int(((cc.sum(0)<=1)&(bc.sum(0)>=5)).sum()))
    results["sampling"] = {a:{k:v for k,v in meta["sampling"][a].items() if k!="checks"} for a in sampling}
    results["sampling_validation"] = {a:len(meta["sampling"][a]["checks"]) for a in sampling}
    write(args.out/"summary.json",results)
    print(json.dumps({k:v for k,v in results.items() if "association" not in k},allow_nan=False),flush=True)


if __name__ == "__main__": main()
