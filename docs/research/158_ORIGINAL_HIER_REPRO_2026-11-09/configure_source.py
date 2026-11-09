"""Adapt the published trainer to torchrun and best-only checkpoint retention."""
import sys
from pathlib import Path


path = Path(sys.argv[1]) / 'hier' / 'train.py'
source = path.read_text(encoding='utf-8')


def change(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError(f'Unexpected upstream source at {old[:80]!r}')
    source = source.replace(old, new)


change('import wandb\n', '')
change('    args = parser.parse_args()\n', '''    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
''')
change('    if args.local_rank == 0:\n        wandb.init(project="hyp_metric", name="{}_{}_{}".format(args.dataset, args.model, args.run_name), config=args)\n', '')
change('    utils.init_distributed_mode(args)\n', '''    utils.init_distributed_mode(args)
    (Path(args.output_dir) / args.dataset).mkdir(exist_ok=True)
    args.best_recall = -1.0
''')
change('    if epoch % args.eval_freq == 0 and epoch >= args.warmup_epochs:\n        args.best_recall = rh_model\n', '')
start = source.index('        save_dict = {')
end = source.index('        log_stats =', start)
source = source[:start] + '''        score = train_stats.get("R@1_head")
        if score is not None:
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite R@1 at epoch {epoch + 1}")
            if score > args.best_recall:
                args.best_recall = score
                if utils.is_main_process():
                    output = Path(args.output_dir)
                    temporary = output / ".checkpoint_best.tmp"
                    payload = {
                        "stduent": model.state_dict(),
                        "sup_metric_loss": sup_metric_loss.state_dict(),
                        "epoch": epoch + 1,
                        "args": args,
                    }
                    if args.lambda2 > 0:
                        payload["cluster_loss"] = cluster_loss.state_dict()
                    torch.save(payload, temporary)
                    os.replace(temporary, output / "checkpoint_best.pth")
                    print(f"BEST_SAVED epoch={epoch+1} R@1={score:.10f}", flush=True)

''' + source[end:]
change('            wandb.log(train_stats, step=epoch)\n', '')
compile(source, str(path), 'exec')
path.write_text(source, encoding='utf-8')
