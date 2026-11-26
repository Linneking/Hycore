"""Fill the presentation manuscript from audited scalar records; no model execution."""
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
rows = {x['id']: x for x in json.loads((ROOT/'experiments.json').read_text(encoding='utf-8'))}
old = json.loads((ROOT.parent/'197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json').read_text(encoding='utf-8'))
for x in old['runs']:
    if x['run_id'] in ('TS02','TS06','TS23'):
        rows[x['run_id']] = dict(best_oa=x['best_test_oa'], best_aa=x['best_test_aa_same_epoch'],
                                tail_oa=x['last10_test_oa_mean'], tail_aa=x['last10_test_aa_mean'])
def fmt(v, signed=False):
    if v is None:return '未测'
    value=str(Decimal(str(round(v,8))).quantize(Decimal('.001'),rounding=ROUND_HALF_UP))
    return ('+' if signed and v>=0 else '')+value
def pair(x, prefix):
    return fmt(x[prefix+'_oa'])+'／'+fmt(x[prefix+'_aa'])

stage_ids = {
 'A':[f'TS{x}' for x in range(27,33)], 'B':[f'TS{x}' for x in range(33,39)],
 'C':[f'TS{x}' for x in range(39,45)], 'D':[f'TS{x}' for x in range(45,51)],
 'E':['TS51','TS52','TS54','TS55','TS53'],
 'F':['SO-H0','SO-HIER','SO-S12-H0','SO-S12-HIER','MN40-AUTHOR-H0-S4780'],
 'G':['SO-S4780-H0','SO-S4780-HIER','MN40-B05-S4781-H0','MN40-B05-S4781-HIER-FIFO1'],
 'H':['DeLA-v1-AUTHOR-S4780','DG-S12-H0','DG-S12-HIER','DG-S4780-H0','DG-S4780-HIER',
      'MN40-B05-S4781-HIER-CURRENT32','MN40-AUTHOR-TRAIN-ORDER-S4780','SO-S12-H0-PAPER-INTRA'],
 'I':['DeLA-H0','DeLA-HIER'],
 'J':['SO-S12-HIER-L0.1','SO-S12-HIER-L1.0','DG-S4780-CB32-H0','DG-S4780-CB32-HIER'],
}
controls = {
 **{f'TS{x}':'TS23' for x in (27,28,29)}, 'TS31':'TS30','TS32':'TS30',
 'TS33':'TS32','TS34':'TS32',**{f'TS{x}':'TS06' for x in range(35,39)},
 'TS40':'TS39','TS41':'TS39','TS42':'TS35','TS43':'TS35','TS44':'TS35',
 'TS45':'TS35','TS46':'TS35','TS47':'TS44','TS48':'TS49','TS49':'TS44','TS50':'TS44',
 'TS52':'TS51','TS55':'TS54','TS53':'TS06',
 'SO-HIER':'SO-H0','SO-S12-HIER':'SO-S12-H0','SO-S4780-HIER':'SO-S4780-H0',
 'MN40-B05-S4781-HIER-FIFO1':'MN40-B05-S4781-H0',
 'MN40-B05-S4781-HIER-CURRENT32':'MN40-B05-S4781-H0',
 'DG-S12-HIER':'DG-S12-H0','DG-S4780-HIER':'DG-S4780-H0',
 'MN40-AUTHOR-TRAIN-ORDER-S4780':'MN40-AUTHOR-H0-S4780','SO-S12-H0-PAPER-INTRA':'SO-S12-H0',
 'SO-S12-HIER-L0.1':'SO-S12-H0','SO-S12-HIER-L1.0':'SO-S12-H0',
 'DG-S4780-CB32-HIER':'DG-S4780-CB32-H0',
}
short_change = {
 'TS27':'代理512→256','TS28':'代理初始系数.9→.5','TS29':'仅代理cap2.3→2.9',
 'TS30':'CB32×200 H0','TS31':'CB32 H0增加至300步/轮','TS32':'CB32 K20/20、λ=.5',
 'TS33':'CB32 A1→A0','TS34':'CB32 λ=.5→1','TS35':'SH32 当前池32、λ=.5',
 'TS36':'SH32 λ=.1','TS37':'SH32 λ=1','TS38':'SH32 K10/10',
 'TS39':'B17两批累积 H0','TS40':'B17各32池挖掘','TS41':'B17实时64池',
 'TS42':'仅Ks20→10','TS43':'仅Kp20→10','TS44':'当前32＋前一步32缓存',
 'TS45':'样本权重渐升；代理项.5','TS46':'仅样本项.5；代理内部项0',
 'TS47':'FIFO两步，池最多96','TS48':'长窗口按类选32','TS49':'同窗口随机选32','TS50':'FIFO64 λ=.1',
 'TS51':'MN20 SH32 H0','TS52':'MN20 SH32 HIER','TS54':'MN20 CB32 H0','TS55':'MN20 CB32 HIER',
 'TS53':'完整MN40 SH分类＋独立CB-HIER',
}
def table(stage):
    if stage=='I':
        result=['| 正式组 | 固定e250 OA／AA | 相对新H0 ΔOA／AA | 尝试／有效更新 |',
                '|---|---:|---:|---:|']
        baseline=rows['DeLA-H0']
        for identity in stage_ids[stage]:
            x=rows[identity]
            delta='—' if identity=='DeLA-H0' else fmt(x['last_oa']-baseline['last_oa'],True)+'／'+fmt(x['last_aa']-baseline['last_aa'],True)
            result.append(f"| {identity} | {pair(x,'last')} | {delta} | 89,000／88,961 |")
        return '\n'.join(result)
    result=['| 正式组／本组改动 | 指定对照 | best OA／同轮AA | 末轮OA／AA | 末10 OA／AA | Δ末10 OA |',
            '|---|---|---:|---:|---:|---:|']
    for identity in stage_ids[stage]:
        x=rows[identity];control=controls.get(identity)
        change=short_change.get(identity,x['change'])
        delta=fmt(x['tail_oa']-rows[control]['tail_oa'],True) if control else '—'
        result.append(f"| **{identity}**：{change} | {control or '本阶段H0／独立基准'} | {pair(x,'best')} | {pair(x,'last')} | {pair(x,'tail')} | {delta} |")
    return '\n'.join(result)

paper_cards=json.loads((ROOT/'paper_cards.json').read_text(encoding='utf-8'))
paper_text=(ROOT/'paper_baselines.md').read_text(encoding='utf-8').split('\n',1)[1].split('\n## DeLA：',1)[0].strip()
paper_text+='\n\n论文／作者重复统计／发布权重复评的完整区分，以及point2vec来源精度差异，见[论文核验页](paper_baselines.md)。'
text=(ROOT/'report_template.md').read_text(encoding='utf-8')
text=text.replace('{{PAPER_BASELINES}}',paper_text)
for key,value in paper_cards.items():text=text.replace('{{'+key+'}}',value)
for stage in stage_ids:text=text.replace('{{TABLE_'+stage+'}}',table(stage))
public='''| 发布权重与推理条件 | 数据集 | 实际单次OA／AA | 本地训练预算／角色 |
|---|---|---:|---|
| DeLA作者权重 | MN40 | 94.000／92.250 | 不训练，独立性能参照 |
| DeLA作者权重 | SO PB_T50_RS | 90.420／89.240 | 不训练，独立性能参照 |
| point2vec FPS0、FP16（预定主条件） | MN40 | 94.287／91.903 | 同一预训练后微调权重 |
| point2vec FPS0、FP32 | MN40 | 94.287／91.903 | 精度对照 |
| point2vec FPS1、FP16 | MN40 | 94.125／92.012 | 只变推理FPS seed |
| point2vec FPS12、FP16 | MN40 | 94.368／91.861 | 只变推理FPS seed |
| point2vec FPS4780、FP16 | MN40 | 94.327／91.845 | 只变推理FPS seed |'''
text=text.replace('{{TABLE_PUBLIC}}',public)
for key,name,caption in [
 ('FIG_MN20','mn20_oa_aa.png','图1：同子集直接配对。SH32的AA改善与OA下降必须一起保留。'),
 ('FIG_PAIRED','paired_seed_effects.png','图2：关键方案的末段增量按seed展开；绝对H0与HIER读数见上表。'),
 ('FIG_DELA','dela_absolute_levels.png','图3：同数据集的绝对成绩。发布权重只作独立参考，新H0→HIER是直接开关。'),
]:
    path='/'+(ROOT/'figures'/name).as_posix()
    text=text.replace('{{'+key+'}}',f'![{caption}]({path})\n\n{caption}')
remaining=re.findall(r'\{\{[^}]+\}\}',text)
if remaining:raise RuntimeError('Unfilled manuscript placeholders: '+repr(remaining))
flat=[identity for ids in stage_ids.values() for identity in ids]
original_data=json.loads((ROOT/'experiments.json').read_text(encoding='utf-8'))
assert len(flat)==52 and len(set(flat))==52 and set(flat)==set(x['id'] for x in original_data)
stage_names={
 'A':'11/12–13凌晨 代理消融、H0和预算', 'B':'11/13晚–15凌晨 SH32、优化器与权重',
 'C':'11/15–16下午 实时池、分项K与缓存', 'D':'11/16晚–18 分项权重与历史候选',
 'E':'11/19 MN20与双采样', 'F':'11/20–21 SO迁移与作者入口',
 'G':'11/23 配对seed复验', 'H':'11/24 发布权重、复现与替代骨干',
 'I':'11/25 DeLA Part–Whole配对', 'J':'11/26 权重与CB32配对'}
by_id={identity:stage for stage,ids in stage_ids.items() for identity in ids}
for x in original_data:
    x['stage']=by_id[x['id']]+'：'+stage_names[by_id[x['id']]]
(ROOT/'experiments.json').write_text(json.dumps(original_data,ensure_ascii=False,indent=2),encoding='utf-8')
coverage=json.loads((ROOT/'coverage.json').read_text(encoding='utf-8'))
for x in coverage['formal_runs']:x['stage']=by_id[x['id']]+'：'+stage_names[by_id[x['id']]]
coverage['counts_by_manuscript_stage']={stage:len(ids) for stage,ids in stage_ids.items()}
(ROOT/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
appendix_path=ROOT/'appendix_experiments.md'
appendix=re.sub(r'^阶段：[^\n]*\n\n', '', appendix_path.read_text(encoding='utf-8'), flags=re.MULTILINE)
appendix_path.write_text(appendix,encoding='utf-8')
(ROOT/'report.md').write_text(text.rstrip()+'\n',encoding='utf-8')
(ROOT/'stage_index.json').write_text(json.dumps(dict(stages=stage_ids,comparison_controls=controls),ensure_ascii=False,indent=2),encoding='utf-8')
print('Built report: 52 unique formal runs in 10 stages; all placeholders filled')
