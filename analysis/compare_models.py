import os
import csv
import math
import matplotlib.pyplot as plt

p1 = 'rl_multi_turn/experiment_logs/20251215_003530/results.csv'
p2 = 'rl_multi_turn/experiment_logs/20251215_230304/results.csv'
labels = ['Qwen3-VL-2B', 'Qwen3-VL-8B']

out_dir = 'rl_multi_turn/experiment_logs/analysis_outputs'
os.makedirs(out_dir, exist_ok=True)

def load_simple(path):
    times = []
    steps = []
    gt = []
    pred = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i,name in enumerate(header)}
        for r in reader:
            # guard short rows
            if len(r) < len(header):
                # try to pad
                r = r + ['']*(len(header)-len(r))
            # answers
            g = r[idx.get('gt_answer','')] if 'gt_answer' in idx else ''
            m = r[idx.get('model_answer','')] if 'model_answer' in idx else ''
            gt.append(g.strip())
            pred.append(m.strip())
            # numeric
            if 'time_seconds' in idx:
                t = r[idx['time_seconds']]
                try:
                    times.append(float(t))
                except Exception:
                    pass
            if 'num_steps' in idx:
                s = r[idx['num_steps']]
                try:
                    steps.append(int(float(s)))
                except Exception:
                    pass
    return {'time_seconds': times, 'num_steps': steps, 'gt': gt, 'pred': pred}

data1 = load_simple(p1)
data2 = load_simple(p2)

def accuracy_simple(gt_list, pred_list):
    n = len(gt_list)
    if n == 0:
        return None, 0
    matches = 0
    for g,p in zip(gt_list, pred_list):
        if g == p and g != '':
            matches += 1
    return matches / n, n

acc1, n1 = accuracy_simple(data1['gt'], data1['pred'])
acc2, n2 = accuracy_simple(data2['gt'], data2['pred'])

# Plot time_seconds overlay
fig1, ax1 = plt.subplots(figsize=(8,5))
ax1.hist(data1['time_seconds'], bins=50, alpha=0.6, label=f"{labels[0]} (n={len(data1['time_seconds'])})")
ax1.hist(data2['time_seconds'], bins=50, alpha=0.6, label=f"{labels[1]} (n={len(data2['time_seconds'])})")
ax1.set_xlabel('time_seconds')
ax1.set_ylabel('count')
ax1.set_title('Overlay histogram: time_seconds')
ax1.legend()
fn_time = os.path.join(out_dir, 'time_seconds_overlay.png')
fig1.tight_layout()
fig1.savefig(fn_time)
plt.close(fig1)

# Plot num_steps overlay
fig2, ax2 = plt.subplots(figsize=(8,5))
ns1 = data1['num_steps']
ns2 = data2['num_steps']
if len(ns1)+len(ns2) > 0:
    all_ns = ns1 + ns2
    mn = min(all_ns)
    mx = max(all_ns)
    bins = range(mn, mx+2)
else:
    bins = 10
ax2.hist(ns1, bins=bins, alpha=0.6, label=f"{labels[0]} (n={len(ns1)})")
ax2.hist(ns2, bins=bins, alpha=0.6, label=f"{labels[1]} (n={len(ns2)})")
ax2.set_xlabel('num_steps')
ax2.set_ylabel('count')
ax2.set_title('Overlay histogram: num_steps')
ax2.legend()
fn_steps = os.path.join(out_dir, 'num_steps_overlay.png')
fig2.tight_layout()
fig2.savefig(fn_steps)
plt.close(fig2)

# Write report
report = []
report.append(f'File1: {p1} (label={labels[0]})')
report.append(f'File2: {p2} (label={labels[1]})')
report.append('')
if acc1 is not None:
    report.append(f'{labels[0]}: accuracy={acc1:.4f} ({int(acc1*n1)}/{n1})')
else:
    report.append(f'{labels[0]}: accuracy=N/A')
if acc2 is not None:
    report.append(f'{labels[1]}: accuracy={acc2:.4f} ({int(acc2*n2)}/{n2})')
else:
    report.append(f'{labels[1]}: accuracy=N/A')
report.append('')
report.append(f'Time histogram: {fn_time}')
report.append(f'Num-steps histogram: {fn_steps}')

report_txt = '\n'.join(report)
with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
    f.write(report_txt)

print(report_txt)
print('\nSaved images:')
print(fn_time)
print(fn_steps)
print('\nReport file:')
print(os.path.join(out_dir, 'report.txt'))
