import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from plot_utils import calculate_errors_per_level

plt.figure(figsize=(6.4, 4.3))

parser = ArgumentParser()
parser.add_argument('-do', '--dataset_orig')
parser.add_argument('-dc', '--dataset_compared')
parser.add_argument('-s', '--simp_level', help="Column that indicates the simplification level")
parser.add_argument('-cc', '--column_compared', help="Column prefix/postfix")
parser.add_argument('-f', '--factuality_tolerance', type=float,
                    help="Levenshtein distance filter, keep only samples with distance below tolerance", required=False)
args = parser.parse_args()

aligned_combined = pd.read_csv('data/' + args.dataset_orig, low_memory=False).fillna('')
simple_versions = sorted(aligned_combined[args.simp_level].unique())
outpath = f"plots_new/{args.dataset_orig.replace('.csv', '')}.pdf"

tasks = ['emotion', 'fake_news', 'topic', 'sentiment_2']
colors = {t: c for t, c in zip(tasks, ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf'])}
line_styles = {t: c for t, c in zip(tasks, ['solid'] * len(tasks))}
labels = {t: c for t, c in zip(tasks, ['Emotion', 'Fake news', 'News Topic', 'Sentiment'])}
markers = {t: c for t, c in zip(tasks, ['*', '^', 'o', 's'])}

print(args.factuality_tolerance)
if args.factuality_tolerance is not None:
    aligned_compared = aligned_combined[aligned_combined.distance <= args.factuality_tolerance].copy()
    outpath = outpath.replace(".pdf", "_factFiltered.pdf")
    comp_label = "filtered"
    orig_label = "unfiltered"
elif args.dataset_compared is not None:
    aligned_compared = pd.read_csv('data/' + args.dataset_compared, low_memory=False)
    outpath = f"plots_new/{args.dataset_compared.replace('.csv', '')}.pdf"
    comp_label = "NE maksed"
    orig_label = "unmasked"
elif args.column_compared is not None:
    aligned_compared = aligned_combined.copy()
    compared_columns = [col for col in aligned_compared.columns if args.column_compared in col]
    column_map = {col: col.replace(args.column_compared, '') for col in compared_columns}
    aligned_compared = aligned_compared.drop(list(set(column_map.values()) & set(aligned_compared.columns)), axis=1).rename(columns=column_map)
    outpath = outpath.replace(".pdf", f"_{args.column_compared}.pdf")
    comp_label = "GPT3.5"
    orig_label = "classifiers"
else:
    raise ValueError("No dataset to compare to!")


errors_per_level_orig = calculate_errors_per_level(dataset=aligned_combined, simple_versions=simple_versions,
                                              tasks=tasks, level_column=args.simp_level)
errors_per_level_compared = calculate_errors_per_level(dataset=aligned_compared, simple_versions=simple_versions,
                                              tasks=tasks, level_column=args.simp_level)

for task in colors.keys():
    plt.plot([level[task] for level in errors_per_level_orig.values()], color=colors[task],
                 marker=markers[task], linestyle='dotted')
    plt.plot([level[task] for level in errors_per_level_compared.values()], color=colors[task],
                 marker=markers[task], linestyle='solid')
    plt.plot([], [], label=labels[task], color=colors[task], linestyle='dashdot', marker=markers[task])

plt.plot([],[], color='grey', linestyle='solid', label=comp_label)
plt.plot([],[], color='grey', linestyle='dotted', label=orig_label)
plt.xticks(range(len(errors_per_level_orig.keys())), labels=errors_per_level_orig.keys())
plt.legend()
#plt.ylabel('Percentage of deviating predictions')
plt.ylabel('Prediction change rate (%)')
plt.xlabel('Simplification strength')
plt.ylim(-1, 53)
#plt.ylim(3, 45)
plt.savefig(outpath)
#plt.savefig(f"plots/{args.dataset}_dist_filtered.pdf")
plt.show()
