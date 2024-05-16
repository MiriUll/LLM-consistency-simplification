import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from plot_utils import calculate_errors_per_level

plt.figure(figsize=(6.4, 4.3))


parser = ArgumentParser()
parser.add_argument('-d', '--dataset')
parser.add_argument('-s', '--simp_level', help="Column that indicates the simplification level")
parser.add_argument('-r', '--reduced_tasks', help="Whether to include the reduced tasks", action="store_true")
parser.add_argument('-f', '--factuality_tolerance', type=float,
                    help="Levenshtein distance filter, keep only samples with distance below tolerance", required=False)
args = parser.parse_args()

aligned_combined = pd.read_csv('data/' + args.dataset, low_memory=False)
simple_versions = sorted(aligned_combined[args.simp_level].unique())
outpath = f"plots_new/{args.dataset.replace('.csv', '')}.pdf"

tasks = ['emotion', 'fake_news', 'topic', 'sentiment_2']
colors = {t: c for t, c in zip(tasks, ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf'])}
line_styles = {t: c for t, c in zip(tasks, ['solid'] * len(tasks))}
labels = {t: c for t, c in zip(tasks, ['Emotion', 'Fake news', 'News Topic', 'Sentiment'])}
markers = {t: c for t, c in zip(tasks, ['*', '^', 'o', 's'])}

print(args.reduced_tasks)
if args.reduced_tasks:
    label_mapping = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2, 'LABEL_3': 3, 'LABEL_4': 4, 'LABEL_5': 5, '1 star': 1, '2 stars': 2, '3 stars': 3, '4 stars': 4, '5 stars': 5, 1.0: 1, 0.0: 0}
    label_mapping_emotion = {'anger': 'negative', 'disgust': 'negative', 'fear': 'negative', 'joy': 'positive', 'sadness': 'negative', 'surprise': 'negative', 'neutral': 'neutral'}
    aligned_combined = aligned_combined.replace(label_mapping)
    aligned_combined['emotion_reduced_simple'] = aligned_combined['emotion_simple'].replace(label_mapping_emotion, inplace=False)
    aligned_combined['emotion_reduced_normal'] = aligned_combined['emotion_normal'].replace(label_mapping_emotion, inplace=False)

    tasks += ['emotion_reduced', 'sentiment_reduced']
    colors.update({'emotion_reduced': '#377eb8', 'sentiment_reduced': '#f781bf'})
    labels.update({'emotion_reduced': 'Emotion reduced', 'sentiment_reduced': 'Sentiment top2'})
    markers.update({'emotion_reduced': '*', 'sentiment_reduced': 's'})
    line_styles = {t: 'solid' if '_reduced' in t else 'dotted' for t in tasks}

    outpath = outpath.replace(".pdf", "_reduced.pdf")

print(args.factuality_tolerance)
if args.factuality_tolerance is not None:
    aligned_combined = aligned_combined[aligned_combined.distance <= args.factuality_tolerance]
    outpath = outpath.replace(".pdf", "_factFiltered.pdf")


errors_per_level = calculate_errors_per_level(dataset=aligned_combined, simple_versions=simple_versions,
                                              tasks=tasks, level_column=args.simp_level)

for task in colors.keys():
    plt.plot([level[task] for level in errors_per_level.values()], label=labels[task], color=colors[task],
                 marker=markers[task], linestyle=line_styles[task])

plt.xticks(range(len(errors_per_level.keys())), labels=errors_per_level.keys())
plt.legend()
#plt.ylabel('Percentage of deviating predictions')
plt.ylabel('Prediction change rate (%)')
plt.xlabel('Simplification strength')
plt.ylim(-1, 53)
plt.savefig(outpath)
plt.show()