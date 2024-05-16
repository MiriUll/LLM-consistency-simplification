import pandas as pd

def compare_task(task, df, tolerance=0):
    same_pred = []
    for n, s in zip(df[task + '_normal'], df[task + '_simple']):
        if type(n)==str and type(s) == str:
            same_pred.append(n.lower()==s.lower())
        else:
            same_pred.append(abs(n - s) <= tolerance)
    equal = same_pred.count(True)
    different = same_pred.count(False)
    error = round(100*different/len(same_pred), 2)
    print(f"**{task} error rate: {error}% with {len(same_pred)} samples")
    return error

def calculate_errors_per_level(dataset:pd.DataFrame, simple_versions:list, tasks:list, level_column:str) -> dict:
    errors_per_level = {}
    for simp in simple_versions:
        group = dataset[dataset[level_column] == simp]
        print('\n*', simp)
        df_tmp = {}
        for task in tasks:
            if task == "sentiment_reduced":
                df_tmp[task] = compare_task("sentiment", group, tolerance=1)
            else:
                df_tmp[task] = compare_task(task, group)
        errors_per_level[simp] = df_tmp
    return errors_per_level