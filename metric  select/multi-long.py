import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LassoCV, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 文件路径
file_path = r"C:\Users\17958\Desktop\train_4.0.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 记录原始数据信息
print("数据加载完成，原始数据信息：")
print(f"数据行数: {df.shape[0]}, 列数: {df.shape[1]}")

# 定义自变量列名（52个）
independent_vars = ['DPT*',  'Inner', 'LCOM',
                      'PDpt',
                    'TCOM_RAT',   'CLOC',  'CONS',  'INNER', 'jf', 'LOC', 'Jm',  'MPC',
                    'NTP',
                     'TODO']

# 定义因变量列名
dependent_var = '1适合LLM'

# 检查列名是否存在
missing_cols = [col for col in independent_vars + [dependent_var] if col not in df.columns]
if missing_cols:
    print(f"以下列名在文件中未找到：{missing_cols}")
    raise ValueError("数据中缺少部分列，请检查！")

# 确保因变量是二元变量
if not df[dependent_var].isin([0, 1]).all():
    print(f"因变量 {dependent_var} 不是二元变量，请检查数据！")
    raise ValueError("因变量必须是 0 或 1 的二元变量")


# 数据清洗前的NaN检查
def check_nan(df, columns, step_name):
    nan_counts = df[columns].isna().sum()
    nan_df = nan_counts[nan_counts > 0].to_frame(name='NaN数量')

    if not nan_df.empty:
        print(f"\n{step_name} - NaN值检查结果：")
        print(nan_df)
        total_nan = nan_df['NaN数量'].sum()
        print(f"总计NaN值: {total_nan}")
        print(f"NaN值占比: {total_nan / (df.shape[0] * len(columns)) * 100:.2f}%")
    else:
        print(f"\n{step_name} - 未检测到NaN值")

    return nan_df


# 检查原始数据中的NaN值
original_nan = check_nan(df, independent_vars + [dependent_var], "原始数据")

# 1. 数据清洗：删除包含NaN的行
df_cleaned = df.dropna(subset=independent_vars + [dependent_var])
print(f"\n数据清洗完成：")
print(f"清洗前行数: {df.shape[0]}, 清洗后行数: {df_cleaned.shape[0]}")
print(f"删除行数: {df.shape[0] - df_cleaned.shape[0]}, 保留率: {df_cleaned.shape[0] / df.shape[0] * 100:.2f}%")

# 检查清洗后数据中的NaN值
cleaned_nan = check_nan(df_cleaned, independent_vars + [dependent_var], "清洗后数据")

# 提取特征和目标变量
X = df_cleaned[independent_vars]
y = df_cleaned[dependent_var]

# 确保有足够的样本和变量
if len(X) < 2 or X.shape[1] < 1:
    raise ValueError(f"数据清洗后样本数量或变量数量不足：样本数={len(X)}, 变量数={X.shape[1]}")

# 2. 标准化自变量
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
df_standardized = pd.DataFrame(X_standardized, columns=independent_vars)
df_standardized[dependent_var] = y

# 检查标准化后数据中的NaN值
standardized_nan = check_nan(df_standardized, independent_vars + [dependent_var], "标准化后数据")


# 3. 多重共线性检测（VIF）
def calculate_vif(X):
    if X.empty or X.shape[1] < 2:
        return pd.DataFrame()  # 没有足够变量计算VIF

    # 创建VIF数据框
    vif_data = pd.DataFrame()
    vif_data["变量"] = X.columns

    # 计算VIF值
    try:
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                           for i in range(len(X.columns))]
    except np.linalg.LinAlgError:
        # 处理矩阵奇异的情况
        vif_data["VIF"] = [np.nan for _ in range(len(X.columns))]

    return vif_data


# 计算初始VIF
vif_initial = calculate_vif(X)
print("\n初始变量VIF值（多重共线性检测）：")
if not vif_initial.empty:
    print(vif_initial.sort_values(by='VIF', ascending=False))
else:
    print("变量数量不足，无法计算VIF")

# 4. 多变量逻辑回归模型构建
# 添加常数项
X_with_const = sm.add_constant(X)

# 拟合全变量模型
try:
    full_model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=200)
    print("\n全变量逻辑回归模型摘要：")
    print(full_model.summary2())
except Exception as e:
    print(f"全变量模型拟合失败：{e}")
    full_model = None


# 5. 变量选择方法1：逐步回归（向前选择）
def forward_selection(X, y, significance_level=0.05, max_iter=20):
    # 初始化所选变量
    selected_vars = []
    available_vars = list(X.columns)
    selection_steps = []

    # 限制最大迭代次数，防止无限循环
    for step in range(max_iter):
        if not available_vars:
            break

        best_p = float('inf')
        best_var = None
        best_model = None

        # 对每个可用变量，尝试加入模型
        for var in available_vars:
            current_vars = selected_vars + [var]
            X_current = X[current_vars]

            # 检查当前变量集合是否有效
            if X_current.shape[1] < 2:
                continue

            X_with_const = sm.add_constant(X_current)

            try:
                model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=200)
                p = model.pvalues[var]

                if p < best_p:
                    best_p = p
                    best_var = var
                    best_model = model
            except Exception as e:
                continue

        # 如果找到显著变量，加入模型
        if best_p < significance_level and best_var and best_model:
            selected_vars.append(best_var)
            available_vars.remove(best_var)

            # 计算加入新变量后的VIF
            X_selected = X[selected_vars]
            vif = calculate_vif(X_selected)

            selection_steps.append({
                'step': len(selected_vars),
                'added_var': best_var,
                'p_value': best_p,
                'vif': vif.to_dict('records') if not vif.empty else []
            })
        else:
            break

    return selected_vars, selection_steps


# 执行向前选择
try:
    selected_vars, selection_steps = forward_selection(X, y)
    print(f"\n逐步回归选择的变量（{len(selected_vars)}个）：")
    print(selected_vars)
except Exception as e:
    print(f"逐步回归选择变量时出错：{e}")
    selected_vars = []

# 6. 变量选择方法2：LASSO正则化
X_np = X.values
y_np = y.values

# 自动选择LASSO的alpha值
try:
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_np, y_np)
    best_alpha = lasso_cv.alpha_

    # 使用最佳alpha值拟合LASSO模型
    lasso = LogisticRegression(penalty='l1', C=1 / best_alpha, solver='saga', random_state=42, max_iter=10000)
    lasso.fit(X_np, y_np)

    # 获取LASSO选择的变量
    lasso_coef = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': lasso.coef_[0]
    })
    lasso_selected = lasso_coef[lasso_coef['Coefficient'] != 0]['Variable'].tolist()
    print(f"\nLASSO选择的变量（{len(lasso_selected)}个）：")
    print(lasso_selected)
except Exception as e:
    print(f"LASSO变量选择出错：{e}")
    lasso_selected = []

# 7. 结合两种方法的变量选择结果
combined_vars = list(set(selected_vars) | set(lasso_selected))
# 确保至少有一个变量
if not combined_vars and X.columns.size > 0:
    combined_vars = [X.columns[0]]  # 选择第一个变量作为回退
print(f"\n结合两种方法的变量（{len(combined_vars)}个）：")
print(combined_vars)

# 8. 拟合最终多变量模型
X_final = X[combined_vars]
X_final_with_const = sm.add_constant(X_final)

try:
    final_model = sm.Logit(y, X_final_with_const).fit(disp=0, maxiter=300)
    print("\n最终多变量逻辑回归模型摘要：")
    print(final_model.summary2())
except Exception as e:
    print(f"最终模型拟合失败：{e}")
    final_model = None

# 9. 模型评估
if final_model:
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # 拟合训练集模型
    model_train = sm.Logit(y_train, X_train_const).fit(disp=0, maxiter=300)

    # 在测试集上预测
    y_pred_proba = model_train.predict(X_test_const)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 计算评估指标
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\n模型评估（测试集）：")
        print(f"AUC值：{auc:.4f}")
        print(f"混淆矩阵：\n{cm}")
        print(f"分类报告：\n{report}")
    except Exception as e:
        print(f"模型评估出错：{e}")

    # 10. 交叉验证评估
    try:
        cv_scores = cross_val_score(
            LogisticRegression(random_state=42, max_iter=1000),
            X_final, y,
            cv=5,
            scoring='roc_auc'
        )
        print(f"\n5折交叉验证AUC值：{cv_scores}")
        print(f"平均AUC：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"交叉验证出错：{e}")

    # 11. 变量重要性分析（基于系数绝对值）
    if hasattr(final_model, 'params') and hasattr(final_model, 'bse'):
        var_importance = pd.DataFrame({
            'Variable': combined_vars,
            'Coefficient': final_model.params[1:].values,  # 排除截距项
            'Odds Ratio': np.exp(final_model.params[1:].values)
        })
        var_importance = var_importance.sort_values(by='Coefficient', key=abs, ascending=False)

        # 计算显著性
        try:
            var_importance['Significance'] = var_importance.apply(
                lambda row: '***' if abs(row['Coefficient'] / final_model.bse[1:][row.name]) > 3.29 else
                '**' if abs(row['Coefficient'] / final_model.bse[1:][row.name]) > 2.58 else
                '*' if abs(row['Coefficient'] / final_model.bse[1:][row.name]) > 1.96 else '',
                axis=1
            )
        except:
            var_importance['Significance'] = ''

        print("\n变量重要性（按系数绝对值排序）：")
        print(var_importance)

        # 12. 绘制变量重要性条形图
        if not var_importance.empty:
            plt.figure(figsize=(15, 8))

            # 创建从蓝色到红色的渐变色
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

            # 归一化系数到 [0, 1] 以映射颜色
            min_coef, max_coef = var_importance['Coefficient'].min(), var_importance['Coefficient'].max()
            if min_coef == max_coef:
                norm = plt.Normalize(min_coef - 0.1, max_coef + 0.1)
            else:
                norm = plt.Normalize(min_coef, max_coef)
            colors_mapped = [cmap(norm(val)) for val in var_importance['Coefficient']]

            # 绘制条形图
            bars = plt.barh(var_importance['Variable'], var_importance['Coefficient'],
                            color=colors_mapped, alpha=0.7, edgecolor='white', linewidth=1.5)

            # 标注 OR 和显著性
            for i, bar in enumerate(bars):
                or_value = var_importance.iloc[i]['Odds Ratio']
                significance = var_importance.iloc[i]['Significance']
                # 在条形图右侧标注 OR 和显著性
                label = f'OR={or_value:.2f}{significance}'
                plt.text(bar.get_width() + 0.02 if bar.get_width() >= 0 else bar.get_width() - 0.02,
                         bar.get_y() + bar.get_height() / 2, label,
                         ha='left' if bar.get_width() >= 0 else 'right', va='center',
                         color='black', fontsize=10)

            # 设置标题和标签
            plt.title('多变量逻辑回归模型中变量的标准化系数', fontsize=16, pad=20)
            plt.xlabel('标准化逻辑回归系数', fontsize=12)
            plt.ylabel('自变量', fontsize=12)
            plt.grid(True, axis='x', linestyle='--', alpha=0.3)

            # 设置背景颜色为纯色
            plt.gca().set_facecolor('#FFFFFF')
            plt.gcf().set_facecolor('#FFFFFF')

            plt.tight_layout()
            plt.savefig('multivariate_logistic_coefficients_bar.png', dpi=300)
            plt.show()

        # 13. 绘制预测概率的ROC曲线
        if 'auc' in locals():
            from sklearn.metrics import roc_curve

            try:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假正例率', fontsize=12)
                plt.ylabel('真正例率', fontsize=12)
                plt.title('多变量逻辑回归模型的ROC曲线', fontsize=16)
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('multivariate_roc_curve.png', dpi=300)
                plt.show()
            except Exception as e:
                print(f"绘制ROC曲线出错：{e}")

        # 14. 保存结果到文件
        try:
            # 保存数据清洗日志
            with open('data_cleaning_log.txt', 'w', encoding='utf-8') as f:
                f.write("数据清洗日志：\n\n")
                f.write(f"原始数据行数: {df.shape[0]}, 列数: {df.shape[1]}\n")
                f.write(f"清洗后数据行数: {df_cleaned.shape[0]}, 列数: {df_cleaned.shape[1]}\n")
                f.write(
                    f"删除行数: {df.shape[0] - df_cleaned.shape[0]}, 保留率: {df_cleaned.shape[0] / df.shape[0] * 100:.2f}%\n\n")

                f.write("原始数据NaN值统计：\n")
                if not original_nan.empty:
                    f.write(original_nan.to_csv(sep='\t', na_rep='nan'))
                else:
                    f.write("未检测到NaN值\n\n")

                f.write("清洗后数据NaN值统计：\n")
                if not cleaned_nan.empty:
                    f.write(cleaned_nan.to_csv(sep='\t', na_rep='nan'))
                else:
                    f.write("未检测到NaN值\n\n")

                f.write("标准化后数据NaN值统计：\n")
                if not standardized_nan.empty:
                    f.write(standardized_nan.to_csv(sep='\t', na_rep='nan'))
                else:
                    f.write("未检测到NaN值\n")

            # 保存模型摘要
            if final_model:
                with open('全集/multivariate_logistic_summary.txt', 'w', encoding='utf-8') as f:
                    if full_model:
                        f.write("全变量模型摘要：\n")
                        f.write(full_model.summary2().as_text())
                        f.write('\n\n\n')

                    f.write("最终模型摘要：\n")
                    f.write(final_model.summary2().as_text())

            # 保存变量重要性
            if 'var_importance' in locals() and not var_importance.empty:
                var_importance.to_excel("multivariate_variable_importance.xlsx", index=False)

            # 保存完整结果
            results = {
                '全变量模型存在': bool(full_model),
                '逐步选择变量': selected_vars,
                'LASSO选择变量': lasso_selected,
                '最终模型变量': combined_vars,
                '最终模型存在': bool(final_model),
                '交叉验证结果': cv_scores.tolist() if 'cv_scores' in locals() else []
            }
            pd.DataFrame.from_dict(results, orient='index').to_excel("multivariate_logistic_results.xlsx")

            print("\n多变量逻辑回归分析完成！")
            print("结果已保存至以下文件：")
            print("1. data_cleaning_log.txt - 数据清洗日志")
            print("2. multivariate_logistic_summary.txt - 模型摘要")
            print("3. multivariate_variable_importance.xlsx - 变量重要性")
            print("4. multivariate_logistic_results.xlsx - 完整分析结果")
            if 'multivariate_logistic_coefficients_bar.png' in locals():
                print("5. multivariate_logistic_coefficients_bar.png - 变量系数条形图")
            if 'multivariate_roc_curve.png' in locals():
                print("6. multivariate_roc_curve.png - ROC曲线")
        except Exception as e:
            print(f"保存结果出错：{e}")
else:
    print("无法完成模型评估，因为最终模型拟合失败")