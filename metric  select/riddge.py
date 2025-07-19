import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, LogisticRegression  # 修正导入
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
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
independent_vars = ['DPT*', 'Inner', 'LCOM',
                    'PDpt',
                    'TCOM_RAT', 'CLOC', 'CONS', 'INNER', 'jf', 'LOC', 'Jm', 'MPC',
                    'NTP',
                    'TODO']

# 定义因变量列名（注意：岭回归适用于连续因变量，若原数据为二分类需调整）
dependent_var = '1适合LLM'

# 检查列名是否存在
missing_cols = [col for col in independent_vars + [dependent_var] if col not in df.columns]
if missing_cols:
    print(f"以下列名在文件中未找到：{missing_cols}")
    raise ValueError("数据中缺少部分列，请检查！")


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

# 2. 标准化自变量（岭回归对尺度敏感）
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


# 4. 岭回归模型构建前的变量选择
# 备选变量选择策略：当其他方法失败时使用
def select_backup_vars(X, vif_data, min_vif_threshold=2.5):
    """选择VIF值较低且具有一定解释力的变量作为备选"""
    if vif_data.empty:
        return list(X.columns)  # 如果没有VIF数据，返回所有变量

    # 按VIF升序排序
    sorted_vif = vif_data.sort_values(by='VIF')

    # 选择VIF低于阈值的变量
    low_vif_vars = sorted_vif[sorted_vif['VIF'] < min_vif_threshold]['变量'].tolist()

    if not low_vif_vars:
        # 如果没有VIF低于阈值的变量，选择VIF最低的前3个变量
        low_vif_vars = sorted_vif.head(3)['变量'].tolist()

    return low_vif_vars


# 变量选择方法1：LASSO正则化（用于特征筛选）
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


# 变量选择方法2：逐步回归（适用于回归问题的向前选择）
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
                model = sm.OLS(y, X_with_const).fit(disp=0)
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

# 7. 结合两种方法的变量选择结果
combined_vars = list(set(selected_vars) | set(lasso_selected))

# 确保至少有一个变量：使用备选策略
if not combined_vars:
    print("\n警告：两种变量选择方法均未选出变量，使用备选策略...")
    backup_vars = select_backup_vars(X, vif_initial)
    combined_vars = backup_vars[:1]  # 选择第一个备选变量
    print(f"备选变量选择（{len(combined_vars)}个）：{combined_vars}")
else:
    print(f"\n结合两种方法的变量（{len(combined_vars)}个）：")
    print(combined_vars)

# 8. 拟合最终岭回归模型
X_final = X[combined_vars]
X_final_with_const = sm.add_constant(X_final)

try:
    # 使用RidgeCV自动选择alpha值
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5, scoring='r2')
    ridge_cv.fit(X_final, y)
    best_alpha = ridge_cv.alpha_

    # 构建最终岭回归模型
    final_model = Ridge(alpha=best_alpha, random_state=42)
    final_model.fit(X_final, y)

    # 使用statsmodels构建模型以获取摘要信息
    X_final_sm = sm.add_constant(X_final)
    sm_model = sm.OLS(y, X_final_sm).fit()

    print("\n最终岭回归模型摘要：")
    print(sm_model.summary2())
except Exception as e:
    print(f"最终模型拟合失败：{e}")
    final_model = None
    sm_model = None

# 9. 模型评估
if final_model:
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)

    # 拟合训练集模型
    model_train = Ridge(alpha=best_alpha, random_state=42)
    model_train.fit(X_train, y_train)

    # 在测试集上预测
    y_pred = model_train.predict(X_test)

    # 计算评估指标
    try:
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

        print(f"\n模型评估（测试集）：")
        print(f"R²值：{r2:.4f}")
        print(f"调整R²值：{adj_r2:.4f}")
        print(f"均方误差(MSE)：{mse:.4f}")
        print(f"均方根误差(RMSE)：{rmse:.4f}")
        print(f"平均绝对误差(MAE)：{mae:.4f}")
        print(f"解释方差分数(EVS)：{evs:.4f}")
    except Exception as e:
        print(f"模型评估出错：{e}")

    # 10. 交叉验证评估
    try:
        cv_scores = cross_val_score(
            Ridge(alpha=best_alpha, random_state=42),
            X_final, y,
            cv=5,
            scoring='r2'
        )
        print(f"\n5折交叉验证R²值：{cv_scores}")
        print(f"平均R²：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"交叉验证出错：{e}")

    # 11. 变量重要性分析（基于系数绝对值）
    if hasattr(final_model, 'coef_') and combined_vars:
        var_importance = pd.DataFrame({
            'Variable': combined_vars,
            'Coefficient': final_model.coef_
        })
        var_importance = var_importance.sort_values(by='Coefficient', key=abs, ascending=False)

        # 计算显著性（使用statsmodels的t值）
        if sm_model:
            try:
                t_values = sm_model.tvalues[1:]  # 排除截距项
                var_importance['t_value'] = t_values
                var_importance['Significance'] = var_importance.apply(
                    lambda row: '***' if abs(row['t_value']) > 3.29 else
                    '**' if abs(row['t_value']) > 2.58 else
                    '*' if abs(row['t_value']) > 1.96 else '',
                    axis=1
                )
            except:
                var_importance['Significance'] = ''
        else:
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

            # 标注显著性
            for i, bar in enumerate(bars):
                significance = var_importance.iloc[i]['Significance']
                # 在条形图右侧标注显著性
                plt.text(bar.get_width() + 0.02 if bar.get_width() >= 0 else bar.get_width() - 0.02,
                         bar.get_y() + bar.get_height() / 2, significance,
                         ha='left' if bar.get_width() >= 0 else 'right', va='center',
                         color='black', fontsize=12, weight='bold')

            # 设置标题和标签
            plt.title('岭回归模型中变量的标准化系数', fontsize=16, pad=20)
            plt.xlabel('标准化岭回归系数', fontsize=12)
            plt.ylabel('自变量', fontsize=12)
            plt.grid(True, axis='x', linestyle='--', alpha=0.3)

            # 设置背景颜色为纯色
            plt.gca().set_facecolor('#FFFFFF')
            plt.gcf().set_facecolor('#FFFFFF')

            plt.tight_layout()
            plt.savefig('ridge_regression_coefficients_bar.png', dpi=300)
            plt.show()

        # 13. 绘制预测值与实际值对比图
        try:
            plt.figure(figsize=(12, 10))

            # 主图：预测值 vs 实际值
            plt.subplot(2, 2, 1)
            plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='white')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

            # 计算偏差
            bias = y_pred - y_test
            mean_bias = np.mean(bias)
            std_bias = np.std(bias)

            plt.title(f'岭回归模型预测值 vs 实际值 (偏差: {mean_bias:.4f}±{std_bias:.4f})', fontsize=14)
            plt.xlabel('实际值', fontsize=12)
            plt.ylabel('预测值', fontsize=12)
            plt.grid(True, alpha=0.3)

            # 残差图
            plt.subplot(2, 2, 2)
            plt.scatter(y_pred, bias, color='green', alpha=0.6, edgecolor='white')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('残差图', fontsize=14)
            plt.xlabel('预测值', fontsize=12)
            plt.ylabel('残差', fontsize=12)
            plt.grid(True, alpha=0.3)

            # 残差分布直方图
            plt.subplot(2, 2, 3)
            sns.histplot(bias, kde=True, color='purple', alpha=0.6)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('残差分布', fontsize=14)
            plt.xlabel('残差', fontsize=12)
            plt.ylabel('频率', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Q-Q图：检验残差正态性
            plt.subplot(2, 2, 4)
            sm.qqplot(bias, line='s', ax=plt.gca())
            plt.title('残差Q-Q图', fontsize=14)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('ridge_regression_diagnostics.png', dpi=300)
            plt.show()
        except Exception as e:
            print(f"绘制预测对比图出错：{e}")

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
            if sm_model:
                with open('ridge_regression_summary.txt', 'w', encoding='utf-8') as f:
                    f.write("最终岭回归模型摘要：\n")
                    f.write(sm_model.summary2().as_text())

            # 保存变量重要性
            if not var_importance.empty:
                var_importance.to_excel("ridge_variable_importance.xlsx", index=False)

            # 保存完整结果
            results = {
                '逐步选择变量': selected_vars,
                'LASSO选择变量': lasso_selected,
                '最终模型变量': combined_vars,
                '最终模型存在': bool(final_model),
                '最佳alpha值': best_alpha if 'best_alpha' in locals() else None,
                '交叉验证结果': cv_scores.tolist() if 'cv_scores' in locals() else [],
                '测试集R²': r2 if 'r2' in locals() else 0,
                '测试集调整R²': adj_r2 if 'adj_r2' in locals() else 0,
                '测试集MSE': mse if 'mse' in locals() else 0,
                '测试集RMSE': rmse if 'rmse' in locals() else 0,
                '测试集MAE': mae if 'mae' in locals() else 0,
                '测试集EVS': evs if 'evs' in locals() else 0
            }

            # 转换布尔值为字符串，避免保存错误
            for key, value in results.items():
                if isinstance(value, bool):
                    results[key] = str(value)

            pd.DataFrame.from_dict(results, orient='index').to_excel("ridge_regression_results.xlsx")

            print("\n岭回归分析完成！")
            print("结果已保存至以下文件：")
            print("1. data_cleaning_log.txt - 数据清洗日志")
            print("2. ridge_regression_summary.txt - 模型摘要")
            print("3. ridge_variable_importance.xlsx - 变量重要性")
            print("4. ridge_regression_results.xlsx - 完整分析结果")
            if 'ridge_regression_coefficients_bar.png' in locals():
                print("5. ridge_regression_coefficients_bar.png - 变量系数条形图")
            if 'ridge_regression_diagnostics.png' in locals():
                print("6. ridge_regression_diagnostics.png - 模型诊断图")
        except Exception as e:
            print(f"保存结果出错：{e}")
else:
    print("无法完成模型评估，因为最终模型拟合失败")