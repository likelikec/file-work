import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_with_model():
    # 配置文件路径
    model_path = 'trained_model-1.h5'
    test_data_path = r"C:\Users\17958\Desktop\start_test.xlsx"
    output_path = r"C:\Users\17958\Desktop\strat_test_with_predictions.xlsx"

    try:
        # 1. 加载测试数据
        test_df = pd.read_excel(test_data_path)
        print(f"原始测试集样本数: {len(test_df)}")

        # 2. 定义特征列（必须与训练时完全一致）
        feature_columns = [
            "WMC", "DIT", "NOC", "B", "D", "N", "n", "V",
            "NCLOC", "LOC", "Cyclic", "Dcy*", "DPT*", "PDcy",
            "PDpt", "NOIC", "Level", "INNER", "OCmax", "OSmax",
            "STAT", "SUB", "TCOM_RAT"
        ]

        # 3. 数据预处理
        # 3.1 确保特征列存在
        missing_cols = [col for col in feature_columns if col not in test_df.columns]
        if missing_cols:
            raise ValueError(f"缺失必要特征列: {missing_cols}")

        # 3.2 处理缺失值（使用中位数填充）
        imputer = SimpleImputer(strategy='median')
        X_test = imputer.fit_transform(test_df[feature_columns])

        # 3.3 标准化处理（重要！需使用训练时的参数）
        # 注意：实际使用时应加载训练时保存的scaler，此处为演示使用临时scaler
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)  # 警告：这会导致分布不一致

        # 4. 加载训练好的模型
        model = load_model(model_path)
        print("模型加载成功")

        # 5. 进行预测
        predictions = model.predict(X_test_scaled)
        test_df['预测概率'] = predictions  # 添加概率预测列
        test_df['预测结果'] = (predictions > 0.5).astype(int)  # 添加二分类结果

        # 6. 保存结果
        test_df.to_excel(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")
        print(f"新增列预览:\n{test_df[['预测概率', '预测结果']].head()}")

    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    predict_with_model()