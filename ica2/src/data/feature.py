import numpy as np
import pandas as pd


def _cyclic(series: pd.Series, period: float):
    """将周期性数值编码为 sin/cos 对。"""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """极其克制的物理特征提取。

    反思深刻教训：在预测步长仅为1（往前看30分钟），且自回归输入长达 48 步（一整天）的LSTM中，
    输入维度（n_features）的膨胀会严重稀释核心负荷信号(tsd)在 Linear Projector 里的权重。
    
    1. LSTM 可以通过近48步的形态完美获知“今天是几点”和“现在是冬天还是夏天（看整体发力量级）”。
       所以，任何日内周期 (sp_sin) 和年度周期 (doy_sin) 都是在加噪和拖累模型收敛！
    2. LSTM 同样能够捕捉连贯的自回归惯性，任何显式特征延迟（滞后项）在这里不仅冗余，还会破坏平滑性。
    3. LSTM 唯一绝对无法通过过去48步历史推断出来的，就是《日历规则的突然跳变》——
       比如今晚跨越零点后，明天是不是法定假日？明天是不是周末休息日？
    
    所以，最高效的手段是拔除其他一切干扰，仅喂给模型最精准的一把钥匙：对齐到目标时刻的 is_day_off！
    """
    df = df.copy()

    # 处理 is_holiday 类型，防止字符型 1 和数值型 1 不匹配
    is_holiday = df["is_holiday"].astype(str).str.strip() == "1"
    
    # 休息日指示器 (极其纯粹的二元物理特征：周末或节假日)
    df["is_day_off"] = ((df.index.dayofweek >= 5) | is_holiday).astype(np.float32)

    # ============== 核心的时间对齐 (Target Alignment) ==============
    # 我们用时刻 t 的内部状态预测时刻 t+1。如果是周五晚23:30，它不知道下一秒进入周末。
    # 必须通过 shift(-1) 把目标步(t+1)的已知休息日历赋予给当前输入矩阵X的最末端！
    df["is_day_off"] = df["is_day_off"].shift(-1)
    
    # 填补最后一行
    df.loc[df.index[-1], "is_day_off"] = df.loc[df.index[-2], "is_day_off"]

    return df

