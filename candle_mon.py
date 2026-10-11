import time
import requests
import pandas as pd
import numpy as np


def fetch_upbit_candles(market: str, candle_type: str, count: int = 100, unit: int = None) -> pd.DataFrame:
    base_url = "https://api.upbit.com/v1/candles"
    url = f"{base_url}/days" if candle_type == "days" else f"{base_url}/minutes/{unit}"
    params = {"market": market, "count": count}
    headers = {"Accept": "application/json"}

    res = requests.get(url, params=params, headers=headers)
    if res.status_code != 200:
        raise ConnectionError(f"API Error {res.status_code}: {res.text}")

    df = pd.DataFrame(res.json()).iloc[::-1].reset_index(drop=True)
    numeric_cols = ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    return df


def analyze_bb_dynamic_pattern(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0) -> dict:
    close = df['trade_price']
    ma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std()

    upper = ma + (std * bb_std)
    lower = ma - (std * bb_std)
    band_width = (upper - lower) / ma  # 정규화된 밴드폭 (BandWidth)
    pct_b = (close - lower) / (upper - lower)

    # 최근 3개 캔들의 시계열 데이터 추출
    c_now, c_prev, c_prev2 = close.iloc[-1], close.iloc[-2], close.iloc[-3]
    o_now = df['opening_price'].iloc[-1]

    pb_now, pb_prev, pb_prev2 = pct_b.iloc[-1], pct_b.iloc[-2], pct_b.iloc[-3]
    ma_now, ma_prev = ma.iloc[-1], ma.iloc[-2]
    bw_now = band_width.iloc[-1]

    # 20봉 기준 밴드 수축(스퀴즈) 상태 판단
    min_bw_20 = band_width.tail(20).min()
    is_squeeze = bw_now <= (min_bw_20 * 1.15)
    bw_expanding = bw_now > band_width.iloc[-2]

    # --- 패턴 식별 로직 ---
    pattern_name = "횡보 / 방향 탐색"
    bias = "중립"
    strength = "보통"

    # 1. 밴드 라이딩 (Band Walking)
    if pb_now >= 0.85 and pb_prev >= 0.85 and ma_now > ma_prev:
        pattern_name = "강력 밴드 라이딩 (상승 추세 유지)"
        bias = "상승 기대 (추세 지속)"
        strength = "강함"

    # 2. 스퀴즈 후 밴드 확장 + 상향 분출
    elif is_squeeze and bw_expanding and (pb_now > 0.8 and pb_prev <= 0.8):
        pattern_name = "스퀴즈 후 상단 돌파 (변동성 확장)"
        bias = "강력 상승 기대 (폭발 구간)"
        strength = "매우 강함"

    # 3. 상단 저항 맞고 이탈 (False Breakout / 고점 반락)
    elif pb_prev >= 1.0 and pb_now < 0.9 and (c_now < o_now):
        pattern_name = "상단 저항 피뢰침/음봉 이탈"
        bias = "하락 주의 (단기 고점 조정)"
        strength = "강함"

    # 4. 하단 터치 후 양봉 반등 (W 바닥 또는 지지 반등)
    elif pb_prev <= 0.1 and pb_now > 0.2 and (c_now > o_now):
        pattern_name = "하단 지지 확인 후 양봉 반등"
        bias = "상승 기대 (기술적 반등)"
        strength = "강함"

    # 5. 중심선(20MA) 골든크로스 / 데드크로스
    elif pb_prev < 0.5 and pb_now >= 0.5:
        pattern_name = "중심선(20MA) 상향 돌파"
        bias = "상승 전환 기대"
        strength = "보통"
    elif pb_prev > 0.5 and pb_now <= 0.5:
        pattern_name = "중심선(20MA) 하향 이탈"
        bias = "하락 전환 주의"
        strength = "보통"

    # 6. 하단 붕괴 추세 지속 (하향 밴드 라이딩)
    elif pb_now <= 0.15 and pb_prev <= 0.15 and ma_now < ma_prev:
        pattern_name = "하향 밴드 라이딩 (패닉/하락 추세)"
        bias = "추가 하락 주의"
        strength = "매우 강함"

    # 평균 변동량 (최근 14개봉 ATR 대용)
    candle_ranges = df['high_price'] - df['low_price']
    avg_volatility_krw = candle_ranges.tail(14).mean()
    avg_volatility_pct = (candle_ranges / df['opening_price'] * 100).tail(14).mean()

    return {
        "price": c_now,
        "pct_b_now": round(pb_now * 100, 1),
        "pct_b_prev": round(pb_prev * 100, 1),
        "pattern": pattern_name,
        "bias": bias,
        "strength": strength,
        "bandwidth_state": "확장 중" if bw_expanding else "수축/유지",
        "expected_range_krw": round(avg_volatility_krw, 2),
        "expected_range_pct": round(avg_volatility_pct, 2)
    }


def run_pattern_analysis(market: str = "KRW-BTC"):
    timeframes = [
        {"name": "일봉 (1D)", "type": "days", "unit": None},
        {"name": "4시간봉 (4H)", "type": "minutes", "unit": 240},
        {"name": "1시간봉 (1H)", "type": "minutes", "unit": 60},
        {"name": "30분봉 (30M)", "type": "minutes", "unit": 30},
        {"name": "10분봉 (10M)", "type": "minutes", "unit": 10}
    ]

    rows = []
    for tf in timeframes:
        try:
            df = fetch_upbit_candles(market, tf["type"], count=100, unit=tf["unit"])
            res = analyze_bb_dynamic_pattern(df)
            rows.append({
                "타임프레임": tf["name"],
                "현재가": f"{res['price']:,}원",
                "%B (직전→현재)": f"{res['pct_b_prev']}% → {res['pct_b_now']}%",
                "감지된 캔들 패턴": res["pattern"],
                "예상 방향": res["bias"],
                "신호 강도": res["strength"],
                "예상 변동폭": f"±{res['expected_range_pct']}%"
            })
            time.sleep(0.12)
        except Exception as e:
            rows.append({"타임프레임": tf["name"], "현재가": "오류", "감지된 캔들 패턴": str(e)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    report = run_pattern_analysis("KRW-CFX")
    print(report.to_string(index=False))