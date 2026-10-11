import time
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

UPBIT_BASE = "https://api.upbit.com/v1"

def get_markets(fiat="KRW"):
    r = requests.get(f"{UPBIT_BASE}/market/all", params={"isDetails":"false"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    return df[df["market"].str.startswith(fiat)].reset_index(drop=True)

def get_candles_daily(market, count=200, to=None):
    # to: 'YYYY-MM-DD HH:MM:SS' KST 기준. 없으면 최신
    params = {"market": market, "count": count}
    if to:
        params["to"] = to
    r = requests.get(f"{UPBIT_BASE}/candles/days", params=params)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    # 컬럼 정리
    df = df.rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_price": "acc_price",   # 거래대금 (KRW)
        "candle_acc_trade_volume": "acc_volume"  # 거래량 (코인 수)
    })
    # 최신이 앞에 오므로 뒤집기
    df = df.iloc[::-1].reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    return df

def add_features(df, vol_win=20, mom_win=10):
    df = df.copy()
    # 수익률, 모멘텀
    df["ret1"] = df["close"].pct_change()
    df["ma_short"] = df["close"].rolling(5).mean()
    df["ma_long"]  = df["close"].rolling(20).mean()
    df["ma_cross_up"] = (df["ma_short"] > df["ma_long"]).astype(int)

    # 전고/전저 대비 위치
    df["hhv20"] = df["high"].rolling(20).max()
    df["llv20"] = df["low"].rolling(20).min()
    df["pos_hhv"] = (df["close"] - df["llv20"]) / (df["hhv20"] - df["llv20"]).replace(0, np.nan)

    # 거래대금 평균 대비 비율
    df["acc_price_ma"] = df["acc_price"].rolling(vol_win).mean()
    df["acc_price_ratio"] = df["acc_price"] / df["acc_price_ma"]

    # OBV 대용(거래대금 부호 누적)
    df["sign"] = np.sign(df["ret1"]).fillna(0)
    df["obv_like"] = (df["acc_price"] * df["sign"]).cumsum()

    # 변동성: ATR, 볼밴 폭
    tr = np.maximum.reduce([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ])
    # numpy 배열을 pandas Series로 변환
    tr_series = pd.Series(tr, index=df.index)
    df["atr14"] = tr_series.rolling(14).mean()
    
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_width"] = (df["bb_std"] * 2) / df["bb_mid"]

    # 갭 강세: 오늘 시가가 어제 종가 대비 몇 % 위?
    df["gap_up"] = df["open"] / df["close"].shift(1) - 1

    return df

def score_row(r):
    # 각 항목을 0~1 스케일로 간단 정규화 후 가중합
    s = 0.0
    # 강한 거래대금: 평균 대비 2배 이상이면 가점
    acc_ratio = np.clip((r.acc_price_ratio - 1.0) / 3.0, 0, 1)  # 1~4배 → 0~1
    s += 0.35 * acc_ratio

    # 모멘텀: 이동평균 상회 및 전고 대비 상단 위치
    pos_hhv = np.nan_to_num(r.pos_hhv, nan=0.0)
    s += 0.25 * pos_hhv
    s += 0.10 * r.ma_cross_up

    # 갭 강세가 과도하지 않은 범위에서 가점 (0~5% 구간)
    gap_u = np.clip(r.gap_up, 0, 0.05) / 0.05
    s += 0.15 * gap_u

    # 변동성/ATR 대비 본봉 몸통 크기 비율
    if r.atr14 and r.atr14 > 0:
        body = abs(r.close - r.open)
        body_ratio = np.clip(body / r.atr14, 0, 2) / 2
    else:
        body_ratio = 0
    s += 0.15 * body_ratio

    return s

def pick_candidates_daily(min_acc_price=1e9, lookback=120, top_k=10):
    markets = get_markets("KRW")["market"].tolist()
    rows = []

    for m in markets:
        df = get_candles_daily(m, count=lookback)
        if df is None or len(df) < 40:
            continue
        df = add_features(df)

        # 유효한 마지막 일자(가장 최근 종가가 확정된 캔들)의 스코어
        r = df.iloc[-1]
        # 필터: 거래대금 하한
        if r["acc_price"] < min_acc_price:
            continue

        score = score_row(r)
        rows.append({
            "market": m,
            "time": r["time"],
            "close": r["close"],
            "acc_price": r["acc_price"],
            "score": score,
            "gap_up": r["gap_up"],
            "acc_ratio": r["acc_price_ratio"]
        })
        time.sleep(0.06)  # rate limit 여유

    if not rows:
        return pd.DataFrame()

    dfc = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return dfc

def backtest_nextday_hit10p(market, lookback=240, min_acc_price=1e9):
    df = get_candles_daily(market, count=lookback+1)
    if df is None or len(df) < 60:
        return None
    df = add_features(df)

    trades = []
    # 마지막 캔들은 미래가 없으므로 -2까지, 그날 시그널 → 다음날 도달 여부 체크
    for i in range(40, len(df)-1):
        r = df.iloc[i]
        if r["acc_price"] < min_acc_price:
            continue
        score = score_row(r)
        # 임계치: 스코어 >= 0.6 예시
        if score >= 0.6:
            next_bar = df.iloc[i+1]
            # 다음 날 고가가 당일 종가 대비 +10% 도달?
            target = r["close"] * 1.10
            hit = next_bar["high"] >= target
            trades.append({
                "date": r["time"].date(),
                "close": r["close"],
                "score": round(score, 3),
                "next_open": next_bar["open"],
                "next_high": next_bar["high"],
                "hit10p": bool(hit),
                "max_ret_next": (next_bar["high"] / r["close"]) - 1.0
            })
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)

if __name__ == "__main__":
    # 1) 오늘 후보 상위 리스트
    cand = pick_candidates_daily(min_acc_price=2e9, lookback=200, top_k=12)
    print("Top candidates (daily):")
    print(cand)

    # 2) 임의의 종목 백테스트
    if len(cand) > 0:
        m = cand.loc[0, "market"]
        print(f"\nBacktest for {m}:")
        bt = backtest_nextday_hit10p(m, lookback=300, min_acc_price=2e9)
        if bt is not None and not bt.empty:
            hit_rate = bt["hit10p"].mean()
            avg_max = bt["max_ret_next"].mean()
            print(bt.tail(10))
            print(f"Hit rate(10% next-day high): {hit_rate:.2%}, avg max next-day return: {avg_max:.2%}")
        else:
            print("No signals in backtest window.")
