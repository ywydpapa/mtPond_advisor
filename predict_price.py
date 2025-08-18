import asyncio
import requests
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import time
import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# 코인별 결과를 저장하는 글로벌 딕셔너리
cross_memory = {}

async def peak_trade(
        ticker='KRW-BTC',
        short_window=3,
        long_window=20,
        count=180,
        candle_unit='1h'
):
    # 0. 캔들 단위 변환
    candle_map = {
        '1d': ('days', ''),
        '4h': ('minutes', 240),
        '1h': ('minutes', 60),
        '30m': ('minutes', 30),
        '15m': ('minutes', 15),
        '10m': ('minutes', 10),
        '5m': ('minutes', 5),
        '3m': ('minutes', 3),
        '1m': ('minutes', 1),
    }
    global trguide
    if candle_unit not in candle_map:
        raise ValueError(f"지원하지 않는 단위입니다: {candle_unit}")
    api_type, minute = candle_map[candle_unit]
    if api_type == 'days':
        url = f'https://api.upbit.com/v1/candles/days?market={ticker}&count={count}'
    else:
        url = f'https://api.upbit.com/v1/candles/minutes/{minute}?market={ticker}&count={count}'
    # 1. 데이터 가져오기
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)

    # 타임존 일관화 처리
    if 'candle_date_time_utc' in df.columns:
        idx = pd.to_datetime(
            df['candle_date_time_utc'],
            format='%Y-%m-%dT%H:%M:%S',
            utc=True
        ).dt.tz_convert('Asia/Seoul')
        df.set_index(idx, inplace=True)
        df.index.name = 'candle_time_kst'
    else:
        # Upbit KST 문자열은 타임존 정보가 없으므로 KST로 로컬라이즈
        idx = pd.to_datetime(
            df['candle_date_time_kst'],
            format='%Y-%m-%dT%H:%M:%S'
        ).dt.tz_localize('Asia/Seoul')
        df.set_index(idx, inplace=True)
        df.index.name = 'candle_time_kst'

    df = df.sort_index(ascending=True)

    # VWMA 계산 (3, 20)
    df['VWMA_3'] = (
            (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=3).sum() /
            df['candle_acc_trade_volume'].rolling(window=3).sum()
    )
    df['VWMA_20'] = (
            (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=20).sum() /
            df['candle_acc_trade_volume'].rolling(window=20).sum()
    )
    # VWMA 차이 계산
    df['vwma_diff'] = df['VWMA_3'] - df['VWMA_20']

    # 골든크로스와 데드크로스 찾기
    golden_cross = df[(df['vwma_diff'].shift(1) < 0) & (df['vwma_diff'] > 0)].copy()
    golden_cross['cross_type'] = 'golden'
    dead_cross = df[(df['vwma_diff'].shift(1) > 0) & (df['vwma_diff'] < 0)].copy()
    dead_cross['cross_type'] = 'dead'

    # 합치고 시간순 정렬
    crosses = pd.concat([golden_cross, dead_cross]).sort_index()
    last_2_crosses = crosses.tail(2)

    # 교차 신호가 없는 경우 방어 처리
    if last_2_crosses.empty:
        print("교차 신호가 없어 경과 시간을 계산할 수 없습니다.")
        return

    # 최종 크로스 이후 경과시간 계산 (둘 다 tz-aware)
    latest_cross_time = last_2_crosses.index[-1]
    now = pd.Timestamp.now(tz='Asia/Seoul')  # 한국표준시 기준
    elapsed = now - latest_cross_time

    # VWMA 방향(상승/하강) 및 기울기(변화량) 판별

    if len(df) >= 2:
        vwma3_dir = '상승' if df['VWMA_3'].iloc[-1] > df['VWMA_3'].iloc[-2] else '하강'
        vwma20_dir = '상승' if df['VWMA_20'].iloc[-1] > df['VWMA_20'].iloc[-2] else '하강'
        vwma3_slope = df['VWMA_3'].iloc[-1] - df['VWMA_3'].iloc[-2]
        vwma20_slope = df['VWMA_20'].iloc[-1] - df['VWMA_20'].iloc[-2]
        volume_slope = df['candle_acc_trade_volume'].iloc[-1] - df['candle_acc_trade_volume'].iloc[-2]
        # 각도 변환
        vwma3_angle = np.degrees(np.arctan(vwma3_slope))
        vwma20_angle = np.degrees(np.arctan(vwma20_slope))
        volume_angle = np.degrees(np.arctan(volume_slope))
    else:
        vwma3_dir = vwma20_dir = 'N/A'
        vwma3_slope = vwma20_slope = volume_slope = None
        vwma3_angle = vwma20_angle = volume_angle = None

    cross_memory[ticker] = {
        'last_2_crosses': [
            {
                'type': row['cross_type'],
                'time': idx
            } for idx, row in last_2_crosses.iterrows()
        ],
        'latest_cross_type': last_2_crosses.iloc[-1]['cross_type'],
        'latest_cross_time': latest_cross_time,
        'now': now,
        'elapsed': elapsed,
        'VWMA_3': df['VWMA_3'].iloc[-1],
        'VWMA_20': df['VWMA_20'].iloc[-1],
        'VWMA_3_dir': vwma3_dir,
        'VWMA_20_dir': vwma20_dir,
        'VWMA_3_angle': vwma3_angle,
        'VWMA_20_angle': vwma20_angle,
        'volume': df['candle_acc_trade_volume'].iloc[-1],
        'volume_angle': volume_angle
    }
    # print(f"\n대상 코인:",ticker)
    # print(f"최종 크로스 종류: {last_2_crosses.iloc[-1]['cross_type']}")
    # print(f"최종 크로스 발생 시간: {latest_cross_time}")
    # print(f"현재 시간: {now}")
    # print(f"최종 크로스 이후 경과시간: {elapsed}")
    # print(f"VWMA_3: {df['VWMA_3'].iloc[-1]:.2f} ({vwma3_dir}, 기울기: {vwma3_angle:.2f})")
    # print(f"VWMA_20: {df['VWMA_20'].iloc[-1]:.2f} ({vwma20_dir}, 기울기: {vwma20_angle:.2f})")
    # print(f"거래량 기울기: {volume_angle:.2f}")

async def get_trendcoins():
    url = 'http://ywydpapa.iptime.org:8000/api/top30coins'
    response = requests.get(url)
    data = response.json()
    coins = data['markets']
    return coins

async def main_loop():
    while True:
        coins = await get_trendcoins()
        print(f"코인 리스트: {coins}")
        # 각 코인에 대해 순차적으로 0.5초 간격 처리
        for coin in coins:
            await peak_trade(ticker=coin, candle_unit='3m')
            await asyncio.sleep(0.5)
        print("1분 대기...")
        print(cross_memory)
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main_loop())