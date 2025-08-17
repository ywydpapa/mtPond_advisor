import asyncio
import requests
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import time
import datetime
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

def compute_stoch_rsi(series, window=14, smooth_k=3, smooth_d=3):
    # 1. RSI 먼저 계산
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # 2. StochRSI 계산
    min_rsi = rsi.rolling(window).min()
    max_rsi = rsi.rolling(window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    # 3. %K, %D smoothing
    stoch_k = stoch_rsi.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(smooth_d).mean()
    return stoch_rsi, stoch_k, stoch_d


def peak_trade(
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
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('candle_date_time_kst', inplace=True)
    df = df.sort_index(ascending=True)
    try:
        freq = pd.infer_freq(df.index)
        if freq:
            df.index = pd.DatetimeIndex(df.index, freq=freq)
    except Exception as e:
        print("freq 지정 실패:", e)

    df = df[['trade_price', 'candle_acc_trade_volume']]
    # 2. VWMA 및 MA 계산
    df[f'VWMA_{short_window}'] = (
            (df['trade_price'] * df['candle_acc_trade_volume'])
            .rolling(window=short_window).sum() /
            df['candle_acc_trade_volume'].rolling(window=short_window).sum()
    )
    df[f'VWMA_{long_window}'] = (
            (df['trade_price'] * df['candle_acc_trade_volume'])
            .rolling(window=long_window).sum() /
            df['candle_acc_trade_volume'].rolling(window=long_window).sum()
    )
    df[f'MA_{short_window}'] = df['trade_price'].rolling(window=short_window).mean()
    df[f'MA_{long_window}'] = df['trade_price'].rolling(window=long_window).mean()
    # === 상승/하강봉 평균 변화율 계산 추가 ===
    df['prev_price'] = df['trade_price'].shift(1)
    df['change'] = df['trade_price'] - df['prev_price']
    df['rate'] = (df['trade_price'] - df['prev_price']) / df['prev_price']
    up_candles = df[df['change'] > 0]
    down_candles = df[df['change'] < 0]
    avg_up_rate = up_candles['rate'].mean() * 100  # %
    avg_down_rate = down_candles['rate'].mean() * 100  # %
    print(f"상승봉 평균 상승률: {avg_up_rate:.3f}%")
    print(f"하강봉 평균 하강률: {avg_down_rate:.3f}%")
    # =====================================
    # 3. 크로스 포인트 계산
    golden_cross = df[
        (df[f'VWMA_{short_window}'] > df[f'VWMA_{long_window}']) &
        (df[f'VWMA_{short_window}'].shift(1) <= df[f'VWMA_{long_window}'].shift(1))
        ]
    dead_cross = df[
        (df[f'VWMA_{short_window}'] < df[f'VWMA_{long_window}']) &
        (df[f'VWMA_{short_window}'].shift(1) >= df[f'VWMA_{long_window}'].shift(1))
        ]
    # RSI 추가
    stoch_rsi, stoch_k, stoch_d = compute_stoch_rsi(df['trade_price'], window=14, smooth_k=3, smooth_d=3)
    df['stoch_rsi'] = stoch_rsi
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    # AI 신호예측 부분
    trsignal = ''
    try:
        freq = 'h' if 'h' in candle_unit else 'min'
        # future_price = predict_future_price(df, periods=3, freq=freq)
        future_price_arima = predict_future_price_arima(df, periods=3)
        future_price_xgb = predict_price_xgb(df, periods=3)
        now_price = df['trade_price'].iloc[-1]
        pred_rate = (future_price_xgb - now_price) / now_price * 100
        pred_rate_xgb = (future_price_xgb - now_price) / now_price * 100 if future_price_xgb is not None else None
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"현재가: {now_price:.2f}, 3캔들 뒤 예측가: {future_price_arima:.2f}")
        print(f"예측 변화율: {pred_rate_xgb:.3f}%")
        print(f"상승봉 평균 변화율: {avg_up_rate:.3f}%")
        print(f"하강봉 평균 변화율: {avg_down_rate:.3f}%")
        # 3. 비교 및 신호 판단
        if pred_rate_xgb > avg_up_rate:
            print("예측 변화율이 상승봉 평균 변화율보다 높음 → 강한 매수 신호!")
            trsignal = 'BUY'
            # 필요시 trguide = 'BUY'
        elif pred_rate_xgb <= avg_down_rate:
            print("예측 변화율이 하강봉 평균 변화율보다 낮음 → 강한 매도 신호!")
            trsignal = 'SELL'
            # 필요시 trguide = 'SELL'
        else:
            print("예측 변화율이 평균 변화율 범위 내 → 특별 신호 없음")
            trsignal = 'HOLD'
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    except Exception as e:
        print("예측 실패:", e)

    # 최근 크로스 판단
    recent = df.tail(5)
    now = df.index[-1]
    now_price = df['trade_price'].iloc[-1]
    golden_times = golden_cross.index[golden_cross.index <= now]
    dead_times = dead_cross.index[dead_cross.index <= now]
    last_golden = golden_times[-1] if len(golden_times) > 0 else None
    last_dead = dead_times[-1] if len(dead_times) > 0 else None
    if last_golden and last_dead:
        if last_golden > last_dead:
            last_cross_type = 'golden'
            last_cross_time = last_golden
        else:
            last_cross_type = 'dead'
            last_cross_time = last_dead
    elif last_golden:
        last_cross_type = 'golden'
        last_cross_time = last_golden
    elif last_dead:
        last_cross_type = 'dead'
        last_cross_time = last_dead
    else:
        last_cross_type = None
        last_cross_time = None
    recent5_idx = df.index[-5:]
    recent3_idx = df.index[-3:]
    recent_golden = [idx for idx in golden_cross.index if idx in recent3_idx]
    recent_dead = [idx for idx in dead_cross.index if idx in recent3_idx]
    recent_golden_5 = [idx for idx in golden_cross.index if idx in recent5_idx]
    recent_dead_5 = [idx for idx in dead_cross.index if idx in recent5_idx]
    trguide = "BHOLD"
    if recent_golden_5 and recent_dead_5:
        print("최근 5개 캔들에 골든/데드가 모두 있습니다. 매매 대기!")
        trguide = "HOLD"
        return trguide, 'goldanddead', 'HOLD', now_price
    if recent_golden:
        print("최근 3개 캔들에 골든크로스 발생! 매수 신호! 보유하고 있지 않다면 매수")
        now_price = df['trade_price'].iloc[-1]
        volum = 500000 / now_price
        print(f"매수 실행: {now_price}에 {volum:.6f}코인")
        trguide = "BUY"
        return trguide, 'golden', 'BUY', now_price
    if recent_dead:
        print("최근 3개 캔들에 데드크로스 발생! 매도 신호! 보유중인 코인 판매")
        now_price = df['trade_price'].iloc[-1]
        print(f"매도 실행: {now_price}에 보유코인 전량")
        trguide = "SELL"
        return trguide, 'dead', 'SELL', now_price
    if last_cross_type is not None:
        up_threshold = abs(avg_up_rate) * 2.5 / 100
        down_threshold = abs(avg_down_rate) * 2.5 / 100
        trend = analyze_cross_with_peak_and_vwma(
            df, last_cross_type, last_cross_time,
            short_window=short_window,
            long_window=long_window,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
            close_threshold=0.001
        )
    else:
        print("아직 골든/데드 크로스가 없습니다.")
        return trguide, 'nocross', trsignal, now_price
    return trguide, trend[1], trsignal, now_price


def analyze_cross_with_peak_and_vwma(
        df,
        last_cross_type,
        last_cross_time,
        short_window,
        long_window,
        up_threshold=0.015,
        down_threshold=0.015,
        close_threshold=0.001
):
    subtrguide = None
    # 1. 구간 결정
    if last_cross_type is not None and last_cross_time is not None:
        # 크로스가 있으면, 크로스 이후 데이터만 사용
        prices = df.loc[last_cross_time:]['trade_price']
        vwmashort = df.loc[last_cross_time:][f'VWMA_{short_window}']
        vwmalong = df.loc[last_cross_time:][f'VWMA_{long_window}']
        print(f"최근 크로스({last_cross_type})가 {last_cross_time}에 발생, 이후 데이터 기준으로 판단합니다.")
    else:
        # 크로스가 없으면 전체 데이터 사용
        prices = df['trade_price']
        vwmashort = df[f'VWMA_{short_window}']
        vwmalong = df[f'VWMA_{long_window}']
        print("최근 크로스가 없습니다. 전체 데이터에서 최고점/최저점 기준으로 판단합니다.")

    now_price = prices.iloc[-1]
    now_vwmalong = vwmalong.iloc[-1]
    vwma_gap = abs(now_price - now_vwmalong) / now_vwmalong

    # 꼭지점 찾기
    peak_indices, _ = find_peaks(prices)
    valley_indices, _ = find_peaks(-prices)

    if len(peak_indices) > 0:
        last_peak_time = prices.index[peak_indices[-1]]
        last_peak_value = prices.iloc[peak_indices[-1]]
        print(f"마지막 최고점: {last_peak_time} / {last_peak_value}")

    if len(valley_indices) > 0:
        last_valley_time = prices.index[valley_indices[-1]]
        last_valley_value = prices.iloc[valley_indices[-1]]
        print(f"마지막 최저점: {last_valley_time} / {last_valley_value}")

    # 최고점, 최저점 및 신호 판단
    max_price = prices.max()
    max_time = prices.idxmax()
    min_price = prices.min()
    min_time = prices.idxmin()
    fall_rate = (max_price - now_price) / max_price
    rise_rate = (now_price - min_price) / min_price
    print(f"최고가: {max_price:.2f} ({max_time}), 최저가: {min_price:.2f} ({min_time}), 현재가: {now_price:.2f}")
    print(f"최고가 대비 하락률: {fall_rate * 100:.2f}%")
    print(f"최저가 대비 상승률: {rise_rate * 100:.2f}%")
    subtrguide = "HOLD"
    # 신호 판단 (최고점/최저점 모두 체크)
    if fall_rate >= down_threshold:
        print(f"→ {down_threshold * 100:.1f}% 이상 하락! 매도 신호!")
        print(f"→최고가 대비  {down_threshold * 100:.1f}% 이상 하락으로 보유 코인 전액 현재가 {now_price} 매도 실행!")
        subtrguide = "SELL"
    elif vwma_gap <= close_threshold:
        print(f"→ 가격이 long VWMA({long_window})와 0.1% 이내로 접근! 추가 매도 신호!")
        print(f"→ 가격이 long VWMA({long_window})와 0.1% 이내로 접근 보유코인이 있을 경우 전액 현재가 {now_price} 매도 실행!")
        subtrguide = "SELL"
    else:
        print("→ 아직 매도 신호 아님(최고가 하락 미달, long VWMA 접근 미달)")
    if rise_rate >= up_threshold:
        print(f"→최저가 대비  {up_threshold * 100:.1f}% 이상 상승! 매수 신호!")
        print(f"현재가 {now_price}로 500,000원 매수")
        subtrguide = "BUY"
    elif vwma_gap <= close_threshold:
        print(f"→ 가격이 long VWMA({long_window})와 0.1% 이내로 접근! 추가 매수 신호!")
        print(f"보유 코인 없을 경우 현재가 {now_price}로 매수")
        subtrguide = "BUY"
    else:
        print("→ 아직 매수 신호 아님(최저가 상승 미달, long VWMA 접근 미달)")
    vwma_long_series = df.loc[last_cross_time:][f'VWMA_{long_window}']
    if len(vwma_long_series) >= 2:
        first_vwma = vwma_long_series.iloc[0]
        last_vwma = vwma_long_series.iloc[-1]
        delta = last_vwma - first_vwma
        if delta > 0:
            trend = "상승추세"
        elif delta < 0:
            trend = "하락추세"
        else:
            trend = "횡보"
        print(
            f"\n[추가분석] 크로스 이후 VWMA{long_window} 변화: {first_vwma:.2f} → {last_vwma:.2f} ({'+' if delta > 0 else ''}{delta:.2f})")
        print(f"[추가분석] 크로스 이후 VWMA{long_window}는 '{trend}'입니다.")
    else:
        print(f"[추가분석] VWMA{long_window} 데이터가 충분하지 않습니다.")
    return trend, subtrguide


async def get_wallet(uno):
    url = f'http://ywydpapa.iptime.org:8000/restwallet/{uno}'
    response = requests.get(url)
    data = response.json()
    return data


async def get_setup(uno):
    url = f'http://ywydpapa.iptime.org:8000/restsetup/{uno}'
    response = requests.get(url)
    data = response.json()
    setups = data['data']
    return setups


async def buy_crypto(ticker, uno):
    url = f'http://ywydpapa.iptime.org:8000/restbuymarket/{uno}/{ticker}'
    response = requests.get(url)
    return response


async def add_crypto(ticker, uno):
    url = f'http://ywydpapa.iptime.org:8000/restbuymarketadd/{uno}/{ticker}'
    response = requests.get(url)
    return response


async def sell_crypto(ticker, uno):
    url = f'http://ywydpapa.iptime.org:8000/restsellmarket/{uno}/{ticker}'
    response = requests.get(url)
    return response


async def cut_crypto(ticker, uno):
    url = f'http://ywydpapa.iptime.org:8000/restsellcut/{uno}/{ticker}'
    response = requests.get(url)
    return response


def predict_future_price(df, periods=3, freq='3min'):
    prophet_df = df.reset_index()[['candle_date_time_kst', 'trade_price']].rename(
        columns={'candle_date_time_kst': 'ds', 'trade_price': 'y'}
    )
    model = Prophet(daily_seasonality=False, yearly_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    future_price = forecast['yhat'].iloc[-periods:].mean()
    return future_price


def predict_future_price_arima(df, periods=3):
    # ARIMA는 시계열 index가 필요합니다.
    y = df['trade_price']
    # 간단한 (자동) 파라미터: (1,1,1) 또는 더 정교하게 pmdarima의 auto_arima로 최적화 가능
    try:
        model = ARIMA(y, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        future_price = forecast.mean()  # ARIMA는 미래값이 여러개면 평균을 사용
        return future_price
    except Exception as e:
        print("ARIMA 예측 실패:", e)
        return None


def predict_price_xgb(df, periods=3):
    import xgboost as xgb
    df['lag_1'] = df['trade_price'].shift(1)
    df['ma_3'] = df['trade_price'].rolling(3).mean()
    df = df.dropna()
    X = df[['lag_1', 'ma_3']]
    y = df['trade_price']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    pred = model.predict(X.tail(periods))
    return pred.mean()


async def get_trendCoins():
    url = 'http://127.0.0.1:8000/api/top30coins'
    response = requests.get(url)
    data = response.json()
    coins = data['markets']
    return coins


STOP_LOSS_RATE = -4.5  # 손절 설정 -4.5%


async def main_trade(uno):
    try:
        coins = await get_trendCoins()
        for coinn in coins:  # 설정 코인 순으로 진행
            print("로드된 설정 코인 :", coinn)
            wallets = await get_wallet(uno)
            walltems = wallets['data']['wallet_list']  # 지갑내 코인 로드
            curprice = wallets['data']['wallet_dict']
            remaincoin = 0
            avgprice = 0
            trade_state = ('BUY'
                           '')  # 기본값: 지갑에 없다고 가정
            found = False
            for witem in walltems:
                if coinn == witem[5]:
                    avgprice = witem[6]
                    remaincoin = witem[9]
                    found = True
                    if remaincoin > 0:
                        trade_state = 'SELL'
                    else:
                        trade_state = 'BUY'
                    break
            print("지갑내 코인 : ", remaincoin)
            print("매수 평균가 :", float(avgprice))
            print("매매 전략 :", trade_state)
            nowt = datetime.datetime.now()
            print('예측 시간 : ', nowt.strftime("%Y-%m-%d %H:%M:%S"))
            print("5m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~5m")
            short_position = peak_trade(coinn, 1, 20, 200, '5m')
            print("short_position", short_position)
            print('예측 시간 : ', nowt.strftime("%Y-%m-%d %H:%M:%S"))
            print("15m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~15m")
            mid_position = peak_trade(coinn, 1, 20, 200, '15m')
            print("mid_position", mid_position)
            print('예측 시간 : ', nowt.strftime("%Y-%m-%d %H:%M:%S"))
            print("3m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3m")
            test_position = peak_trade(coinn, 1, 20, 200, '3m')
            print("test_position", test_position)
            print('예측 시간 : ', nowt.strftime("%Y-%m-%d %H:%M:%S"))
            avg_price = float(avgprice)
            now_price = short_position[3]
            if avg_price == 0:
                loss_rate = 0  # 또는 적절한 기본값/에러 처리
            else:
                loss_rate = (now_price - avg_price) / avg_price * 100
            if trade_state == 'SELL' and short_position[2] == 'SELL':
                if now_price < avg_price:
                    if loss_rate <= STOP_LOSS_RATE:
                        print(f"[손절] 현재가({now_price})가 매수평균가({avg_price})보다 {loss_rate:.2f}% 낮음. 손절 실행!")
                        cut_response = await cut_crypto(coinn, uno)
                        print(cut_response)
                    else:
                        print(f"[대기] 현재가({now_price})가 매수평균가({avg_price}) 미만이지만, 손절 기준 미충족. 매도 대기.")
                elif now_price >= avg_price * 1.005:  # 익절 실행
                    print(f"[익절] 현재가({now_price})가 매수평균가({avg_price}) 이상. 매도 실행!")
                    sell_response = await sell_crypto(coinn, uno)
                    print(sell_response)
                else:
                    print("매도 대기 상태로 진행")
            elif trade_state == 'SELL' and short_position[2] == 'BUY':  # 상승으로 예상
                if now_price < avg_price:
                    if loss_rate < STOP_LOSS_RATE:
                        print(f"[추가매수] 현재가({now_price})가 매수평균가({avg_price})보다 {loss_rate:.2f}% 낮음. 상승 예상 추가매수 실행!")
                        add_response = await add_crypto(coinn, uno)
                        print(add_response)
                    else:
                        print(f"[대기] 현재가({now_price})가 매수평균가({avg_price}) 미만이지만, 손절 기준 미충족. 매도 대기.")
                elif now_price > avg_price * 1.005:  # 익절 실행
                    print(f"[추가매수] 현재가({now_price})가 매수평균가({avg_price}) 이상. 상승예상 추가 매수 실행!")
                    add_response = await add_crypto(coinn, uno)
                    print(add_response)
                else:
                    print("양쪽 도달 하지 않아 매도 대기 상태로 진행")
            elif trade_state == 'BUY' and short_position[2] == 'BUY':
                buy_response = await buy_crypto(coinn, uno)
                print(buy_response)
            elif trade_state == 'BUY' and short_position[2] == 'SELL':
                print('더 떨어질 것으로 예측, 매수 대기함')
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    except Exception as e:
        print(e)



