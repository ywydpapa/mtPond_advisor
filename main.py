import time

from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
import asyncio
import websockets
import json
from collections import defaultdict, deque
import httpx
from typing import List
import datetime
from fastapi.responses import JSONResponse
from scipy.signal import find_peaks
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from lightgbm import LGBMRegressor
import xgboost as xgb
import pandas as pd
import requests
from contextlib import asynccontextmanager
import warnings


@asynccontextmanager
async def lifespan(app: FastAPI):
    global krw_markets, _collect_tasks_started
    if not _collect_tasks_started:
        krw_markets = await get_krw_markets()
        for chunk in chunk_markets(krw_markets, 50):
            asyncio.create_task(upbit_collector(chunk))
        asyncio.create_task(reset_trade_counts())
        asyncio.create_task(update_trends())
        _collect_tasks_started = True
    yield
    # (종료시 정리 코드 필요시 여기에)

warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.")
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
trade_queues = defaultdict(lambda: {'BID': deque(maxlen=240), 'ASK': deque(maxlen=240)})
krw_markets = []
_collect_tasks_started = False
trade_counts = defaultdict(int)
trend_cache = {}



# 전체 코인 코드 가져오기
async def get_krw_markets() -> List[str]:
    url = "https://api.upbit.com/v1/market/all?isDetails=false"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        markets = [m['market'] for m in resp.json() if m.get('market', '').startswith("KRW-")]
        return markets


# 50개씩 마켓 나누기
def chunk_markets(markets: List[str], size: int = 50):
    for i in range(0, len(markets), size):
        yield markets[i:i + size]


async def upbit_collector(markets_chunk: List[str]):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_fmt = [
        {"ticket": "mtPond"},
        {"type": "trade", "codes": markets_chunk},
    ]
    # 재연결 루프
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps(subscribe_fmt))
                while True:
                    data = await ws.recv()  # Upbit는 바이너리 JSON을 보냄
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8")
                    trade = json.loads(data)

                    market = trade.get('code')
                    side = trade.get('ask_bid')  # 'BID' or 'ASK'
                    # 필수 필드 검증
                    if not market or side not in ('BID', 'ASK'):
                        continue

                    volume = float(trade.get('trade_volume', 0.0) or 0.0)
                    price = float(trade.get('trade_price', 0.0) or 0.0)
                    amount = volume * price
                    trade_queues[market][side].append(amount)
                    trade_counts[market] += 1
        except asyncio.CancelledError:
            # 서버 종료 시 안전하게 종료
            raise
        except Exception:
            # 네트워크/프로토콜 오류 시 잠시 대기 후 재시도
            await asyncio.sleep(1.0)


async def reset_trade_counts():
    while True:
        now = datetime.datetime.now()
        if now.minute < 30:
            next_reset = now.replace(minute=30, second=0, microsecond=0)
        else:
            next_reset = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_seconds = (next_reset - now).total_seconds()
        await asyncio.sleep(sleep_seconds)
        for market in trade_counts.keys():
            trade_counts[market] = 0


def compute_stoch_rsi(series, window=14, smooth_k=3, smooth_d=3):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    min_rsi = rsi.rolling(window).min()
    max_rsi = rsi.rolling(window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    stoch_k = stoch_rsi.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(smooth_d).mean()
    return stoch_rsi, stoch_k, stoch_d


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
    if last_cross_type is not None and last_cross_time is not None:
        prices = df.loc[last_cross_time:]['trade_price']
        vwmashort = df.loc[last_cross_time:][f'VWMA_{short_window}']
        vwmalong = df.loc[last_cross_time:][f'VWMA_{long_window}']
        # print(f"최근 크로스({last_cross_type})가 {last_cross_time}에 발생, 이후 데이터 기준으로 판단합니다.")
    else:
        prices = df['trade_price']
        vwmashort = df[f'VWMA_{short_window}']
        vwmalong = df[f'VWMA_{long_window}']
        # print("최근 크로스가 없습니다. 전체 데이터에서 최고점/최저점 기준으로 판단합니다.")
    now_price = prices.iloc[-1]
    now_vwmalong = vwmalong.iloc[-1]
    vwma_gap = abs(now_price - now_vwmalong) / now_vwmalong
    peak_indices, _ = find_peaks(prices)
    valley_indices, _ = find_peaks(-prices)
    if len(peak_indices) > 0:
        last_peak_time = prices.index[peak_indices[-1]]
        last_peak_value = prices.iloc[peak_indices[-1]]
        # print(f"마지막 최고점: {last_peak_time} / {last_peak_value}")
    if len(valley_indices) > 0:
        last_valley_time = prices.index[valley_indices[-1]]
        last_valley_value = prices.iloc[valley_indices[-1]]
        # print(f"마지막 최저점: {last_valley_time} / {last_valley_value}")
    max_price = prices.max()
    max_time = prices.idxmax()
    min_price = prices.min()
    min_time = prices.idxmin()
    fall_rate = (max_price - now_price) / max_price
    rise_rate = (now_price - min_price) / min_price
    # print(f"최고가: {max_price:.2f} ({max_time}), 최저가: {min_price:.2f} ({min_time}), 현재가: {now_price:.2f}")
    # print(f"최고가 대비 하락률: {fall_rate * 100:.2f}%")
    # print(f"최저가 대비 상승률: {rise_rate * 100:.2f}%")
    subtrguide = "HOLD"
    # 신호 판단 (최고점/최저점 모두 체크)
    if fall_rate >= down_threshold:
        # print(f"→ {down_threshold * 100:.1f}% 이상 하락! 매도 신호!")
        # print(f"→최고가 대비  {down_threshold * 100:.1f}% 이상 하락으로 보유 코인 전액 현재가 {now_price} 매도 실행!")
        subtrguide = "SELL"
    elif vwma_gap <= close_threshold:
        # print(f"→ 가격이 long VWMA({long_window})와 0.1% 이내로 접근! 추가 매도 신호!")
        # print(f"→ 가격이 long VWMA({long_window})   와 0.1% 이내로 접근 보유코인이 있을 경우 전액 현재가 {now_price} 매도 실행!")
        subtrguide = "SELL"
    else:
        # print("→ 아직 매도 신호 아님(최고가 하락 미달, long VWMA 접근 미달)")
        pass
    if rise_rate >= up_threshold:
        # print(f"→최저가 대비  {up_threshold * 100:.1f}% 이상 상승! 매수 신호!")
        # print(f"현재가 {now_price}로 500,000원 매수")
        subtrguide = "BUY"
    elif vwma_gap <= close_threshold:
        # print(f"→ 가격이 long VWMA({long_window})와 0.1% 이내로 접근! 추가 매수 신호!")
        # print(f"보유 코인 없을 경우 현재가 {now_price}로 매수")
        subtrguide = "BUY"
    else:
        # print("→ 아직 매수 신호 아님(최저가 상승 미달, long VWMA 접근 미달)")
        pass
    vwma_long_series = df.loc[last_cross_time:][f'VWMA_{long_window}']
    if len(vwma_long_series) >= 2:
        first_vwma = vwma_long_series.iloc[0]
        last_vwma = vwma_long_series.iloc[-1]
        delta = last_vwma - first_vwma
        if delta > 0:
            trend = "UPTREND"
        elif delta < 0:
            trend = "DOWNTREND"
        else:
            trend = "EVENTREND"
        # print(f"\n[추가분석] 크로스 이후 VWMA{long_window} 변화: {first_vwma:.2f} → {last_vwma:.2f} ({'+' if delta > 0 else ''}{delta:.2f})")
        # print(f"[추가분석] 크로스 이후 VWMA{long_window}는 '{trend}'입니다.")
    else:
        pass
        # print(f"[추가분석] VWMA{long_window} 데이터가 충분하지 않습니다.")
    return trend, subtrguide


def predict_future_price(df, periods=3, freq='3min'):
    try:
        prophet_df = df.reset_index()[['candle_date_time_kst', 'trade_price']].rename(columns={'candle_date_time_kst': 'ds', 'trade_price': 'y'})
        model = Prophet(daily_seasonality=False, yearly_seasonality=False)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        future_price = forecast['yhat'].iloc[-periods:].mean()
        return future_price
    except Exception as e:
        print("가격예측 실패:", e)
        return None


def predict_future_price_arima(df, periods=3):
    y = df['trade_price']
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
    try:
        df['lag_1'] = df['trade_price'].shift(1)
        df['ma_3'] = df['trade_price'].rolling(3).mean()
        df = df.dropna()
        X = df[['lag_1', 'ma_3']]
        y = df['trade_price']
        model = xgb.XGBRegressor()
        model.fit(X, y)
        pred = model.predict(X.tail(periods))
        return pred.mean()
    except Exception as e:
        print("XGB 예측실패:",e)
        return None


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
        else:
            df.index = pd.DatetimeIndex(df.index, freq='h')
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
    # print(f"상승봉 평균 상승률: {avg_up_rate:.3f}%")
    # print(f"하강봉 평균 하강률: {avg_down_rate:.3f}%")
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
        future_price = predict_future_price(df, periods=3, freq=freq)
        future_price_arima = predict_future_price_arima(df, periods=3)
        future_price_xgb = predict_price_xgb(df, periods=3)
        now_price = df['trade_price'].iloc[-1]
        pred_rate = (future_price_xgb - now_price) / now_price * 100
        pred_rate_xgb = (future_price_xgb - now_price) / now_price * 100 if future_price_xgb is not None else None
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(f"현재가: {now_price:.2f}, 3캔들 뒤 예측가: {future_price_arima:.2f}")
        # print(f"예측 변화율: {pred_rate_xgb:.3f}%")
        # print(f"상승봉 평균 변화율: {avg_up_rate:.3f}%")
        # print(f"하강봉 평균 변화율: {avg_down_rate:.3f}%")
        # 3. 비교 및 신호 판단
        if pred_rate_xgb > avg_up_rate:
            # print("예측 변화율이 상승봉 평균 변화율보다 높음 → 강한 매수 신호!")
            trsignal = 'BUY-SIGNAL'
            # 필요시 trguide = 'BUY'
        elif pred_rate_xgb <= avg_down_rate:
            # print("예측 변화율이 하강봉 평균 변화율보다 낮음 → 강한 매도 신호!")
            trsignal = 'SELL-SIGNAL'
            # 필요시 trguide = 'SELL'
        else:
            # print("예측 변화율이 평균 변화율 범위 내 → 특별 신호 없음")
            trsignal = 'HOLD-SIGNAL'
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
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
    trguide = "INITV"
    vwmatrend = None
    if last_cross_type is not None:
        up_threshold = abs(avg_up_rate) * 2.5 / 100
        down_threshold = abs(avg_down_rate) * 2.5 / 100
        vwmatrend = analyze_cross_with_peak_and_vwma(
            df, last_cross_type, last_cross_time,
            short_window=short_window,
            long_window=long_window,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
            close_threshold=0.001
        )
    else:
        # print("아직 골든/데드 크로스가 없습니다.")
        trguide = "HOLD"
        return trguide, 'NOCROSS', now_price, trsignal, avg_up_rate, avg_down_rate, vwmatrend
    if recent_golden_5 and recent_dead_5:
        # print("최근 5개 캔들에 골든/데드가 모두 있습니다. 매매 대기!")
        trguide = "HOLD"
        return trguide, last_cross_type, now_price, trsignal, avg_up_rate, avg_down_rate, vwmatrend
    if recent_golden:
        # print("최근 3개 캔들에 골든크로스 발생! 매수 신호! 보유하고 있지 않다면 매수")
        now_price = df['trade_price'].iloc[-1]
        # volum = 500000 / now_price
        # print(f"매수 실행: {now_price}에 {volum:.6f}코인")
        trguide = "BUY"
        return trguide, last_cross_type, now_price, trsignal, avg_up_rate, avg_down_rate, vwmatrend
    if recent_dead:
        # print("최근 3개 캔들에 데드크로스 발생! 매도 신호! 보유중인 코인 판매")
        now_price = df['trade_price'].iloc[-1]
        # print(f"매도 실행: {now_price}에 보유코인 전량")
        trguide = "SELL"
        return trguide, last_cross_type, now_price, trsignal, avg_up_rate, avg_down_rate, vwmatrend
    return None


async def get_suggestcoins():
    result = []
    try:
        for market in krw_markets:
            market_data = trade_queues.get(market)
            if not market_data:
                continue
            bid30 = round(sum(list(market_data['BID'])[-30:]), 0)
            if bid30 > 10000000:
                result.append(market)
        # speed 기준 정렬
        result.sort(key=lambda m: trade_counts.get(m, 0), reverse=True)
        top30 = result[:30]
        return top30
    except Exception as e:
        print("30 목록 에러:",e)
        return None


async def update_trends():
    await asyncio.sleep(15)
    while True:
        coins = await get_suggestcoins()  # 코인 리스트 가져오기
        for coin in coins:
            try:
                # 여러 단위(5m, 15m, 3m)로 트렌드 계산
                short_position = peak_trade(coin, 1, 20, 200, '5m')
                time.sleep(0.2)
                mid_position = peak_trade(coin, 1, 20, 200, '15m')
                time.sleep(0.2)
                test_position = peak_trade(coin, 1, 20, 200, '3m')
                time.sleep(0.2)
                trend_cache[coin] = {
                    '5m': short_position,
                    '15m': mid_position,
                    '3m': test_position,
                    'updated': datetime.datetime.now().isoformat()
                }
            except Exception as e:
                print(f"{coin} 트렌드 계산 실패:", e)
        await asyncio.sleep(60)  # 1분마다 갱신


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # 마켓명 리스트 전달
    return templates.TemplateResponse("index.html", {"request": request, "markets": krw_markets})


@app.get("/alltrend", response_class=HTMLResponse)
async def alltrend(request: Request):
    # startup_event는 앱 시작 시 한 번만 실행됨 (중복 실행 방지)
    return templates.TemplateResponse("alltrend.html", {"request": request, "markets": krw_markets})

@app.get("/top30trend", response_class=HTMLResponse)
async def t30trend(request: Request):
    # 마켓 리스트는 필요하면 전달, 실제로는 JS에서 데이터 받아옴
    return templates.TemplateResponse("top30trend.html", {"request": request, "markets": krw_markets})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            grid = []
            for market in krw_markets:
                market_data = trade_queues.get(market)
                if not market_data:
                    continue
                # 누적합
                bid10 = round(sum(list(market_data['BID'])[-10:]), 0)
                ask10 = round(sum(list(market_data['ASK'])[-10:]), 0)
                bid30 = round(sum(list(market_data['BID'])[-30:]), 0)
                ask30 = round(sum(list(market_data['ASK'])[-30:]), 0)
                bid120 = round(sum(list(market_data['BID'])[-120:]), 0)
                ask120 = round(sum(list(market_data['ASK'])[-120:]), 0)
                bid240 = round(sum(list(market_data['BID'])[-240:]), 0)
                ask240 = round(sum(list(market_data['ASK'])[-240:]), 0)
                # 비율
                ratio10 = round((bid10/ask10)*100, 1) if ask10 > 0 else 0.0
                ratio30 = round((bid30/ask30)*100, 1) if ask30 > 0 else 0.0
                ratio120 = round((bid120/ask120)*100, 1) if ask120 > 0 else 0.0
                ratio240 = round((bid240 / ask240) * 100, 1) if ask240 > 0 else 0.0
                # 체결속도(예시: 1초마다 1씩 증가, 1분마다 리셋)
                speed = trade_counts.get(market, 0)
                row = {
                    "market": market,
                    "speed": speed,
                    "bid10": bid10,
                    "ask10": ask10,
                    "bid30": bid30,
                    "ask30": ask30,
                    "bid120": bid120,
                    "ask120": ask120,
                    "bid240": bid240,
                    "ask240": ask240,
                    "ratio10": ratio10,
                    "ratio30": ratio30,
                    "ratio120": ratio120,
                    "ratio240": ratio240,
                }
                grid.append(row)
            await websocket.send_json(grid)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    except Exception:
        pass

@app.websocket("/ws/top30trend")
async def websocket_top30trend(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            result = []
            for market in krw_markets:
                market_data = trade_queues.get(market)
                if not market_data:
                    continue
                bid10 = round(sum(list(market_data['BID'])[-10:]), 0)
                ask10 = round(sum(list(market_data['ASK'])[-10:]), 0)
                bid30 = round(sum(list(market_data['BID'])[-30:]), 0)
                ask30 = round(sum(list(market_data['ASK'])[-30:]), 0)
                bid120 = round(sum(list(market_data['BID'])[-120:]), 0)
                ask120 = round(sum(list(market_data['ASK'])[-120:]), 0)
                bid240 = round(sum(list(market_data['BID'])[-240:]), 0)
                ask240 = round(sum(list(market_data['ASK'])[-240:]), 0)
                ratio10 = round((bid10/ask10)*100, 1) if ask10 > 0 else 0.0
                ratio30 = round((bid30/ask30)*100, 1) if ask30 > 0 else 0.0
                ratio120 = round((bid120/ask120)*100, 1) if ask120 > 0 else 0.0
                ratio240 = round((bid240 / ask240) * 100, 1) if ask240 > 0 else 0.0
                speed = trade_counts.get(market, 0)
                row = {
                    "market": market,
                    "speed": speed,
                    "bid10": bid10,
                    "ask10": ask10,
                    "bid30": bid30,
                    "ask30": ask30,
                    "bid120": bid120,
                    "ask120": ask120,
                    "bid240": bid240,
                    "ask240": ask240,
                    "ratio10": ratio10,
                    "ratio30": ratio30,
                    "ratio120": ratio120,
                    "ratio240": ratio240,
                }
                # 30개의 매수금액 합이 천만원 넘는 것만 리스트에 넣음
                if bid30 > 10000000 :
                    result.append(row)
            result.sort(key=lambda x: x['speed'], reverse=True)
            top30 = result[:30]
            await websocket.send_json({
                "count": len(top30),
                "coins": top30
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

@app.get("/api/top30coins")
async def get_top30coins():
    result = []
    for market in krw_markets:
        market_data = trade_queues.get(market)
        if not market_data:
            continue
        bid30 = round(sum(list(market_data['BID'])[-30:]), 0)
        if bid30 > 10000000:
            result.append(market)
    # speed 기준 정렬
    result.sort(key=lambda m: trade_counts.get(m, 0), reverse=True)
    top30 = result[:30]
    return JSONResponse(content={"markets": top30})

@app.get("/api/aitrend/{coin}")
async def get_trend(coin: str):
    result = trend_cache.get(coin)
    if result:
        return JSONResponse(content={"coin": coin, "trend": result})
    else:
        return JSONResponse(content={"error": "대상 코인 정보 없음"}, status_code=404)

@app.get("/api/aitrendall")
async def get_all_trends():
    return JSONResponse(content=trend_cache)