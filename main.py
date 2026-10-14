import os
import asyncio
import json
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import warnings
import re
import requests
import httpx
import websockets
import numpy as np
import pandas as pd
import time
import random
from typing import List, Optional
import math
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import dotenv

dotenv.load_dotenv()

# -------------------- DB 세팅 --------------------
DATABASE_URL = os.getenv("dburl")
engine = create_async_engine(DATABASE_URL, echo=True) if DATABASE_URL else None
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False) if engine else None

# -------------------- FastAPI 앱 --------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -------------------- 전역 상태 --------------------
trade_queues = defaultdict(lambda: {'BID': deque(maxlen=240), 'ASK': deque(maxlen=240)})
trade_counts = defaultdict(int)
krw_markets: List[str] = []
_collect_tasks_started = False

trend_cache = {}
cross_memory = {}

warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================== 다중 타임프레임 Bollinger Collector 설정 ==========================
TOP30_ENDPOINT = os.getenv("TOP30_ENDPOINT", "http://hitoff.iptime.org/api/top30coins")

BB_TIMEFRAMES = [3, 5, 15, 30]   # 분 단위
BB_CANDLE_COUNT = 200
BB_SEMAPHORE_LIMIT = 8
BB_LOOP_INTERVAL = 180        # 3분

BB_PERIOD = 20
BB_K = 2
HHLL_PERIOD = 20
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

bbtrend_cache = {
    "updated": None,
    "timeframes": {},
    "markets_used": [],
    "status": "initial"  # initial | ok | empty
}
_bb_collector_started = False
_FORCE_LOCK = asyncio.Lock()
_EMPTY_STREAK = 0
_EMPTY_FALLBACK_THRESHOLD = 3

# ===============================================================================================
# DB 종속 함수
# ===============================================================================================
async def get_db():
    if not async_session:
        yield None
        return
    async with async_session() as session:
        yield session

# ===============================================================================================
# 마켓 목록 수집
# ===============================================================================================
async def get_krw_markets() -> List[str]:
    url = "https://api.upbit.com/v1/market/all?isDetails=false"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        markets = [m['market'] for m in resp.json() if m.get('market', '').startswith("KRW-")]
        return markets

def chunk_markets(markets: List[str], size: int = 50):
    for i in range(0, len(markets), size):
        yield markets[i:i + size]

# ===============================================================================================
# 실시간 체결 웹소켓 수집
# ===============================================================================================
async def upbit_collector(markets_chunk: List[str]):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_fmt = [
        {"ticket": "mtPond"},
        {"type": "trade", "codes": markets_chunk},
    ]
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps(subscribe_fmt))
                while True:
                    data = await ws.recv()
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8")
                    trade = json.loads(data)

                    market = trade.get('code')
                    side = trade.get('ask_bid')
                    if not market or side not in ('BID', 'ASK'):
                        continue

                    volume = float(trade.get('trade_volume', 0.0) or 0.0)
                    price = float(trade.get('trade_price', 0.0) or 0.0)
                    amount = volume * price
                    trade_queues[market][side].append(amount)
                    trade_counts[market] += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(1.0)

async def reset_trade_counts():
    while True:
        now = datetime.now()
        if now.minute < 30:
            next_reset = now.replace(minute=30, second=0, microsecond=0)
        else:
            next_reset = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_seconds = (next_reset - now).total_seconds()
        await asyncio.sleep(sleep_seconds)
        for market in list(trade_counts.keys()):
            trade_counts[market] = 0

# ===============================================================================================
# 보조 지표 계산 함수
# ===============================================================================================
def calculate_advanced_signals(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14, atr_period: int = 14) -> dict:
    if len(df) < max(bb_period, rsi_period, atr_period) + 10:
        return None

    close = df['trade_price']
    high = df['high_price']
    low = df['low_price']
    vol = df['candle_acc_trade_volume']

    # 1. 볼린저 밴드
    ma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std(ddof=0)
    upper = ma + (std * bb_std)
    lower = ma - (std * bb_std)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    band_width = (upper - lower) / ma.replace(0, np.nan)

    # 2. 거래량 폭증 (직전 20봉 평균 대비 200% 이상)
    vol_ma20 = vol.rolling(20).mean()
    curr_vol = vol.iloc[-1]
    prev_vol_avg = vol_ma20.iloc[-2]
    volume_ratio = (curr_vol / prev_vol_avg) if prev_vol_avg > 0 else 1.0
    is_volume_spike = volume_ratio >= 2.0

    # 3. RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 4. RSI 약세 다이버전스
    is_bearish_div = False
    if high.iloc[-1] >= high.tail(15).max() * 0.99:
        prev_rsi_peak = rsi.tail(15).iloc[:-2].max()
        if pd.notna(prev_rsi_peak) and (rsi.iloc[-1] < prev_rsi_peak - 3.0):
            is_bearish_div = True

    # 5. ATR 및 손익비(Risk/Reward Ratio)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    curr_close = close.iloc[-1]
    curr_upper = upper.iloc[-1]
    curr_atr = atr.iloc[-1] if pd.notna(atr.iloc[-1]) else 0.0

    expected_profit = max(0.0, curr_upper - curr_close)
    expected_risk = max(curr_close * 0.005, curr_atr * 1.5)
    rr_ratio = expected_profit / expected_risk if expected_risk > 0 else 0.0

    # 6. 진입 시그널 판정
    pb_now, pb_prev = pct_b.iloc[-1], pct_b.iloc[-2]
    min_bw_20 = band_width.tail(20).min()
    is_squeeze = band_width.iloc[-1] <= (min_bw_20 * 1.15) if pd.notna(min_bw_20) else False
    bw_expanding = band_width.iloc[-1] > band_width.iloc[-2] if pd.notna(band_width.iloc[-1]) else False

    signal_status = "HOLD"
    reasons = []

    if is_squeeze and bw_expanding and (pb_now >= 0.8):
        if is_volume_spike and not is_bearish_div:
            signal_status = "STRONG_BUY"
            reasons.append("스퀴즈 탈출 + 거래량 폭증")
        elif not is_volume_spike:
            reasons.append("가짜 돌파 의심 (거래량 부족)")
    elif pb_prev < 0.5 and pb_now >= 0.5:
        if not is_bearish_div:
            if rr_ratio >= 1.5:
                signal_status = "BUY"
                reasons.append(f"중심선 돌파 (손익비 {rr_ratio:.2f})")
            else:
                reasons.append(f"손익비 미달 ({rr_ratio:.2f} < 1.5)")

    if is_bearish_div and pb_now >= 0.8:
        signal_status = "WARNING_SELL"
        reasons.append("RSI 약세 다이버전스 (고점 이탈 주의)")

    return {
        "price": curr_close,
        "pct_b": round(float(pb_now) * 100, 1) if pd.notna(pb_now) else 50.0,
        "volume_ratio": round(float(volume_ratio) * 100, 1),
        "is_vol_spike": is_volume_spike,
        "is_bearish_div": is_bearish_div,
        "atr": round(float(curr_atr), 2),
        "rr_ratio": round(float(rr_ratio), 2),
        "signal": signal_status,
        "reason": ", ".join(reasons) if reasons else "특이 패턴 없음"
    }

async def fetch_top30_from_api(max_retry: int = 3, retry_delay: float = 1.0) -> list:
    for attempt in range(max_retry):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(TOP30_ENDPOINT)
                if r.status_code == 200:
                    data = r.json()
                    return data.get("markets", [])
        except Exception as e:
            print(f"[fetch_top30_from_api] attempt={attempt+1} error={e}")
        await asyncio.sleep(retry_delay)
    return []

async def fetch_candles_async(client: httpx.AsyncClient, market: str, unit: int, count: int) -> pd.DataFrame:
    url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    params = {"market": market, "count": count}
    r = await client.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()
    rows = []
    for item in reversed(data):
        rows.append({
            "dt": item["candle_date_time_kst"],
            "open": item["opening_price"],
            "high": item["high_price"],
            "low": item["low_price"],
            "close": item["trade_price"],
            "volume": item["candle_acc_trade_volume"],
            "timestamp": item["timestamp"]
        })
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df.set_index("dt", inplace=True)
    return df

def evaluate_coin_recommendation(market: str, tf_results: dict) -> dict:
    daily = tf_results.get("1d")
    h4 = tf_results.get("4h")
    m30 = tf_results.get("30m")
    m10 = tf_results.get("10m")

    if not (daily and h4 and m30 and m10):
        return None

    if daily["pct_b"] < 40.0 and h4["pct_b"] < 40.0:
        return None

    if m10["is_bearish_div"] or m30["is_bearish_div"]:
        return None

    trigger_tf = m30 if m30["signal"] in ("BUY", "STRONG_BUY") else (m10 if m10["signal"] in ("BUY", "STRONG_BUY") else None)
    if not trigger_tf:
        return None

    if trigger_tf["rr_ratio"] < 1.5:
        return None

    score = (trigger_tf["volume_ratio"] / 100.0) * 2.0 + trigger_tf["rr_ratio"]

    return {
        "market": market,
        "current_price": trigger_tf["price"],
        "recommend_type": trigger_tf["signal"],
        "trigger_timeframe": "30m" if trigger_tf == m30 else "10m",
        "volume_surge": f"{trigger_tf['volume_ratio']}%",
        "rr_ratio": f"1:{trigger_tf['rr_ratio']}",
        "reason": trigger_tf["reason"],
        "priority_score": round(score, 2)
    }

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if len(out) >= BB_PERIOD:
        out["MA"] = out["close"].rolling(BB_PERIOD).mean()
        out["STD"] = out["close"].rolling(BB_PERIOD).std(ddof=0)
        out["Upper"] = out["MA"] + BB_K * out["STD"]
        out["Lower"] = out["MA"] - BB_K * out["STD"]
        denom = BB_K * out["STD"]
        out["BB_Pos"] = (out["close"] - out["MA"]) / denom * 100
        out.loc[(denom == 0) | denom.isna(), "BB_Pos"] = 0.0
        bw_denom = out["MA"]
        out["BandWidth"] = (out["Upper"] - out["Lower"]) / bw_denom * 100
        out.loc[(bw_denom == 0) | bw_denom.isna(), "BandWidth"] = 0.0
    else:
        out["MA"] = np.nan
        out["STD"] = np.nan
        out["Upper"] = np.nan
        out["Lower"] = np.nan
        out["BB_Pos"] = 0.0
        out["BandWidth"] = 0.0

    out["HH20"] = out["high"].rolling(HHLL_PERIOD).max()
    out["LL20"] = out["low"].rolling(HHLL_PERIOD).min()
    span = out["HH20"] - out["LL20"]
    out["Range_Pos20"] = (out["close"] - out["LL20"]) / span * 100
    out.loc[(span == 0) | span.isna(), "Range_Pos20"] = 0.0

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI"] = 100 - (100 / (1 + rs))
    out.loc[avg_loss == 0, "RSI"] = 100
    out.loc[(avg_gain == 0) & (avg_loss > 0), "RSI"] = 0

    prev_close = out["close"].shift(1)
    tr_components = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    out["ATR"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_Signal"] = out["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    out = _add_trend10_for_collector(out, window=10)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    core_numeric_cols = [
        "close", "BB_Pos", "BandWidth", "Range_Pos20",
        "RSI", "ATR", "MACD", "MACD_Signal", "MACD_Hist"
    ]
    core_non_numeric_cols = ["Trend10"]
    exist_num = [c for c in core_numeric_cols if c in out.columns]
    if exist_num:
        for c in exist_num:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out[exist_num] = out[exist_num].fillna(0.0)

    for c in core_non_numeric_cols:
        if c in out.columns:
            out[c] = out[c].where(out[c].notna(), None)
    return out

def _add_trend10_for_collector(df: pd.DataFrame, window: int = 10, doji_char: str = "-") -> pd.DataFrame:
    out = df.copy()
    if not {"open", "close"}.issubset(out.columns):
        out["Trend10"] = None
        return out
    out["_mid"] = (out["open"] + out["close"]) / 2.0
    patterns = [None] * len(out)
    for i in range(window - 1, len(out)):
        win = out.iloc[i - window + 1: i + 1]
        avg_mid = win["_mid"].mean()
        chars = []
        for _, row in win.iterrows():
            o = row["open"]; c = row["close"]
            m = (o + c) / 2.0
            high_flag = m >= avg_mid
            if c > o:
                chars.append("U" if high_flag else "u")
            elif c < o:
                chars.append("D" if high_flag else "d")
            else:
                chars.append(doji_char)
        patterns[i] = "".join(chars)
    out["Trend10"] = patterns
    out.drop(columns=["_mid"], inplace=True, errors="ignore")
    return out

def _sanitize_value(v, use_null=False):
    if isinstance(v, (float, int)):
        if math.isnan(v) or math.isinf(v):
            return None if use_null else 0.0
        return float(v)
    return v

def _sanitize_record(d: dict, use_null=False):
    for k, v in d.items():
        if isinstance(v, (float, int)):
            if math.isnan(v) or math.isinf(v):
                d[k] = None if use_null else 0.0
    return d

async def collect_bb_for_timeframe(client: httpx.AsyncClient, markets: list, unit: int, sem: asyncio.Semaphore):
    results = []
    for market in markets:
        async with sem:
            try:
                df = await fetch_candles_async(client, market, unit, BB_CANDLE_COUNT)
            except Exception:
                await asyncio.sleep(0.1)
                continue
        if df.empty:
            continue
        df_ind = compute_indicators(df)
        last = df_ind.iloc[-1]
        record = {
            "market": market,
            "close": _sanitize_value(last.get("close")),
            "BB_Pos": _sanitize_value(last.get("BB_Pos")),
            "BandWidth": _sanitize_value(last.get("BandWidth")),
            "Range_Pos20": _sanitize_value(last.get("Range_Pos20")),
            "RSI": _sanitize_value(last.get("RSI")),
            "ATR": _sanitize_value(last.get("ATR")),
            "MACD": _sanitize_value(last.get("MACD")),
            "MACD_Signal": _sanitize_value(last.get("MACD_Signal")),
            "MACD_Hist": _sanitize_value(last.get("MACD_Hist")),
            "Trend10": last.get("Trend10", None),
            "volume": _sanitize_value(last.get("volume", None)),
            "time": last.name.isoformat()
        }
        record = _sanitize_record(record)
        results.append(record)
        await asyncio.sleep(0.02)
    results.sort(key=lambda x: x["BB_Pos"], reverse=True)
    return results

async def _run_bb_collection_once():
    global bbtrend_cache, _EMPTY_STREAK

    markets = await fetch_top30_from_api()
    if not markets:
        _EMPTY_STREAK += 1
        if _EMPTY_STREAK >= _EMPTY_FALLBACK_THRESHOLD:
            markets = krw_markets[:30]
        else:
            bbtrend_cache.update({
                "updated": datetime.now(timezone.utc).isoformat(),
                "timeframes": {},
                "markets_used": [],
                "status": "empty"
            })
            return
    else:
        _EMPTY_STREAK = 0

    sem = asyncio.Semaphore(BB_SEMAPHORE_LIMIT)
    async with httpx.AsyncClient(timeout=10) as client:
        tf_results = {}
        for unit in BB_TIMEFRAMES:
            data = await collect_bb_for_timeframe(client, markets, unit, sem)
            tf_results[f"{unit}m"] = data

    status_flag = "ok" if any(tf_results.values()) else "empty"
    bbtrend_cache.update({
        "updated": datetime.now(timezone.utc).isoformat(),
        "timeframes": tf_results,
        "markets_used": markets,
        "status": status_flag
    })

async def bb_collector_loop():
    await asyncio.sleep(5)
    while True:
        try:
            await _run_bb_collection_once()
        except Exception as e:
            print(f"[bb_collector_loop] error: {e}")
        await asyncio.sleep(BB_LOOP_INTERVAL)

def start_bb_collector_once():
    global _bb_collector_started
    if not _bb_collector_started:
        asyncio.create_task(bb_collector_loop())
        _bb_collector_started = True

# ===============================================================================================
# VWMA 크로스 관련 (trend_loop)
# ===============================================================================================
async def get_trendcoins():
    url = 'http://becog.iptime.org:8088/api/top30coins'
    def fetch():
        s = requests.Session()
        s.trust_env = False
        return s.get(url, timeout=5, proxies={"http": None, "https": None})
    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(None, fetch)
        response.raise_for_status()
        data = response.json()
        coins = data.get('markets', [])
        return coins if isinstance(coins, list) else []
    except Exception as e:
        print(f"get_trendcoins: 요청 실패 {e}")
        return []

RL_RE = re.compile(r"group=(\w+); *min=(\d+); *sec=(\d+)")

def parse_remaining_req(header_value: str):
    if not header_value:
        return None
    m = RL_RE.search(header_value)
    if not m:
        return None
    group, min_cnt, sec_cnt = m.groups()
    return {"group": group, "min": int(min_cnt), "sec": int(sec_cnt)}

def polite_get(url, max_retry=8, base_sleep=0.12, jitter=0.05):
    for attempt in range(1, max_retry+1):
        resp = requests.get(url)
        rem_header = resp.headers.get("Remaining-Req")
        rem = parse_remaining_req(rem_header)

        if resp.status_code == 200:
            if rem and rem["sec"] <= 1:
                time.sleep(0.25 + random.uniform(0, 0.05))
            return resp

        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            wait = float(ra) if ra else min(2.0, base_sleep * (2 ** (attempt - 1)) + random.uniform(0, jitter))
            if rem and rem["sec"] == 0:
                wait = max(wait, 0.6)
            print(f"[WARN] 429 too_many_requests attempt={attempt} waiting {wait:.3f}s rem={rem}")
            time.sleep(wait)
            continue

        resp.raise_for_status()

    raise RuntimeError("429 재시도 초과")

async def peak_trade(ticker='KRW-BTC', short_window=3, long_window=20, count=180, candle_unit='1h'):
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
    if candle_unit not in candle_map:
        raise ValueError(f"지원하지 않는 단위: {candle_unit}")
    api_type, minute = candle_map[candle_unit]

    url = f'https://api.upbit.com/v1/candles/days?market={ticker}&count={count}' if api_type == 'days' else f'https://api.upbit.com/v1/candles/minutes/{minute}?market={ticker}&count={count}'
    resp = polite_get(url)
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"JSON 파싱 실패 status={resp.status_code} text={resp.text[:200]}")

    if isinstance(data, dict) and 'error' in data:
        raise RuntimeError(f"Upbit API 오류: {data['error']}")
    if not isinstance(data, list) or len(data) == 0:
        cross_memory[ticker] = {'error': '빈 데이터', 'ticker': ticker}
        return

    df = pd.DataFrame(data)
    idx = pd.to_datetime(df['candle_date_time_utc'], format='%Y-%m-%dT%H:%M:%S', utc=True).dt.tz_convert('Asia/Seoul')
    df.index = idx
    df.index.name = 'candle_time_kst'
    df = df.sort_index()

    need_len = max(short_window, long_window)
    if len(df) < need_len:
        cross_memory[ticker] = {'error': '데이터 길이 부족', 'len': len(df), 'need': need_len}
        return

    sw_col, lw_col = f'VWMA_{short_window}', f'VWMA_{long_window}'
    df[sw_col] = (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=short_window).sum() / df['candle_acc_trade_volume'].rolling(window=short_window).sum()
    df[lw_col] = (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=long_window).sum() / df['candle_acc_trade_volume'].rolling(window=long_window).sum()
    df['vwma_diff'] = df[sw_col] - df[lw_col]

    valid = df.dropna(subset=['vwma_diff'])
    if valid.empty:
        cross_memory[ticker] = {'error': 'VWMA NaN 지속', 'ticker': ticker}
        return

    golden = valid[(valid['vwma_diff'].shift(1) < 0) & (valid['vwma_diff'] > 0)].copy()
    golden['cross_type'] = 'golden'
    dead = valid[(valid['vwma_diff'].shift(1) > 0) & (valid['vwma_diff'] < 0)].copy()
    dead['cross_type'] = 'dead'

    crosses = pd.concat([golden, dead]).sort_index()
    last_2 = crosses.tail(2)
    now = pd.Timestamp.now(tz='Asia/Seoul')

    if last_2.empty:
        cross_memory[ticker] = {'last_2_crosses': [], 'latest_cross_type': None, 'note': '교차 없음', 'updated': now}
        return

    latest_cross_time = last_2.index[-1]
    elapsed = now - latest_cross_time

    if len(df) >= 6 and not df[sw_col].iloc[-2:].isna().any():
        sw_dir = 'UP' if df[sw_col].iloc[-1] > df[sw_col].iloc[-2] else 'DN'
        lw_dir = 'UP' if df[lw_col].iloc[-1] > df[lw_col].iloc[-2] else 'DN'
        sw_slope = df[sw_col].iloc[-1] - df[sw_col].iloc[-2]
        lw_now = df[lw_col].iloc[-1]
        lw_mean6 = df[lw_col].iloc[-6:].mean()
        lw_slope = lw_now - lw_mean6
        sw_angle = np.degrees(np.arctan(sw_slope))
        lw_angle = np.degrees(np.arctan(lw_slope))
        vol_slope = df['candle_acc_trade_volume'].iloc[-1] - df['candle_acc_trade_volume'].iloc[-2]
        vol_angle = np.degrees(np.arctan(vol_slope))
    else:
        sw_dir = lw_dir = 'N/A'
        sw_angle = lw_angle = vol_angle = None

    cross_memory[ticker] = {
        'last_2_crosses': [{'type': r['cross_type'], 'time': t} for t, r in last_2.iterrows()],
        'latest_cross_type': last_2.iloc[-1]['cross_type'],
        'latest_cross_time': latest_cross_time,
        'now': now,
        'elapsed': elapsed,
        sw_col: df[sw_col].iloc[-1],
        lw_col: df[lw_col].iloc[-1],
        f'{sw_col}_dir': sw_dir,
        f'{lw_col}_dir': lw_dir,
        f'{sw_col}_angle': sw_angle,
        f'{lw_col}_angle': lw_angle,
        'volume': df['candle_acc_trade_volume'].iloc[-1],
        'volume_angle': vol_angle
    }

async def trend_loop():
    while True:
        coins = await get_trendcoins()
        if not coins:
            await asyncio.sleep(5)
            continue
        for coin in coins:
            try:
                await peak_trade(ticker=coin, candle_unit='1m')
            except Exception as e:
                print(f"peak_trade 실패({coin}): {e}")
            await asyncio.sleep(0.5)
        await asyncio.sleep(15)

async def get_current_prices():
    server_url = "https://api.upbit.com"
    params = {"quote_currencies": "KRW"}
    res = requests.get(server_url + "/v1/ticker/all", params=params)
    data = res.json()
    result = []
    for item in data:
        market = item.get("market")
        trade_price = item.get("trade_price")
        if market and trade_price:
            result.append({"market": market, "trade_price": trade_price})
    return result

# ===============================================================================================
# 패턴 분석 전역 캐시 & 체결 건수 함수
# ===============================================================================================
BB_ANALYSIS_TIMEFRAMES = [
    {"key": "10m", "type": "minutes", "unit": 10},
    {"key": "30m", "type": "minutes", "unit": 30},
    {"key": "1h", "type": "minutes", "unit": 60},
    {"key": "4h", "type": "minutes", "unit": 240},
    {"key": "1d", "type": "days", "unit": None},
]

BB_PATTERN_INTERVAL = 600
BB_ANALYSIS_SEMAPHORE_LIMIT = 5

bb_pattern_cache = {
    "updated": None,
    "status": "initial",
    "total_markets": 0,
    "data": []
}
_bb_pattern_lock = asyncio.Lock()
_bb_pattern_collector_started = False

async def fetch_pattern_candles_async(client: httpx.AsyncClient, market: str, candle_type: str, count: int = 60, unit: int = None) -> pd.DataFrame:
    base_url = "https://api.upbit.com/v1/candles"
    url = f"{base_url}/days" if candle_type == "days" else f"{base_url}/minutes/{unit}"
    params = {"market": market, "count": count}

    try:
        r = await client.get(url, params=params, timeout=8)
        if r.status_code == 200:
            data = r.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
            num_cols = ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
            return df
    except Exception:
        pass
    return pd.DataFrame()

async def get_trade_activity(market: str, client: httpx.AsyncClient):
    """
    최근 500개 체결 틱을 기준으로:
    1) 직전 10분 체결 건수
    2) 직전 1시간 체결 건수
    3) 마지막 체결 경과 시간(초)
    """
    url = f"https://api.upbit.com/v1/trades/ticks?market={market}&count=500"
    try:
        res = await client.get(url, timeout=5.0)
        if res.status_code != 200:
            return {"trade_count_10m": 0, "trade_count_1h": 0, "last_trade_sec": 9999, "activity_level": "cold"}

        ticks = res.json()
        if not ticks:
            return {"trade_count_10m": 0, "trade_count_1h": 0, "last_trade_sec": 9999, "activity_level": "cold"}

        now_utc = datetime.now(timezone.utc)
        ten_min_ago = now_utc - timedelta(minutes=10)
        one_hour_ago = now_utc - timedelta(hours=1)

        c_10m = 0
        c_1h = 0

        for t in ticks:
            dt_str = f"{t['trade_date_utc']} {t['trade_time_utc']}"
            t_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if t_dt >= ten_min_ago:
                c_10m += 1
            if t_dt >= one_hour_ago:
                c_1h += 1

        latest_str = f"{ticks[0]['trade_date_utc']} {ticks[0]['trade_time_utc']}"
        latest_dt = datetime.strptime(latest_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        last_trade_sec = int((now_utc - latest_dt).total_seconds())

        if c_10m >= 60:
            activity_level = "hot"
        elif c_10m >= 10:
            activity_level = "normal"
        else:
            activity_level = "cold"

        return {
            "trade_count_10m": c_10m,
            "trade_count_1h": c_1h,
            "last_trade_sec": last_trade_sec,
            "activity_level": activity_level
        }
    except Exception:
        return {"trade_count_10m": 0, "trade_count_1h": 0, "last_trade_sec": 9999, "activity_level": "cold"}

def analyze_tf_pattern(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14, atr_period: int = 14) -> dict:
    if len(df) < max(bb_period, rsi_period, atr_period) + 10:
        return None

    close = df['trade_price']
    high = df['high_price']
    low = df['low_price']
    vol = df['candle_acc_trade_volume']

    ma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std(ddof=0)
    upper = ma + (std * bb_std)
    lower = ma - (std * bb_std)

    band_denom = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / band_denom
    band_width = (upper - lower) / ma.replace(0, np.nan)

    c_now, c_prev = close.iloc[-1], close.iloc[-2]
    o_now = df['opening_price'].iloc[-1]
    pb_now = pct_b.iloc[-1]
    pb_prev = pct_b.iloc[-2]
    ma_now, ma_prev = ma.iloc[-1], ma.iloc[-2]
    bw_now = band_width.iloc[-1]

    min_bw_20 = band_width.tail(20).min()
    is_squeeze = (bw_now <= (min_bw_20 * 1.15)) if pd.notna(min_bw_20) else False
    bw_expanding = (bw_now > band_width.iloc[-2]) if pd.notna(bw_now) else False

    vol_ma20 = vol.rolling(20).mean()
    curr_vol = vol.iloc[-1]
    prev_vol_avg = vol_ma20.iloc[-2]
    volume_ratio = (curr_vol / prev_vol_avg) if prev_vol_avg > 0 else 1.0
    is_volume_spike = bool(volume_ratio >= 2.0)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    is_bearish_div = False
    if high.iloc[-1] >= high.tail(15).max() * 0.99:
        prev_rsi_peak = rsi.tail(15).iloc[:-2].max()
        if pd.notna(prev_rsi_peak) and (rsi.iloc[-1] < prev_rsi_peak - 3.0):
            is_bearish_div = True

    prev_close_s = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close_s).abs(),
        (low - prev_close_s).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    curr_close = close.iloc[-1]
    curr_upper = upper.iloc[-1]
    curr_atr = atr.iloc[-1] if pd.notna(atr.iloc[-1]) else 0.0

    expected_profit = max(0.0, curr_upper - curr_close)
    expected_risk = max(curr_close * 0.005, curr_atr * 1.5)
    rr_ratio = expected_profit / expected_risk if expected_risk > 0 else 0.0

    pattern = "횡보 / 방향 탐색"
    bias = "중립"
    score = 0
    signal_status = "HOLD"
    reasons = []

    if pb_now >= 0.85 and pb_prev >= 0.85 and ma_now > ma_prev:
        pattern = "상승 밴드 라이딩"
        bias = "상승 지속"
        score = 2
    elif is_squeeze and bw_expanding and (pb_now > 0.8 and pb_prev <= 0.8):
        pattern = "스퀴즈 후 상단 돌파"
        bias = "강력 상승 기대"
        score = 3
    elif pb_prev >= 1.0 and pb_now < 0.9 and (c_now < o_now):
        pattern = "상단 저항 피뢰침 이탈"
        bias = "단기 조정/하락 주의"
        score = -2
    elif pb_prev <= 0.1 and pb_now > 0.2 and (c_now > o_now):
        pattern = "하단 지지 양봉 반등"
        bias = "기술적 반등 기대"
        score = 2
    elif pb_prev < 0.5 and pb_now >= 0.5:
        pattern = "중심선 상향 돌파"
        bias = "상승 전환"
        score = 1
    elif pb_prev > 0.5 and pb_now <= 0.5:
        pattern = "중심선 하향 이탈"
        bias = "하락 전환"
        score = -1
    elif pb_now <= 0.15 and pb_prev <= 0.15 and ma_now < ma_prev:
        pattern = "하락 밴드 라이딩"
        bias = "하락 추세 지속"
        score = -3

    if is_squeeze and bw_expanding and (pb_now >= 0.8):
        if is_volume_spike and not is_bearish_div:
            signal_status = "STRONG_BUY"
            reasons.append("스퀴즈 탈출 + 거래량 폭증")
        elif not is_volume_spike:
            reasons.append("가짜 돌파 의심 (거래량 부족)")
    elif pb_prev < 0.5 and pb_now >= 0.5:
        if not is_bearish_div:
            if rr_ratio >= 1.5:
                signal_status = "BUY"
                reasons.append(f"중심선 돌파 (손익비 {rr_ratio:.2f})")
            else:
                reasons.append(f"손익비 미달 ({rr_ratio:.2f} < 1.5)")

    if is_bearish_div and pb_now >= 0.8:
        signal_status = "WARNING_SELL"
        reasons.append("RSI 약세 다이버전스 (고점 이탈 주의)")

    candle_ranges = high - low
    avg_volatility_pct = (candle_ranges / df['opening_price'] * 100).tail(14).mean()

    return {
        "price": curr_close,
        "pct_b": round(float(pb_now) * 100, 1) if pd.notna(pb_now) else 50.0,
        "pattern": pattern,
        "bias": bias,
        "score": score,
        "bandwidth_expanding": bool(bw_expanding),
        "expected_volatility_pct": round(float(avg_volatility_pct), 2) if pd.notna(avg_volatility_pct) else 0.0,
        "volume_ratio": round(float(volume_ratio) * 100, 1),
        "is_vol_spike": is_volume_spike,
        "is_bearish_div": is_bearish_div,
        "atr": round(float(curr_atr), 2),
        "rr_ratio": round(float(rr_ratio), 2),
        "signal": signal_status,
        "reason": ", ".join(reasons) if reasons else "특이 패턴 없음"
    }

async def analyze_single_market(client: httpx.AsyncClient, market: str, sem: asyncio.Semaphore) -> dict:
    tf_data = {}
    total_score = 0
    current_price = 0.0

    for tf in BB_ANALYSIS_TIMEFRAMES:
        async with sem:
            df = await fetch_pattern_candles_async(
                client, market, candle_type=tf["type"], count=60, unit=tf["unit"]
            )
            await asyncio.sleep(0.08)

        if not df.empty:
            if current_price == 0.0:
                current_price = float(df['trade_price'].iloc[-1])
            res = analyze_tf_pattern(df)
            if res:
                tf_data[tf["key"]] = res
                weight = 2 if tf["key"] in ("1d", "4h") else 1
                total_score += res["score"] * weight

    if not tf_data:
        return None

    recommendation = evaluate_coin_recommendation(market, tf_data)
    # [체결 활동도 데이터 결합]
    trade_activity = await get_trade_activity(market, client)

    return {
        "market": market,
        "price": current_price,
        "overall_score": total_score,
        "trade_activity": trade_activity,
        "recommendation": recommendation,
        "timeframes": tf_data
    }

async def run_market_pattern_collector_once():
    global bb_pattern_cache
    if not krw_markets:
        return

    sem = asyncio.Semaphore(BB_ANALYSIS_SEMAPHORE_LIMIT)
    all_results = []

    async with httpx.AsyncClient(timeout=10) as client:
        for market in krw_markets:
            try:
                res = await analyze_single_market(client, market, sem)
                if res:
                    all_results.append(res)
            except Exception as e:
                print(f"[bb_pattern] market {market} 분석 실패: {e}")

    all_results.sort(
        key=lambda x: (
            x["overall_score"],
            1 if x.get("recommendation") else 0,
            x["recommendation"]["priority_score"] if x.get("recommendation") else 0
        ),
        reverse=True
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    async with _bb_pattern_lock:
        bb_pattern_cache = {
            "updated": now_str,
            "status": "ok",
            "total_markets": len(all_results),
            "recommended_count": len([x for x in all_results if x.get("recommendation")]),
            "data": all_results
        }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] KRW {len(all_results)}개 코인 분석 완료 (추천 통과: {bb_pattern_cache['recommended_count']}개)")

async def bb_pattern_collector_loop():
    await asyncio.sleep(5)
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 초기 전체 코인 패턴 수집 시작")
        await run_market_pattern_collector_once()
    except Exception as e:
        print(f"[bb_pattern_collector_loop Initial Error]: {e}")

    while True:
        now = datetime.now()
        # 10분 정각 스케줄링 계산
        minute_offset = 10 - (now.minute % 10)
        target_time = (now + timedelta(minutes=minute_offset)).replace(second=2, microsecond=0)
        sleep_seconds = (target_time - now).total_seconds()

        print(f"[{now.strftime('%H:%M:%S')}] 다음 수집 예정 시각: {target_time.strftime('%H:%M:%S')} (약 {int(sleep_seconds)}초 대기)")
        await asyncio.sleep(sleep_seconds)

        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 10분 정각 정기 수집 시작")
            await run_market_pattern_collector_once()
        except Exception as e:
            print(f"[bb_pattern_collector_loop Error]: {e}")

# ===============================================================================================
# 통합 Lifespan 관리
# ===============================================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global krw_markets, _collect_tasks_started, _bb_pattern_collector_started
    if not _collect_tasks_started:
        krw_markets = await get_krw_markets()
        for chunk in chunk_markets(krw_markets, 50):
            asyncio.create_task(upbit_collector(chunk))
        asyncio.create_task(reset_trade_counts())
        asyncio.create_task(trend_loop())
        start_bb_collector_once()

        if not _bb_pattern_collector_started:
            asyncio.create_task(bb_pattern_collector_loop())
            _bb_pattern_collector_started = True

        _collect_tasks_started = True
    yield

app.router.lifespan_context = lifespan

# ===============================================================================================
# 라우터 엔드포인트
# ===============================================================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "markets": krw_markets})

@app.get("/alltrend", response_class=HTMLResponse)
async def alltrend(request: Request):
    return templates.TemplateResponse("alltrend.html", {"request": request, "markets": krw_markets})

@app.get("/top30trend", response_class=HTMLResponse)
async def t30trend(request: Request):
    return templates.TemplateResponse("top30trend.html", {"request": request, "markets": krw_markets})

@app.get("/bbtrend30", response_class=HTMLResponse)
async def bbtrend30(request: Request):
    return templates.TemplateResponse("trendbb30.html", {"request": request, "markets": krw_markets})

@app.post("/api/bb-patterns/force-update")
async def force_bb_patterns_update():
    asyncio.create_task(run_market_pattern_collector_once())
    return {
        "status": "triggered",
        "message": "백그라운드에서 전체 코인 강제 재계산이 시작되었습니다."
    }

@app.get("/api/bb-patterns")
async def get_bb_patterns(
    limit: Optional[int] = 150,
    min_score: Optional[int] = None,
    only_recommended: Optional[str] = "false",
    activity_filter: Optional[str] = None  # 예: 'hot', 'normal', 'cold' 필터 지원 (선택사항)
):
    if bb_pattern_cache.get("status") == "initial":
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "message": "서버 시작 후 최초 수집 작업이 진행 중입니다. 잠시 후 다시 호출해 주세요."
            }
        )

    is_only_rec = str(only_recommended).strip().lower() in ("true", "1", "t", "y", "yes")

    data = list(bb_pattern_cache.get("data", []))

    if is_only_rec:
        data = [
            item for item in data
            if item.get("recommendation") is not None and isinstance(item.get("recommendation"), dict)
        ]

    if min_score is not None:
        data = [item for item in data if item.get("overall_score", 0) >= min_score]

    if activity_filter:
        act_key = activity_filter.strip().lower()
        data = [
            item for item in data
            if item.get("trade_activity", {}).get("activity_level") == act_key
        ]

    result_data = data[:limit] if limit is not None else data

    return {
        "status": bb_pattern_cache.get("status"),
        "updated": bb_pattern_cache.get("updated"),
        "total_analyzed": bb_pattern_cache.get("total_markets", 0),
        "recommended_count": bb_pattern_cache.get("recommended_count", 0),
        "result_count": len(result_data),
        "coins": result_data
    }


@app.get("/bb-patterns", response_class=HTMLResponse)
async def aiget_bb_patterns(
    request: Request,
    limit: Optional[int] = 150,
    min_score: Optional[int] = None,
    only_recommended: Optional[str] = "false"
):
    if bb_pattern_cache["status"] == "initial":
        return templates.TemplateResponse(
            "bb_patterns.html",
            {
                "request": request,
                "status": "initial",
                "message": "서버 시작 후 최초 수집 작업이 진행 중입니다. 잠시 후 새로고침해 주세요.",
                "coins": [],
                "updated": None,
                "total_analyzed": 0,
                "recommended_count": 0,
                "only_recommended": False
            }
        )

    is_only_rec = str(only_recommended).strip().lower() in ("true", "1", "t", "y", "yes")
    data = list(bb_pattern_cache.get("data", []))

    if is_only_rec:
        data = [
            item for item in data
            if item.get("recommendation") is not None and isinstance(item.get("recommendation"), dict)
        ]

    if min_score is not None:
        data = [item for item in data if item.get("overall_score", 0) >= min_score]

    coins_to_render = data[:limit] if limit is not None else data

    return templates.TemplateResponse(
        "bb_patterns.html",
        {
            "request": request,
            "status": bb_pattern_cache.get("status"),
            "updated": bb_pattern_cache.get("updated"),
            "total_analyzed": bb_pattern_cache.get("total_markets", 0),
            "recommended_count": bb_pattern_cache.get("recommended_count", 0),
            "coins": coins_to_render,
            "min_score": min_score,
            "limit": limit,
            "only_recommended": is_only_rec
        }
    )

# ===============================================================================================
# WebSocket 및 기타 API
# ===============================================================================================
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
                bid10 = round(sum(list(market_data['BID'])[-10:]), 0)
                ask10 = round(sum(list(market_data['ASK'])[-10:]), 0)
                bid30 = round(sum(list(market_data['BID'])[-30:]), 0)
                ask30 = round(sum(list(market_data['ASK'])[-30:]), 0)
                bid120 = round(sum(list(market_data['BID'])[-120:]), 0)
                ask120 = round(sum(list(market_data['ASK'])[-120:]), 0)
                bid240 = round(sum(list(market_data['BID'])[-240:]), 0)
                ask240 = round(sum(list(market_data['ASK'])[-240:]), 0)
                ratio10 = round((bid10 / ask10) * 100, 1) if ask10 > 0 else 0.0
                ratio30 = round((bid30 / ask30) * 100, 1) if ask30 > 0 else 0.0
                ratio120 = round((bid120 / ask120) * 100, 1) if ask120 > 0 else 0.0
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
                ratio10 = round((bid10 / ask10) * 100, 1) if ask10 > 0 else 0.0
                ratio30 = round((bid30 / ask30) * 100, 1) if ask30 > 0 else 0.0
                ratio120 = round((bid120 / ask120) * 100, 1) if ask120 > 0 else 0.0
                ratio240 = round((bid240 / ask240) * 100, 1) if ask240 > 0 else 0.0
                speed = trade_counts.get(market, 0)
                if bid30 > 10_000_000:
                    result.append({
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
                    })
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
        if bid30 > 5_000_000:
            result.append(market)
    result.sort(key=lambda m: trade_counts.get(m, 0), reverse=True)
    return JSONResponse(content={"markets": result[:30]})

@app.get("/api/aitrendt30")
async def get_all_trends():
    return JSONResponse(content=jsonable_encoder(cross_memory))

@app.get("/api/bbtrend30")
async def api_bbtrend30(tf: str = None):
    if not bbtrend_cache["updated"]:
        return {"status": "initial", "message": "첫 수집 전입니다. 잠시 후 다시 시도하세요."}
    if bbtrend_cache["status"] == "empty" and not bbtrend_cache["timeframes"]:
        return {"status": "empty", "message": "현재 /api/top30coins 결과가 없거나 캔들 수집 불가.", "updated": bbtrend_cache["updated"]}
    if tf:
        if tf not in bbtrend_cache["timeframes"]:
            return {"status": "error", "message": f"지원하지 않는 타임프레임: {tf}", "available": list(bbtrend_cache["timeframes"].keys())}
        return {
            "status": bbtrend_cache["status"],
            "updated": bbtrend_cache["updated"],
            "markets_used": bbtrend_cache["markets_used"],
            "timeframe": tf,
            "data": bbtrend_cache["timeframes"][tf]
        }
    return {
        "status": bbtrend_cache["status"],
        "updated": bbtrend_cache["updated"],
        "markets_used": bbtrend_cache["markets_used"],
        "timeframes": bbtrend_cache["timeframes"]
    }

@app.post("/api/bbtrend30/force")
async def api_bbtrend30_force():
    async with _FORCE_LOCK:
        await _run_bb_collection_once()
    return {
        "status": bbtrend_cache["status"],
        "updated": bbtrend_cache["updated"],
        "markets_used": bbtrend_cache["markets_used"]
    }

@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("/privacy/privacy.html", {"request": request})