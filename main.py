import os
import asyncio
import json
import datetime
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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from typing import List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import math
from datetime import datetime as _dt
import dotenv

dotenv.load_dotenv()

# -------------------- DB 세팅 --------------------
DATABASE_URL = os.getenv("dburl")
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

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
TOP30_ENDPOINT = os.getenv("TOP30_ENDPOINT", "http://127.0.0.1:8000/api/top30coins")

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
        now = datetime.datetime.now()
        if now.minute < 30:
            next_reset = now.replace(minute=30, second=0, microsecond=0)
        else:
            next_reset = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_seconds = (next_reset - now).total_seconds()
        await asyncio.sleep(sleep_seconds)
        for market in list(trade_counts.keys()):
            trade_counts[market] = 0

# ===============================================================================================
# Top30 외부(내부 API) 기반 수집 함수
# ===============================================================================================
async def fetch_top30_from_api(max_retry: int = 3, retry_delay: float = 1.0) -> list:
    for attempt in range(max_retry):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(TOP30_ENDPOINT)
                if r.status_code == 200:
                    data = r.json()
                    markets = data.get("markets", [])
                    return markets
        except Exception as e:
            print(f"[fetch_top30_from_api] attempt={attempt+1} error={e}")
        await asyncio.sleep(retry_delay)
    return []

# ===============================================================================================
# 지표 계산 관련
# ===============================================================================================
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

# --- (기존 전체 코드 중 compute_indicators 함수만 교체하는 패치) ---
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    # Bollinger
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
    # HH/LL
    out["HH20"] = out["high"].rolling(HHLL_PERIOD).max()
    out["LL20"] = out["low"].rolling(HHLL_PERIOD).min()
    span = out["HH20"] - out["LL20"]
    out["Range_Pos20"] = (out["close"] - out["LL20"]) / span * 100
    out.loc[(span == 0) | span.isna(), "Range_Pos20"] = 0.0
    # RSI
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI"] = 100 - (100 / (1 + rs))
    out.loc[avg_loss == 0, "RSI"] = 100
    out.loc[(avg_gain == 0) & (avg_loss > 0), "RSI"] = 0
    # ATR
    prev_close = out["close"].shift(1)
    tr_components = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    out["ATR"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    # MACD
    ema_fast = out["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = out["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_Signal"] = out["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    # Trend10 (문자열 패턴)
    out = _add_trend10_for_collector(out, window=10)
    # NaN/inf 정리
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 숫자 컬럼과 문자열 컬럼 분리
    core_numeric_cols = [
        "close", "BB_Pos", "BandWidth", "Range_Pos20",
        "RSI", "ATR", "MACD", "MACD_Signal", "MACD_Hist"
    ]
    core_non_numeric_cols = ["Trend10"]  # 필요 시 추가
    # 존재하는 숫자 컬럼만 선택
    exist_num = [c for c in core_numeric_cols if c in out.columns]
    if exist_num:
        # 명시적 float 변환 후 결측치 0.0
          # (astype 전에 object 섞인 경우를 대비해 to_numeric 사용 옵션도 가능)
        for c in exist_num:
            # to_numeric로 강제 변환 (errors='coerce'로 안전하게 NaN 처리)
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out[exist_num] = out[exist_num].fillna(0.0)
    # 비숫자 컬럼은 건드리지 않거나 필요 시 결측치 None 처리
    for c in core_non_numeric_cols:
        if c in out.columns:
            # 패턴이 아직 생성 안된 구간은 그대로 None 유지
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

# 새 헬퍼 추가
def _sanitize_value(v, use_null=False):
    """
    숫자 값에서 NaN/inf 제거.
    use_null=True 이면 None 반환, False 이면 0.0 반환.
    """
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
            "time": last.name.isoformat()
        }
        # 혹시 남은 NaN/inf 정리 (미방어 컬럼 대비)
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
                "updated": _dt.utcnow().isoformat(),
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
        "updated": _dt.utcnow().isoformat(),
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
    url = 'http://ywydpapa.iptime.org:8000/api/top30coins'
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
        if not isinstance(coins, list):
            print("get_trendcoins: 예상치 못한 응답 형식.")
            return []
        return coins
    except Exception as e:
        print(f"get_trendcoins: 요청 실패 - {e}")
        return []

cross_memory = {}  # 전역 (최초 1회)

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
            # 남은 초당 호출량이 1 이하이면 잠깐 쉬어 버스트 방지
            if rem and rem["sec"] <= 1:
                time.sleep(0.25 + random.uniform(0, 0.05))
            return resp

        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            if ra:
                wait = float(ra)
            else:
                wait = min(2.0, base_sleep * (2 ** (attempt - 1)) + random.uniform(0, jitter))
                if rem and rem["sec"] == 0:
                    wait = max(wait, 0.6)
            print(f"[WARN] 429 too_many_requests attempt={attempt} waiting {wait:.3f}s rem={rem}")
            time.sleep(wait)
            continue

        # 기타 오류
        resp.raise_for_status()

    raise RuntimeError("429 재시도 초과")

async def peak_trade(
    ticker='KRW-BTC',
    short_window=3,
    long_window=20,
    count=180,
    candle_unit='1h'
):
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

    if api_type == 'days':
        url = f'https://api.upbit.com/v1/candles/days?market={ticker}&count={count}'
    else:
        url = f'https://api.upbit.com/v1/candles/minutes/{minute}?market={ticker}&count={count}'

    resp = polite_get(url)
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"JSON 파싱 실패 status={resp.status_code} text={resp.text[:200]}")

    if isinstance(data, dict) and 'error' in data:
        raise RuntimeError(f"Upbit API 오류: {data['error']}")

    if not isinstance(data, list):
        raise RuntimeError(f"예상치 못한 응답 타입: {type(data)} {str(data)[:200]}")

    if len(data) == 0:
        cross_memory[ticker] = {'error': '빈 데이터', 'ticker': ticker}
        return

    df = pd.DataFrame(data)

    if 'candle_date_time_utc' not in df.columns:
        raise RuntimeError(f"필수 컬럼 없음 columns={list(df.columns)}")

    idx = (
        pd.to_datetime(df['candle_date_time_utc'], format='%Y-%m-%dT%H:%M:%S', utc=True)
        .dt.tz_convert('Asia/Seoul')
    )
    df.index = idx
    df.index.name = 'candle_time_kst'
    df = df.sort_index()

    # 최소 롤링 길이 확보
    need_len = max(short_window, long_window)
    if len(df) < need_len:
        cross_memory[ticker] = {
            'error': '데이터 길이 부족',
            'len': len(df),
            'need': need_len
        }
        return

    sw_col = f'VWMA_{short_window}'
    lw_col = f'VWMA_{long_window}'

    df[sw_col] = (
        (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=short_window).sum()
        / df['candle_acc_trade_volume'].rolling(window=short_window).sum()
    )
    df[lw_col] = (
        (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=long_window).sum()
        / df['candle_acc_trade_volume'].rolling(window=long_window).sum()
    )

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
        cross_memory[ticker] = {
            'last_2_crosses': [],
            'latest_cross_type': None,
            'note': '교차 없음',
            'updated': now
        }
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
        'last_2_crosses': [
            {'type': r['cross_type'], 'time': t}
            for t, r in last_2.iterrows()
        ],
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

# ===============================================================================================
# 가격 조회
# ===============================================================================================
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

async def get_current_price(coinn):
    server_url = "https://api.upbit.com"
    params = {"quote_currencies": "KRW"}
    res = requests.get(server_url + "/v1/ticker/all", params=params)
    data = res.json()
    result = []
    for item in data:
        market = item.get("market")
        trade_price = item.get("trade_price")
        if market and trade_price and market == coinn:
            result.append({"market": market, "trade_price": trade_price})
    return result

# ===============================================================================================
# lifespan
# ===============================================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global krw_markets, _collect_tasks_started
    if not _collect_tasks_started:
        krw_markets = await get_krw_markets()
        # 실시간 체결 수집
        for chunk in chunk_markets(krw_markets, 50):
            asyncio.create_task(upbit_collector(chunk))
        asyncio.create_task(reset_trade_counts())
        asyncio.create_task(trend_loop())
        start_bb_collector_once()  # 다중 타임프레임 collector 시작
        _collect_tasks_started = True
    yield
    # 종료 시 정리 로직 필요하면 추가

app.router.lifespan_context = lifespan

# ===============================================================================================
# 라우트 / WebSocket
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
    top30 = result[:30]
    return JSONResponse(content={"markets": top30})

@app.get("/api/aitrendt30")
async def get_all_trends():
    return JSONResponse(content=jsonable_encoder(cross_memory))

# ------------------- 다중 타임프레임 조회 & 강제 수집 -------------------
@app.get("/api/bbtrend30")
async def api_bbtrend30(tf: str = None):
    if not bbtrend_cache["updated"]:
        return {
            "status": "initial",
            "message": "첫 수집 전입니다. 잠시 후 다시 시도하세요."
        }
    if bbtrend_cache["status"] == "empty" and not bbtrend_cache["timeframes"]:
        return {
            "status": "empty",
            "message": "현재 /api/top30coins 결과가 없거나 캔들 수집 불가.",
            "updated": bbtrend_cache["updated"]
        }
    if tf:
        if tf not in bbtrend_cache["timeframes"]:
            return {
                "status": "error",
                "message": f"지원하지 않는 타임프레임: {tf}",
                "available": list(bbtrend_cache["timeframes"].keys())
            }
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

# ===============================================================================================
# 사용자/지갑/예측 관련 기존 엔드포인트
# ===============================================================================================
@app.get("/phapp/tradesetup/{uno}")
async def phapp_tradesetup(uno: int, db: AsyncSession = Depends(get_db)):
    setups = None
    try:
        query = text("SELECT * FROM polarisSets where userNo = :uno and attrib not like :attxx")
        result = await db.execute(query, {"uno": uno, "attxx": "%XXX%"})
        rows = result.fetchall()
        setups = [
            {
                "coinName": row[2],
                "stepAmt": row[3],
                "tradeType": row[4],
                "maxAmt": row[5],
                "useYN": row[6]
            } for row in rows
        ]
        query2 = text("SELECT changeType, currency,unitPrice,inAmt,outAmt,remainAmt,regDate FROM trWallet where userNo = :uno and attrib not like :attxx order by currency ")
        result2 = await db.execute(query2, {"uno": uno, "attxx": "%XXX%"})
        rows2 = result2.fetchall()
        mycoins = [{
            "changeType": row2[0],
            "currency": row2[1],
            "unitPrice": row2[2],
            "inAmt": row2[3],
            "outAmt": row2[4],
            "remainAmt": row2[5],
            "regDate": row2[6]
        } for row2 in rows2]
        cprices = await get_current_prices()
        return setups, mycoins, cprices
    except Exception as e:
        print("Init Error !!", e)

@app.get("/phapp/tradelog/{uno}")
async def tradelog(uno: int, db: AsyncSession = Depends(get_db)):
    mycoins = None
    try:
        query = text("SELECT changeType, currency,unitPrice,inAmt,outAmt,remainAmt,regDate FROM trWallet where linkNo = (select max(linkNo) from trWallet where userNo = :uno) order by regDate asc")
        result = await db.execute(query, {"uno": uno})
        rows = result.fetchall()
        mycoins = [{
            "changeType": row[0],
            "currency": row[1],
            "unitPrice": row[2],
            "inAmt": row[3],
            "outAmt": row[4],
            "remainAmt": row[5],
            "regDate": row[6]
        } for row in rows]
    except Exception as e:
        print("Init Error !!", e)
    return mycoins

@app.get("/phapp/hotcoinlist")
async def hotcoins(db: AsyncSession = Depends(get_db)):
    try:
        query = text("SELECT * FROM orderbookAmt where dateTag = (select max(dateTag) from orderbookAmt)")
        result = await db.execute(query)
        rows = result.fetchall()
        orderbooks = [
            {
                "dateTag": row[1],
                "idxRow": row[2],
                "coinName": row[3],
                "bidAmt": row[4],
                "askAmt": row[5],
                "totalAmt": row[6],
                "amtDiff": row[7]
            }
            for row in rows
        ]
        return orderbooks
    except Exception as e:
        print("Get Hotcoins Error !!", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/phapp/mlogin/{phoneno}/{passwd}")
async def mlogin(phoneno: str, passwd: str, db: AsyncSession = Depends(get_db)):
    result = None
    try:
        query = text("SELECT userNo, userName,setupKey from trUser where userId = :phoneno and userPasswd = PASSWORD(:passwd)")
        r = await db.execute(query, {"phoneno": phoneno, "passwd": passwd})
        rows = r.fetchone()
        if rows is None:
            return {"error": "No data found for the given data."}
        result = {"userno": rows[0], "username": rows[1], "setupkey": rows[2]}
    except:
        print("mLogin error")
    finally:
        return result

@app.get("/rest_add_predict/{dateTag}/{coinName}/{avgUprate}/{avgDownrate}/{currentPrice}/{predictA}/{predictB}/{predictC}/{predictD}/{rateA}/{rateB}/{rateC}/{rateD}/{intv}")
async def rest_add_predict(request: Request, dateTag: str, coinName: str, avgUprate: float, avgDownrate: float,
                           currentPrice: float, predictA: float, predictB: float, predictC: float, predictD: float,
                           rateA: float, rateB: float, rateC: float, rateD: float, intv: str,
                           db: AsyncSession = Depends(get_db)):
    result = await rest_predict(dateTag, coinName, avgUprate, avgDownrate, currentPrice,
                                predictA, predictB, predictC, predictD,
                                rateA, rateB, rateC, rateD, intv, db)
    return bool(result)

@app.get("/restaddorderbookamt/{datetag}/{idxrow}/{coinn}/{bidamt}/{askamt}/{totalamt}/{amtdiff}")
async def restaddorderbookamt(request: Request, datetag: str, idxrow: int, coinn: str, bidamt: int, askamt: int,
                              totalamt: int, amtdiff: float, db: AsyncSession = Depends(get_db)):
    try:
        act = await rest_add_orderbook_amt(datetag, idxrow, coinn, bidamt, askamt, totalamt, amtdiff, db)
        return JSONResponse({"success": True})
    except Exception as e:
        print("Error!!", e)
        return JSONResponse({"success": False})

@app.get("/restaddtradeamt/{bidamt}/{askamt}")
async def restaddtradeamt(request: Request, bidamt: int, askamt: int, db: AsyncSession = Depends(get_db)):
    try:
        act = await rest_add_trade_amt(bidamt, askamt, db)
        return JSONResponse({"success": True})
    except Exception as e:
        print("Error!!", e)
        return JSONResponse({"success": False})

@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("/privacy/privacy.html", {"request": request})

async def rest_add_trade_amt(bidamt, askamt, db):
    try:
        query = text("INSERT INTO tradeAmt (bidAmt, askAmt) VALUES (:bidamt, :askamt)")
        await db.execute(query, {"bidamt": bidamt, "askamt": askamt})
        await db.commit()
        return True
    except Exception as e:
        print("Error!!", e)
        return False

async def rest_add_orderbook_amt(datetag, idxrow, coinn, bidamt, askamt, totalamt, amtdiff, db):
    try:
        query = text(
            "INSERT INTO orderbookAmt (dateTag, idxRow, coinName, bidAmt, askAmt, totalAmt, amtDiff) values (:dateTag, :idxRow, :coinName, :bidAmt, :askAmt, :totalAmt, :amtDiff)")
        await db.execute(query, {
            "dateTag": datetag, "idxRow": idxrow, "coinName": coinn, "bidAmt": bidamt,
            "askAmt": askamt, "totalAmt": totalamt, "amtDiff": amtdiff
        })
        await db.commit()
        return True
    except Exception as e:
        print("Error!!", e)
        return False

async def rest_predict(dateTag, coinName, avgUprate, avgDownrate, currentPrice,
                       predictA, predictB, predictC, predictD,
                       rateA, rateB, rateC, rateD, intV, db):
    try:
        query = text("""
            INSERT into predictPrice
            (dateTag,coinName,avgUprate,avgDownrate,currentPrice,
             predictA,predictB,predictC,predictD,
             rateA,rateB,rateC,rateD,intV)
            values (:dateTag,:coinName,:avgUprate,:avgDownrate,:currentPrice,
                    :predictA,:predictB,:predictC,:predictD,
                    :rateA,:rateB,:rateC,:rateD,:intv)
        """)
        await db.execute(query, {
            "dateTag": dateTag, "coinName": coinName, "avgUprate": avgUprate,
            "avgDownrate": avgDownrate, "currentPrice": currentPrice,
            "predictA": predictA, "predictB": predictB, "predictC": predictC, "predictD": predictD,
            "rateA": rateA, "rateB": rateB, "rateC": rateC, "rateD": rateD,
            "intv": intV
        })
        await db.commit()
        return True
    except Exception as e:
        print("Error!!", e)
        return False

# ===============================================================================================
# END
# ===============================================================================================