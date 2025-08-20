import asyncio
import requests
import time
import numpy as np
from fastapi import FastAPI, Request, WebSocket, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
import websockets
import json
from fastapi.encoders import jsonable_encoder
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
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import json
import dotenv
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    global krw_markets, _collect_tasks_started
    if not _collect_tasks_started:
        krw_markets = await get_krw_markets()
        for chunk in chunk_markets(krw_markets, 50):
            asyncio.create_task(upbit_collector(chunk))
        asyncio.create_task(reset_trade_counts())
        asyncio.create_task(trend_loop())
        _collect_tasks_started = True
    yield
    # (종료시 정리 코드 필요시 여기에)

dotenv.load_dotenv()
DATABASE_URL = os.getenv("dburl")
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.")
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
trade_queues = defaultdict(lambda: {'BID': deque(maxlen=240), 'ASK': deque(maxlen=240)})
krw_markets = []
_collect_tasks_started = False
trade_counts = defaultdict(int)
trend_cache = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (보안상 실제 운영에서는 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

async def get_db():
    async with async_session() as session:
        yield session

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

cross_memory = {}

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
            print("get_trendcoins: 예상치 못한 응답 형식(markets가 리스트 아님).")
            return []
        return coins
    except Exception as e:
        print(f"get_trendcoins: 요청 실패 - {e}")
        return []


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
    if 'candle_date_time_utc' in df.columns:
        idx = pd.to_datetime(
            df['candle_date_time_utc'],
            format='%Y-%m-%dT%H:%M:%S',
            utc=True
        ).dt.tz_convert('Asia/Seoul')
        df.set_index(idx, inplace=True)
        df.index.name = 'candle_time_kst'
    else:
        idx = pd.to_datetime(
            df['candle_date_time_kst'],
            format='%Y-%m-%dT%H:%M:%S'
        ).dt.tz_localize('Asia/Seoul')
        df.set_index(idx, inplace=True)
        df.index.name = 'candle_time_kst'
    df = df.sort_index(ascending=True)
    df['VWMA_3'] = (
            (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=3).sum() /
            df['candle_acc_trade_volume'].rolling(window=3).sum()
    )
    df['VWMA_20'] = (
            (df['trade_price'] * df['candle_acc_trade_volume']).rolling(window=20).sum() /
            df['candle_acc_trade_volume'].rolling(window=20).sum()
    )
    df['vwma_diff'] = df['VWMA_3'] - df['VWMA_20']
    golden_cross = df[(df['vwma_diff'].shift(1) < 0) & (df['vwma_diff'] > 0)].copy()
    golden_cross['cross_type'] = 'golden'
    dead_cross = df[(df['vwma_diff'].shift(1) > 0) & (df['vwma_diff'] < 0)].copy()
    dead_cross['cross_type'] = 'dead'
    crosses = pd.concat([golden_cross, dead_cross]).sort_index()
    last_2_crosses = crosses.tail(2)
    if last_2_crosses.empty:
        print("교차 신호가 없어 경과 시간을 계산할 수 없습니다.")
        return
    latest_cross_time = last_2_crosses.index[-1]
    now = pd.Timestamp.now(tz='Asia/Seoul')
    elapsed = now - latest_cross_time
    # VWMA 방향(상승/하강) 및 기울기(변화량) 판별
    if len(df) >= 6:
        vwma3_dir = 'UP' if df['VWMA_3'].iloc[-1] > df['VWMA_3'].iloc[-2] else 'DN'
        vwma20_dir = 'UP' if df['VWMA_20'].iloc[-1] > df['VWMA_20'].iloc[-2] else 'DN'
        vwma3_slope = df['VWMA_3'].iloc[-1] - df['VWMA_3'].iloc[-2]
        vwma20_now = df['VWMA_20'].iloc[-1]
        vwma20_mean = df['VWMA_20'].iloc[-6:].mean()
        vwma20_slope = vwma20_now - vwma20_mean
        vwma20_angle = np.degrees(np.arctan(vwma20_slope))
        volume_slope = df['candle_acc_trade_volume'].iloc[-1] - df['candle_acc_trade_volume'].iloc[-2]
        vwma3_angle = np.degrees(np.arctan(vwma3_slope))
        volume_angle = np.degrees(np.arctan(volume_slope))
    else:
        vwma20_slope = None
        vwma20_angle = None
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


async def trend_loop():
    while True:
        coins = await get_trendcoins()
        if not coins:
            # 실패 또는 빈 응답 시 잠시 대기 후 재시도 (프록시/네트워크 불안정 대응)
            await asyncio.sleep(5)
            continue

        for coin in coins:
            try:
                await peak_trade(ticker=coin, candle_unit='1m')
            except Exception as e:
                print(f"peak_trade 실패({coin}): {e}")
            await asyncio.sleep(0.5)
        await asyncio.sleep(60)


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


@app.get("/api/aitrendt30")
async def get_all_trends():
    return JSONResponse(content=jsonable_encoder(cross_memory))

@app.get("/phapp/tradesetup/{uno}")
async def phapp_tradesetup(uno: int, db: AsyncSession = Depends(get_db)):
    setups = None
    try:
        query = text("SELECT * FROM polarisSets where userNo = :uno and attrib not like :attxx")
        result = await db.execute(query, {"uno": uno, "attxx": "%XXX%"})
        rows = result.fetchall()
        setups = [
            {
                "coinName":row[2],
                "stepAmt":row[3],
                "tradeType":row[4],
                "maxAmt":row[5],
                "useYN":row[6]
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
            "changeType":row[0],
            "currency":row[1],
            "unitPrice":row[2],
            "inAmt":row[3],
            "outAmt":row[4],
            "remainAmt":row[5],
            "regDate":row[6]
        }
            for row in rows
        ]
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
                "dateTag":row[1],
                "idxRow":row[2],
                "coinName":row[3],
                "bidAmt":row[4],
                "askAmt":row[5],
                "totalAmt":row[6],
                "amtDiff":row[7]
            }
            for row in rows
        ]
        return orderbooks
    except Exception as e:
        print("Get Hotcoins Error !!", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/phapp/mlogin/{phoneno}/{passwd}")
async def mlogin(phoneno: str,passwd:str, db: AsyncSession = Depends(get_db)):
    try:
        query = text("SELECT userNo, userName,setupKey from trUser where userId = :phoneno and userPasswd = PASSWORD(:passwd)")
        result = await db.execute(query, {"phoneno": phoneno, "passwd": passwd})
        rows = result.fetchone()
        if rows is None:
            return {"error": "No data found for the given data."}
        result = {"userno": rows[0], "username": rows[1], "setupkey": rows[2]}
    except:
        print("mLogin error")
    finally:
        return result

@app.get("/rest_add_predict/{dateTag}/{coinName}/{avgUprate}/{avgDownrate}/{currentPrice}/{predictA}/{predictB}/{predictC}/{predictD}/{rateA}/{rateB}/{rateC}/{rateD}/{intv}")
async def rest_add_predict(request:Request,dateTag:str,coinName:str,avgUprate:float,avgDownrate:float,currentPrice:float,predictA:float,predictB:float,predictC:float,predictD:float,rateA:float,rateB:float,rateC:float,rateD:float,intv:str, db: AsyncSession = Depends(get_db)):
    result = await rest_predict(dateTag,coinName,avgUprate,avgDownrate,currentPrice,predictA,predictB,predictC,predictD,rateA,rateB,rateC,rateD,intv, db)
    if result:
        return True
    else:
        return False

@app.get("/restaddorderbookamt/{datetag}/{idxrow}/{coinn}/{bidamt}/{askamt}/{totalamt}/{amtdiff}")
async def restaddorderbookamt(request: Request, datetag: str, idxrow: int, coinn: str, bidamt: int, askamt: int,
                              totalamt: int, amtdiff: float, db: AsyncSession = Depends(get_db)):
    try:
        act = await rest_add_orderbook_amt(datetag, idxrow, coinn, bidamt, askamt, totalamt, amtdiff, db)
        print(act)
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
        await db.execute(query,
                         {"dateTag": datetag, "idxRow": idxrow, "coinName": coinn, "bidAmt": bidamt, "askAmt": askamt,
                          "totalAmt": totalamt, "amtDiff": amtdiff})
        await db.commit()
        return True
    except Exception as e:
        print("Error!!", e)
        return False


async def rest_predict(dateTag,coinName,avgUprate,avgDownrate,currentPrice,predictA,predictB,predictC,predictD,rateA,rateB,rateC,rateD,intV,db):
    try:
        query = text("INSERT into predictPrice (dateTag,coinName,avgUprate,avgDownrate,currentPrice,predictA,predictB,predictC,predictD,rateA,rateB,rateC,rateD,intV) values (:dateTag,:coinName,:avgUprate,:avgDownrate,:currentPrice,:predictA,:predictB,:predictC,:predictD,:rateA,:rateB,:rateC,:rateD,:intv)")
        await db.execute(query,{"dateTag":dateTag, "coinName": coinName, "avgUprate": avgUprate, "avgDownrate": avgDownrate, "currentPrice": currentPrice, "predictA": predictA,"predictB": predictB,"predictC": predictC,"predictD": predictD, "rateA":rateA,"rateB":rateB,"rateC":rateC,"rateD":rateD,"intv":intV})
        await db.commit()
        return True
    except Exception as e:
        print("Error!!", e)
        return False


