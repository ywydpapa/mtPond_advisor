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

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# trade_queues: 모든 마켓-사이드별로 최근 100건 체결금액 저장
trade_queues = defaultdict(lambda: {'BID': deque(maxlen=120), 'ASK': deque(maxlen=120)})
krw_markets: List[str] = []
_collect_tasks_started = False
trade_counts = defaultdict(int)


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
        await asyncio.sleep(60)  # 1분마다
        for market in trade_counts.keys():
            trade_counts[market] = 0

@app.on_event("startup")
async def startup_event():
    global krw_markets, _collect_tasks_started
    if _collect_tasks_started:
        return
    krw_markets = await get_krw_markets()
    for chunk in chunk_markets(krw_markets, 50):
        asyncio.create_task(upbit_collector(chunk))
    asyncio.create_task(reset_trade_counts())  # 카운트 리셋 태스크 추가
    _collect_tasks_started = True


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
                # 비율
                ratio10 = round((bid10/ask10)*100, 1) if ask10 > 0 else 0.0
                ratio30 = round((bid30/ask30)*100, 1) if ask30 > 0 else 0.0
                ratio120 = round((bid120/ask120)*100, 1) if ask120 > 0 else 0.0
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
                    "ratio10": ratio10,
                    "ratio30": ratio30,
                    "ratio120": ratio120,
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
                ratio10 = round((bid10/ask10)*100, 1) if ask10 > 0 else 0.0
                ratio30 = round((bid30/ask30)*100, 1) if ask30 > 0 else 0.0
                ratio120 = round((bid120/ask120)*100, 1) if ask120 > 0 else 0.0
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
                    "ratio10": ratio10,
                    "ratio30": ratio30,
                    "ratio120": ratio120,
                }
                # ratio30이 100을 넘는 것만 추가
                if ratio30 > 100 and speed > 0:
                    result.append(row)
            # speed 내림차순 정렬
            result.sort(key=lambda x: x['speed'], reverse=True)
            top30 = result[:30]
            await websocket.send_json({
                "count": len(top30),
                "coins": top30
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
