from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import asyncio
import websockets
import json
from collections import defaultdict, deque
import httpx

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 전체 코인 코드 가져오기
async def get_krw_markets():
    url = "https://api.upbit.com/v1/market/all?isDetails=false"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        markets = [m['market'] for m in resp.json() if m['market'].startswith("KRW-")]
        return markets

# trade_queues: 모든 마켓-사이드별로 최근 100건 체결금액 저장
trade_queues = defaultdict(lambda: {'BID': deque(maxlen=100), 'ASK': deque(maxlen=100)})
krw_markets = []

# 50개씩 마켓 나누기
def chunk_markets(markets, size=50):
    for i in range(0, len(markets), size):
        yield markets[i:i+size]

async def upbit_collector(markets_chunk):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_fmt = [
        {"ticket": "test"},
        {"type": "trade", "codes": markets_chunk}
    ]
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(subscribe_fmt))
        while True:
            data = await websocket.recv()
            trade = json.loads(data)
            market = trade['code']
            side = trade['ask_bid']  # 'BID' or 'ASK'
            volume = float(trade['trade_volume'])
            price = float(trade['trade_price'])
            amount = volume * price
            trade_queues[market][side].append(amount)

@app.on_event("startup")
async def startup_event():
    global krw_markets
    krw_markets = await get_krw_markets()
    # 50개씩 소켓 연결
    for chunk in chunk_markets(krw_markets, 50):
        asyncio.create_task(upbit_collector(chunk))

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    # 마켓명 리스트 전달
    return templates.TemplateResponse("index.html", {"request": request, "markets": krw_markets})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # 각 마켓-사이드별로 10, 30, 100개 합계 계산
        result = {}
        for market in krw_markets:
            result[market] = {}
            for side in ['BID', 'ASK']:
                q = list(trade_queues[market][side])
                sum10 = sum(q[-10:]) if len(q) >= 1 else 0
                sum30 = sum(q[-30:]) if len(q) >= 1 else 0
                sum100 = sum(q[-100:]) if len(q) >= 1 else 0
                result[market][side] = {
                    "sum10": round(sum10, 0),
                    "sum30": round(sum30, 0),
                    "sum100": round(sum100, 0)
                }
        await websocket.send_json(result)
        await asyncio.sleep(1)
