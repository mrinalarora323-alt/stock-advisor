import yfinance as yf, numpy as np, warnings, os, json, re, time as time_module, ssl
from flask import Flask, request, jsonify, render_template
from urllib.request import Request, urlopen
warnings.filterwarnings('ignore')

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

# Rate limiter & retry for yfinance
_yf_last_call = 0
_YF_MIN_INTERVAL = 1.0
_YF_MAX_RETRIES = 3

def _yf_call(fn, *args, **kwargs):
    global _yf_last_call
    for attempt in range(_YF_MAX_RETRIES):
        elapsed = time_module.time() - _yf_last_call
        if elapsed < _YF_MIN_INTERVAL:
            time_module.sleep(_YF_MIN_INTERVAL - elapsed)
        _yf_last_call = time_module.time()
        try:
            return fn(*args)
        except Exception as e:
            if 'Too Many Requests' in str(e) or '429' in str(e):
                time_module.sleep(2 ** attempt)
                continue
            raise
    return None

def _get_ticker(symbol):
    return _yf_call(lambda s: yf.Ticker(s+'.NS'), symbol)

def _get_info(ticker):
    return _yf_call(lambda t: t.info, ticker)

def _get_history(ticker, period='3mo'):
    return _yf_call(lambda t: t.history(period=period), ticker)

app = Flask(__name__)

# Simple in-memory cache to avoid yfinance rate limits
_cache = {}
CACHE_TTL = 300  # 5 min

def get_cached(key):
    now = time_module.time()
    if key in _cache and now - _cache[key]['time'] < CACHE_TTL:
        return _cache[key]['data']
    return None

def set_cache(key, data):
    _cache[key] = {'data': data, 'time': time_module.time()}

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
AI_ENABLED = bool(GROQ_API_KEY or OPENROUTER_KEY)

AI_PROMPT = """You are a professional Indian stock market analyst. Analyze this stock and give a clear BUY/HOLD/AVOID verdict.

Stock Data:
- Name: {name}
- Symbol: {symbol}
- Sector: {sector} | Industry: {industry}
- Current Price: ₹{price}
- P/E: {pe} | Forward P/E: {fwd_pe}
- ROE: {roe_pct}% | Profit Margin: {pm_pct}%
- Revenue Growth: {rev_g_pct}%
- Debt/Equity: {de}
- Dividend Yield: {dy_pct}%
- EPS: ₹{eps}
- Market Cap: ₹{mcap_cr}Cr
- Operating Margin: {op_margin_pct}%
- P/B: {pb}

Technical Indicators:
- RSI (14): {rsi}
- 1 Week Change: {mom_1w}%
- 1 Month Change: {mom_1m}%
- Price vs MA20: {pct_ma20}%
- Price vs MA50: {pct_ma50}%
- Volatility: {volatility}%
- 52W High: ₹{high52} | 52W Low: ₹{low52}

Respond ONLY with a valid JSON object (no markdown, no backticks):
{{
  "verdict": "buy" or "hold" or "avoid",
  "verdict_title": "short emoji + title in Hinglish",
  "verdict_text": "2-3 line explanation in Hinglish",
  "reasons_pos": ["reason 1 in Hinglish", "reason 2 in Hinglish"],
  "reasons_neg": ["reason 1 in Hinglish", "reason 2 in Hinglish"],
  "ai_insight": "1-2 line unique insight about this stock in Hinglish"
}}"""

def call_llm_api(url, model, api_key, body_template):
    prompt = AI_PROMPT.format(**body_template)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
    }).encode()
    req = Request(url, data=body,
                  headers={"Authorization": f"Bearer {api_key}",
                           "Content-Type": "application/json",
                           "User-Agent": "Stock-Advisor/1.0"})
    resp = json.loads(urlopen(req, context=_ssl_ctx, timeout=30).read())
    text = resp["choices"][0]["message"]["content"].strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text)

def ask_ai(stock_data):
    if not AI_ENABLED:
        return None
    # Try Groq first, then OpenRouter
    if GROQ_API_KEY:
        try:
            return call_llm_api(
                "https://api.groq.com/openai/v1/chat/completions",
                "llama-3.3-70b-versatile",
                GROQ_API_KEY, stock_data
            )
        except:
            pass
    if OPENROUTER_KEY:
        try:
            return call_llm_api(
                "https://openrouter.ai/api/v1/chat/completions",
                "mistralai/mistral-7b-instruct:free",
                OPENROUTER_KEY, stock_data
            )
        except:
            pass
    return None

# Name-to-ticker mapping for common stocks
NAME_MAP = {
    "itc": "ITC", "tcs": "TCS", "infosys": "INFY", "wipro": "WIPRO", "hcl": "HCLTECH",
    "reliance": "RELIANCE", "ril": "RELIANCE", "hdfc bank": "HDFCBANK", "hdfc": "HDFCBANK",
    "icici": "ICICIBANK", "icici bank": "ICICIBANK", "sbi": "SBIN", "axis bank": "AXISBANK",
    "axis": "AXISBANK", "kotak": "KOTAKBANK", "kotak bank": "KOTAKBANK", "yes bank": "YESBANK",
    "sbi life": "SBILIFE", "hdfc life": "HDFCLIFE", "bajaj finance": "BAJFINANCE",
    "bajaj finserv": "BAJAJFINSV", "bajaj auto": "BAJAJ-AUTO", "hero moto": "HEROMOTOCO",
    "hero": "HEROMOTOCO", "maruti": "MARUTI", "maruti suzuki": "MARUTI", "mahindra": "M&M",
    "tata motors": "TATAMOTORS", "tata steel": "TATASTEEL", "tata power": "TATAPOWER",
    "tata consumer": "TATACONSUM", "tata elxsi": "TATAELXSI", "nifty": "^NSEI",
    "sensex": "^BSESN", "asian paints": "ASIANPAINT", "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR", "nestle": "NESTLEIND", "britannia": "BRITANNIA",
    "dabur": "DABUR", "marico": "MARICO", "godrej cp": "GODREJCP", "colgate": "COLPAL",
    "pidilite": "PIDILITIND", "sun pharma": "SUNPHARMA", "dr reddy": "DRREDDY",
    "cipla": "CIPLA", "lupin": "LUPIN", "divi": "DIVISLAB", "divis lab": "DIVISLAB",
    "apollo hospitals": "APOLLOHOSP", "apollo": "APOLLOHOSP", "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL", "jio": "RELIANCE", "ntpc": "NTPC", "powergrid": "POWERGRID",
    "ongc": "ONGC", "coal india": "COALINDIA", "coalindia": "COALINDIA",
    "adani ports": "ADANIPORTS", "adani green": "ADANIGREEN", "adani power": "ADANIPOWER",
    "dmart": "AVENUE", "avenue supermarts": "AVENUE", "titan": "TITAN", "trent": "TRENT",
    "zomato": "ZOMATO", "paytm": "PAYTM", "nykaa": "FSN",
    "piramal pharma": "PPLPHARMA", "pplpharma": "PPLPHARMA", "groww": "GROWW",
    "nmdcltd": "NMDC", "vedanta": "VEDL", "vedl": "VEDL", "jsw steel": "JSWSTEEL",
    "hindalco": "HINDALCO", "nationalum": "NATIONALUM", "shriram finance": "SHRIRAMFIN",
    "shriram": "SHRIRAMFIN", "bajaj": "BAJFINANCE", "eicher": "EICHERMOT",
    "eicher motors": "EICHERMOT", "bpcl": "BPCL", "hpcl": "HINDPETRO", "ioc": "IOC",
    "adani": "ADANIENT", "adani enterprises": "ADANIENT",
    "beltronics": "BEL", "bharat electronics": "BEL",
    "hal": "HAL", "hindustan aeronautics": "HAL",
    "irctc": "IRCTC", "irfc": "IRFC", "rvnl": "RVNL", "lic": "LICI",
    "lic housing fin": "LICHSGFIN", "lic housing": "LICHSGFIN",
    "muthoot": "MUTHOOTFIN", "muthoot finance": "MUTHOOTFIN",
    "muthootfin": "MUTHOOTFIN",
    "nifty": "^NSEI", "sensex": "^BSESN",
    "bank nifty": "^NSEBANK",
    "pfc": "PFC", "power finance": "PFC",
    "rec": "RECLTD", "rec ltd": "RECLTD",
    "suzlon": "SUZLON", "suzlon energy": "SUZLON",
    "tatachem": "TATACHEM", "tata chemicals": "TATACHEM",
    "hdfc amc": "HDFCAMC", "nia": "HDFCAMC",
    "uti amc": "UTIAMC",
    "ipl": "RELIANCE", "jio": "RELIANCE",
    "pvr": "PVRINOX", "pvr inox": "PVRINOX", "inox": "PVRINOX",
    "gland pharma": "GLAND", "biocon": "BIOCON",
    "torrent pharma": "TORNTPHARM", "torrent power": "TORNTPOWER",
    "abb india": "ABB", "siemens india": "SIEMENS",
    "havells": "HAVELLS", "crompton": "CROMPTON",
    "voltas": "VOLTAS", "blue star": "BLUESTARCO",
    "whirlpool": "WHIRLPOOL", "bajaj electricals": "BAJAJELEC",
    "amara raja": "AMARAJABAT", "amaron": "AMARAJABAT",
    "exide": "EXIDEIND", "exide industries": "EXIDEIND",
    "page industries": "PAGEIND", "page": "PAGEIND",
    "bata": "BATAINDIA", "bata india": "BATAINDIA",
    "metro brands": "METROBRAND",
    "berger paints": "BERGEPAINT", "berger": "BERGEPAINT",
    "kansai nerolac": "KANSAINER", "nerolac": "KANSAINER",
    "naturals": "VBL", "varun beverages": "VBL",
    "birlasoft": "BIRLASOFT",
    "l&t infotech": "LTIM",
    "ltimindtree": "LTIM",
    "mindtree": "LTIM",
    "coforge": "COFORGE", "persistent": "PERSISTENT",
    "cyient": "CYIENT", "l&t technology": "LTTS",
    "ltts": "LTTS",
    "gail": "GAIL", "gail india": "GAIL",
    "petronet": "PETRONET", "petronet lng": "PETRONET",
    "mcdowell": "MCDOWELL-N", "united spirits": "MCDOWELL-N",
    "radico": "RADICO", "radico khaitan": "RADICO",
    "zydus": "ZYDUSLIFE", "zydus lifesciences": "ZYDUSLIFE",
    "torrent pharma": "TORNTPHARM",
    "ajanta pharma": "AJANTPHARM",
    "alkem": "ALKEM", "alkem labs": "ALKEM",
    "aurobindo": "AUROPHARMA", "auropharma": "AUROPHARMA",
    "granules": "GRANULES", "granules india": "GRANULES",
    "laurus labs": "LAURUSLABS", "laurus": "LAURUSLABS",
}

# Build spaceless map for fuzzy matching
SPACELESS_MAP = {k.replace(' ', ''): v for k, v in NAME_MAP.items()}

def resolve_symbol(query):
    q = query.strip().upper()
    if q.endswith('.NS'): q = q[:-3]
    
    # Clean out question words & punctuation for matching
    words = q.replace('?', '').replace('!', '').replace('.', '').split()
    stop_words = {'IS', 'ARE', 'DO', 'DOES', 'CAN', 'SHOULD', 'WILL', 'WOULD',
                  'GOOD', 'BAD', 'BEST', 'WORST', 'BUY', 'SELL', 'HOLD', 'CHECK',
                  'A', 'AN', 'THE', 'FOR', 'TO', 'OF', 'IN', 'AT', 'BY'}
    candidates = [w for w in words if w not in stop_words]
    clean_q = ' '.join(candidates) if candidates else q
    clean_lower = clean_q.lower().replace(' ', '')
    
    # Try longest candidates first
    for trial in sorted(candidates or [q], key=len, reverse=True):
        tu = trial.upper()
        tl = trial.lower()
        key = tl.replace(' ', '').replace(',', '')
        
        if tu in NAME_MAP.values():
            return tu
        if tl in NAME_MAP:
            return NAME_MAP[tl]
        if key in SPACELESS_MAP:
            return SPACELESS_MAP[key]
        if tu in NAME_MAP:
            return NAME_MAP[tu]
    
    # Try yfinance with cleaned query
    for trial in candidates or [q]:
        try:
            info = _get_info(_get_ticker(trial.upper()))
            if info.get('regularMarketPrice') or info.get('currentPrice'):
                return trial.upper()
        except:
            pass
    
    # Fuzzy match each candidate against company names
    for trial in candidates or [q]:
        ql = trial.lower()
        if len(ql) < 3:
            continue
        for sym in sorted(set(NAME_MAP.values())):
            try:
                info = _get_info(_get_ticker(sym))
                n = (info.get('longName') or '').lower()
                if n and (ql in n):
                    return sym
            except:
                pass
    
    return None

def analyze_stock(symbol):
    t = _get_ticker(symbol)
    info = _get_info(t)
    
    price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
    name = info.get('longName', symbol)
    sector = (info.get('sector') or '')[:25]
    industry = (info.get('industry') or '')[:30]
    
    pe = info.get('trailingPE') or 0
    fwd_pe = info.get('forwardPE') or 0
    roe = info.get('returnOnEquity') or 0
    pm = info.get('profitMargins') or 0
    de = info.get('debtToEquity') or 0
    raw_dy = info.get('dividendYield') or 0
    dy = raw_dy / 100 if raw_dy > 1 else raw_dy  # yfinance returns % or decimal inconsistently
    eps = info.get('trailingEps') or 0
    rev_g = info.get('revenueGrowth') or 0
    mcap = info.get('marketCap') or 0
    pb = info.get('priceToBook') or 0
    current_ratio = info.get('currentRatio') or 0
    op_margin = info.get('operatingMargins') or 0
    
    # Historical data for technicals
    hist = _get_history(t)
    c = hist['Close'].dropna().values
    
    # Technicals
    mom_1w = ((c[-1]/c[-6])-1)*100 if len(c)>=6 else 0
    mom_1m = ((c[-1]/c[-21])-1)*100 if len(c)>=21 else 0
    volatility = hist['Close'].dropna().pct_change().std() * (252**0.5) * 100
    
    ma20 = np.mean(c[-20:]) if len(c)>=20 else price
    ma50 = np.mean(c[-50:]) if len(c)>=50 else price
    pct_ma20 = ((price-ma20)/ma20)*100
    pct_ma50 = ((price-ma50)/ma50)*100
    
    deltas = np.diff(c)
    gain = np.where(deltas>0,deltas,0)
    loss = np.where(deltas<0,-deltas,0)
    avg_gain = np.mean(gain[-14:]) if len(gain)>=14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss)>=14 else 0
    rs = avg_gain/avg_loss if avg_loss else 100
    rsi = 100-(100/(1+rs))
    
    recent_high = np.max(c[-30:]) if len(c)>=30 else price*1.1
    recent_low = np.min(c[-30:]) if len(c)>=30 else price*0.9
    pct_from_high = ((price/recent_high)-1)*100
    pct_from_low = ((price/recent_low)-1)*100
    
    vl = hist['Volume'].dropna().values
    vol_ratio = (np.mean(vl[-5:])/np.mean(vl[-20:])) if len(vl)>=20 else 1
    
    # SCORING (rule-based fallback — used when AI is not available)
    score = 0
    reasons_pos = []
    reasons_neg = []
    
    if pe <= 0:
        score -= 15; reasons_neg.append(f"P/E 0 — company profit नहीं कमा रही, loss में है")
    elif pe < 15: score += 25; reasons_pos.append(f"P/E {pe:.1f}x — बहुत सस्ता है")
    elif pe < 25: score += 15; reasons_pos.append(f"P/E {pe:.1f}x — reasonable valuation")
    elif pe < 40: score += 5; reasons_pos.append(f"P/E {pe:.1f}x — थोड़ा expensive है")
    else: score -= 10; reasons_neg.append(f"P/E {pe:.1f}x — बहुत महंगा है, risk ज्यादा")
    
    if fwd_pe and fwd_pe < pe: score += 5; reasons_pos.append(f"Forward P/E {fwd_pe:.1f}x — earnings growth expected")
    
    if roe > 0.20: score += 20; reasons_pos.append(f"ROE {roe*100:.1f}% — कंपनी पैसा बनाने में माहिर है")
    elif roe > 0.12: score += 10; reasons_pos.append(f"ROE {roe*100:.1f}% — decent profitability")
    elif roe <= 0: score -= 15; reasons_neg.append(f"ROE {roe*100:.1f}% — company loss में है")
    
    if pm > 0.20: score += 15; reasons_pos.append(f"Profit Margin {pm*100:.1f}% — बहुत अच्छा मार्जिन")
    elif pm > 0.10: score += 8; reasons_pos.append(f"Margin {pm*100:.1f}% — OK है")
    elif pm <= 0: score -= 10; reasons_neg.append(f"Negative margin — company profit नहीं कमा रही")
    
    if rev_g > 0.20: score += 15; reasons_pos.append(f"Revenue Growth {rev_g*100:.1f}% — तेजी से बढ़ रही है")
    elif rev_g > 0.10: score += 8
    elif rev_g > 0: score += 3
    
    if de == 0: score += 10; reasons_pos.append("Debt/Eq: 0 — कर्ज नहीं है, बहुत safe")
    elif de < 1: score += 5; reasons_pos.append(f"Debt/Eq {de:.1f} — low debt")
    elif de > 3: score -= 10; reasons_neg.append(f"Debt/Eq {de:.1f} — बहुत ज्यादा कर्ज, risk")
    
    if dy > 0.03: score += 5; reasons_pos.append(f"Dividend {dy*100:.2f}% — अच्छा dividend देती है")
    elif dy > 0.01: score += 2
    
    if rsi < 30: score += 15; reasons_pos.append(f"RSI {rsi:.0f} — oversold है, bounce aane ka chance")
    elif rsi < 45: score += 5; reasons_pos.append(f"RSI {rsi:.0f} — cheap zone mein hai")
    elif rsi < 60: score += 3
    elif rsi > 75: score -= 10; reasons_neg.append(f"RSI {rsi:.0f} — overbought, gir sakta hai")
    
    if pct_ma20 > 0: score += 5; reasons_pos.append(f"Price MA20 से {abs(pct_ma20):.1f}% ऊपर — uptrend में है")
    elif pct_ma20 < -5: score -= 5; reasons_neg.append(f"Price MA20 से {abs(pct_ma20):.1f}% नीचे — weakness")
    
    if mom_1w > 2: score += 5; reasons_pos.append(f"Pichle hafte {mom_1w:+.1f}% up — momentum positive")
    elif mom_1w < -5: score -= 5
    
    if vol_ratio > 1.3: score += 3; reasons_pos.append(f"Volume {vol_ratio:.1f}x — buying interest badha hai")

    if score >= 50:
        verdict_type = "buy"
        verdict_title = "✅ BUY — खरीद सकते हैं"
        verdict_text = f"Score {score}/100. यह शेयर मजबूत fundamentals और अच्छी technicals के साथ है। खरीदने पर विचार कर सकते हैं।"
    elif score >= 25:
        verdict_type = "hold"
        verdict_title = "🟡 HOLD — रख सकते हैं / Wait करें"
        verdict_text = f"Score {score}/100. अभी खरीदने का best time नहीं है। अगर पहले से है तो रखें, नया buy सोच समझकर करें।"
    else:
        verdict_type = "avoid"
        verdict_title = "🔴 AVOID — अभी न लें"
        verdict_text = f"Score {score}/100. इस शेयर में अभी कमजोरी है। खरीदने से बचें। जब fundamentals improve हों तब सोचें।"
    
    # Try AI analysis
    ai_result = ask_ai({
        'name': name, 'symbol': symbol, 'price': price,
        'sector': sector, 'industry': industry,
        'pe': f'{pe:.1f}' if pe else 'Loss', 'fwd_pe': f'{fwd_pe:.1f}' if fwd_pe else 'N/A',
        'roe_pct': f'{roe*100:.1f}', 'pm_pct': f'{pm*100:.1f}',
        'rev_g_pct': f'{rev_g*100:.1f}' if rev_g else '0',
        'de': f'{de:.1f}' if de else '0 (Debt free)',
        'dy_pct': f'{dy*100:.2f}', 'eps': f'{eps:.2f}' if eps else '0',
        'mcap_cr': f'{mcap/1e7:.0f}', 'op_margin_pct': f'{op_margin*100:.1f}',
        'pb': f'{pb:.1f}', 'rsi': f'{rsi:.0f}',
        'mom_1w': f'{mom_1w:.1f}', 'mom_1m': f'{mom_1m:.1f}',
        'pct_ma20': f'{pct_ma20:.1f}', 'pct_ma50': f'{pct_ma50:.1f}',
        'volatility': f'{volatility:.0f}',
        'high52': info.get('fiftyTwoWeekHigh', 0),
        'low52': info.get('fiftyTwoWeekLow', 0),
    })
    
    ai_insight = ''
    if ai_result:
        verdict_type = ai_result.get('verdict', verdict_type)
        verdict_title = ai_result.get('verdict_title', verdict_title)
        verdict_text = ai_result.get('verdict_text', verdict_text)
        reasons_pos = ai_result.get('reasons_pos', reasons_pos)
        reasons_neg = ai_result.get('reasons_neg', reasons_neg)
        ai_insight = ai_result.get('ai_insight', '')
    
    # Fetch news
    news_list = []
    try:
        raw_news = (_yf_rate_limit(), t.news or [])[1]
        for item in raw_news[:5]:
            title = item.get('title', '')
            link = item.get('link', '')
            publisher = item.get('publisher', '')
            if title:
                news_list.append({'title': title, 'link': link, 'source': publisher})
    except:
        pass
    
    if not news_list:
        news_list.append({'title': f'{name} — check moneycontrol.com or economictimes for latest news', 'link': '', 'source': ''})

    return {
        'name': name, 'symbol': symbol, 'price': price,
        'sector': sector, 'industry': industry,
        'ai_enabled': AI_ENABLED,
        'ai_insight': ai_insight,
        'changePercent': info.get('regularMarketChangePercent', 0) or 0,
        'verdict': {'type': verdict_type, 'title': verdict_title, 'text': verdict_text},
        'metrics': [
            {'label': 'Score', 'value': f'{score}/100', 'status': 'good' if score>=50 else ('neutral' if score>=25 else 'bad')},
            {'label': 'P/E', 'value': f'{pe:.1f}x' if pe>0 else 'Loss', 'status': 'good' if 0<pe<20 else ('bad' if pe<=0 or pe>40 else 'neutral')},
            {'label': 'ROE', 'value': f'{roe*100:.1f}%', 'status': 'good' if roe>0.2 else ('bad' if roe<0.05 else 'neutral')},
            {'label': 'Profit Margin', 'value': f'{pm*100:.1f}%', 'status': 'good' if pm>0.2 else ('bad' if pm<0.05 else 'neutral')},
            {'label': 'RSI', 'value': f'{rsi:.0f}', 'status': 'good' if rsi<40 else ('bad' if rsi>70 else 'neutral')},
            {'label': 'Momentum', 'value': f'{mom_1w:+.1f}%', 'status': 'good' if mom_1w>2 else ('bad' if mom_1w<-3 else 'neutral')},
        ],
        'details': [
            {'label': 'Revenue Growth', 'value': f'{rev_g*100:.1f}%'},
            {'label': 'Debt/Equity', 'value': f'{de:.1f}' if de else '0 (Debt free)'},
            {'label': 'Dividend Yield', 'value': f'{dy*100:.2f}%'},
            {'label': 'EPS', 'value': f'₹{eps:.2f}'},
            {'label': 'P/B', 'value': f'{pb:.1f}x'},
            {'label': 'Mkt Cap', 'value': f'₹{mcap/1e7:.0f}Cr'},
            {'label': 'Operating Margin', 'value': f'{op_margin*100:.1f}%'},
            {'label': 'Forward P/E', 'value': f'{fwd_pe:.1f}x' if fwd_pe else '-'},
            {'label': 'Volatility', 'value': f'{volatility:.0f}%'},
            {'label': 'MA20 Distance', 'value': f'{pct_ma20:+.1f}%'},
            {'label': 'MA50 Distance', 'value': f'{pct_ma50:+.1f}%'},
            {'label': 'Volume Ratio', 'value': f'{vol_ratio:.1f}x'},
            {'label': '52W High', 'value': f'₹{info.get("fiftyTwoWeekHigh",0):.2f}'},
            {'label': '52W Low', 'value': f'₹{info.get("fiftyTwoWeekLow",0):.2f}'},
        ],
        'reasons_pos': reasons_pos,
        'reasons_neg': reasons_neg,
        'news': news_list,
        'score': score,
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'error': 'कृपया कोई stock name लिखें'})
    
    symbol = resolve_symbol(q)
    if not symbol:
        return jsonify({'error': f'"{q}" का symbol नहीं मिला। कोई और नाम try करें।'})
    
    cached = get_cached(symbol)
    if cached:
        return jsonify(cached)
    
    try:
        result = analyze_stock(symbol)
        set_cache(symbol, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f'  Stock Advisor: http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)
