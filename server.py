"""Election Night Lab: same-origin turnout relay + single-file HTML host."""
import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parent
MAKO = "https://www.mako.co.il/finances-news/Article-b141e24aebc71a1026.htm"
PATTERNS = [
    re.compile(r"(?:שיעור|אחוזי?)\s+ההצבעה.{0,90}?(?:עד|נכון\s+לשעה)\s*(\d{1,2}:\d{2}).{0,110}?(?:עומד(?:ים)?\s+על|הגיע(?:ו)?\s+ל|הינו|הם)\s*(\d{1,2}(?:[.,]\d{1,3})?)\s*%"),
    re.compile(r"נכון\s+לשעה\s*(\d{1,2}:\d{2})\s*[-–:]\s*(\d{1,2}(?:[.,]\d{1,3})?)\s*%\s*הצביעו\s+בבחירות"),
    re.compile(r"נכון\s+לשעה\s*(\d{1,2}:\d{2}).{0,45}?אחוזי?\s+ההצבעה\s+(?:ברחבי\s+הארץ|הארצי(?:ים)?).{0,110}?(\d{1,2}(?:[.,]\d{1,3})?)\s*%"),
    re.compile(r"אחוז(?:י)?\s+ההצבעה\s+הארצי(?:ים)?.{0,55}?(?:ב|עד|נכון\s+לשעה)\s*(\d{1,2}:\d{2}).{0,85}?(\d{1,2}(?:[.,]\d{1,3})?)\s*%"),
]

def extract_turnout(raw):
    clean = re.sub(r"(?is)<script\b.*?</script>|<style\b.*?</style>", " ", raw)
    clean = re.sub(r"(?s)<[^>]*>", " ", clean)
    clean = re.sub(r"\s+", " ", html.unescape(clean))
    found = []
    for pattern in PATTERNS:
        for match in pattern.finditer(clean):
            hh, mm = map(int, match.group(1).split(":"))
            pct = float(match.group(2).replace(",", "."))
            near = clean[max(0, match.start()-55):match.end()+20]
            if 0 <= hh < 24 and 0 <= mm < 60 and 0 < pct <= 100 and not re.search(r'צה["״\x27]?ל|משרתי\s+צה', near):
                found.append((hh*60+mm, pct, f"{hh:02d}:{mm:02d}"))
    if not found:
        raise ValueError("No timestamped nationwide turnout figure found on Mako")
    _, pct, time_of_report = max(found)
    return {"turnout_pct": pct, "as_of": time_of_report, "source": MAKO, "fetched_at": datetime.now(timezone.utc).isoformat()}


# A second, independent news mirror prevents a static Mako article from
# silently being presented as the latest election turnout bulletin.
SOURCES = [
    ("Mako", MAKO),
    ("Ynet election liveblog", "https://www.ynet.co.il/news/blogs/article/hkbbhw6nzg"),
    ("Kan election updates", "https://www.kan.org.il/umbraco/surface/WhatsHappeningSurface/GetNews"),
]
ADDITIONAL_PATTERNS = [
    # e.g. "אחוז ההצבעה עד השעה 12:00 עומד על 31.5%"
    (re.compile(r"(?:שיעור|אחוז(?:י)?)\s*ההצבעה(?:\s*הארצי(?:ים)?)?.{0,80}?(?:עד|נכון\s*ל)\s*(?:ה?שעה)\s*(\d{1,2}:\d{2}).{0,85}?(\d{1,2}(?:[.,]\d{1,2})?)\s*%"), False),
    # e.g. "31.5% הצביעו עד השעה 12:00"
    (re.compile(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*%.{0,90}?(?:הצביעו|מבעלי\s*זכות\s*הבחירה).{0,90}?(?:עד|נכון\s*ל)\s*(?:ה?שעה)\s*(\d{1,2}:\d{2})"), True),
    (re.compile(r"(?:עד|נכון\s*ל)\s*(?:ה?שעה)\s*(\d{1,2}:\d{2}).{0,115}?שיעור\s*ההצבעה.{0,90}?(\d{1,2}(?:[.,]\d{1,2})?)\s*%"), False),
]

def extract_news(raw):
    clean = re.sub(r"(?is)<script\b.*?</script>|<style\b.*?</style>", " ", raw)
    clean = re.sub(r"(?s)<[^>]*>", " ", clean)
    clean = re.sub(r"\s+", " ", html.unescape(clean))
    candidates = []
    for pattern, reversed_groups in ADDITIONAL_PATTERNS:
        for m in pattern.finditer(clean):
            time_text = m.group(2) if reversed_groups else m.group(1)
            pct_text = m.group(1) if reversed_groups else m.group(2)
            try:
                h, minute = map(int, time_text.split(":"))
                pct = float(pct_text.replace(",", "."))
            except ValueError:
                continue
            context = clean[max(0,m.start()-85):min(len(clean),m.end()+25)]
            # Explicit exclusions for historical comparisons, army voting and locality reports.
            segment = m.group(0)
            if (not 0 <= h < 24 or not 0 <= minute < 60 or not 0 < pct <= 100
                or re.search(r'צה["״\x27]?ל|משרתי\s+צה', segment)
                or re.search(r'בבחירות\s+(?:הקודמות|לכנסת\s*ה.25)|ב[-־ ]?2022', segment)):
                continue
            if re.search(r'ירושלים|תל\s*אביב|חיפה|בני\s*ברק|ראשון\s*לציון', segment) and 'הארצי' not in segment:
                continue
            candidates.append((h*60+minute, pct, f"{h:02d}:{minute:02d}"))
    if not candidates:
        raise ValueError("No expressly timestamped nationwide turnout observation")
    _, pct, when = max(candidates, key=lambda z: z[0])
    return {"turnout_pct":pct,"as_of":when}

def _fetch_single(name, url):
    req=urllib.request.Request(
        url + ("&" if "?" in url else "?") + "nocache=" + str(int(time.time())),
        headers={"User-Agent":"Mozilla/5.0 (compatible; ElectionNightLab/1.1)","Accept":"text/html, application/json","Accept-Language":"he-IL,he;q=0.9","Cache-Control":"no-cache"}
    )
    with urllib.request.urlopen(req,timeout=9) as response:
        raw=response.read(1900000).decode("utf-8",errors="replace")
    try:
        result=extract_news(raw)
    except ValueError:
        # Retain the tested, stricter Mako-specific parser as another route.
        if name!="Mako":
            raise
        result=extract_turnout(raw)
    result.update({"source":url,"source_name":name})
    return result

def fetch_turnout():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from zoneinfo import ZoneInfo
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        tasks = {pool.submit(_fetch_single,name,url):name for name,url in SOURCES}
        for future in as_completed(tasks):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"source":tasks[future],"error":str(exc)[:120]})
    if not results:
        raise ValueError("No independently timestamped nationwide turnout found from checked sources")
    selected=max(results,key=lambda r:(int(r["as_of"][:2])*60+int(r["as_of"][3:]),sum(x["as_of"]==r["as_of"] and x["turnout_pct"]==r["turnout_pct"] for x in results)))
    now=datetime.now(ZoneInfo("Asia/Jerusalem"))
    report_minutes=int(selected["as_of"][:2])*60+int(selected["as_of"][3:])
    checked_minutes=now.hour*60+now.minute
    selected["stale"]=now.date().isoformat()!="2026-10-27" or checked_minutes-report_minutes>=150
    selected["fetched_at"]=datetime.now(timezone.utc).isoformat()
    selected["sources_checked"]=[{"name":x["source_name"],"as_of":x["as_of"],"turnout_pct":x["turnout_pct"]} for x in results]
    selected["sources_unavailable"]=failures
    return selected

class Handler(BaseHTTPRequestHandler):
    def respond(self, status, kind, body):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path).path
        if route in ("/","/index.html"):
            self.respond(200,"text/html; charset=utf-8",(BASE/"index.html").read_bytes())
        elif route=="/healthz":
            self.respond(200,"application/json",b'{"ok":true}')
        elif route=="/api/turnout":
            try:
                self.respond(200,"application/json; charset=utf-8",json.dumps(fetch_turnout(),ensure_ascii=False).encode("utf-8"))
            except (TimeoutError, urllib.error.URLError, ValueError, OSError) as exc:
                self.respond(503,"application/json",json.dumps({"error":str(exc)[:180],"source":MAKO}).encode("utf-8"))
        else:
            self.respond(404,"text/plain",b"Not found")

if __name__=="__main__":
    port=int(os.environ.get("PORT","10000"))
    print(f"Election Night Lab listening on {port}",flush=True)
    ThreadingHTTPServer(("0.0.0.0",port), Handler).serve_forever()
