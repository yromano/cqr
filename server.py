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

def fetch_turnout():
    req = urllib.request.Request(
        MAKO+"?nocache="+str(int(time.time())),
        headers={"User-Agent":"Mozilla/5.0 (compatible; ElectionNightLab/1.0)","Accept":"text/html","Accept-Language":"he-IL,he;q=0.9,en;q=0.8","Cache-Control":"no-cache"}
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        return extract_turnout(resp.read(1750000).decode("utf-8", errors="replace"))

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
