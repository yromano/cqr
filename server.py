"""This deployment was withdrawn at the owner's request. No application is served."""
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

class RemovedSite(BaseHTTPRequestHandler):
    def _respond(self):
        path=urlsplit(self.path).path
        if path=="/healthz":
            self.send_response(200)
            self.send_header("Content-Type","text/plain; charset=utf-8")
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            if self.command!="HEAD":
                self.wfile.write(b"ok")
            return
        self.send_response(410, "Gone")
        self.send_header("Cache-Control","no-store, max-age=0")
        self.send_header("Content-Type","text/plain; charset=utf-8")
        self.send_header("Content-Length","0")
        self.end_headers()

    do_GET = _respond
    do_HEAD = _respond
    do_POST = _respond
    do_OPTIONS = _respond

if __name__=="__main__":
    ThreadingHTTPServer(("0.0.0.0",int(os.environ.get("PORT","10000"))),RemovedSite).serve_forever()
