#!/usr/bin/env python3
"""Turn a proxy entry (local URL or share link) into a runnable Xray client config.

Reads the first non-comment, non-empty line of a config file:
  - socks5://host:port | socks5h://... | http://... | https://...
        -> no core needed, this is a local proxy URL
  - vmess://... vless://... trojan://... ss://...
        -> generates an Xray config exposing a local SOCKS inbound

Only the Python standard library is used.
"""
import argparse
import base64
import json
import sys

SHARE_PREFIXES = ("vmess://", "vless://", "trojan://", "ss://")


def read_entry(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            i_eq, i_scheme = t.find("="), t.find("://")
            if i_eq != -1 and (i_scheme == -1 or i_eq < i_scheme):
                t = t[i_eq + 1:].strip()
            return t
    raise SystemExit("no proxy entry found in config file")


def b64d(s):
    s = s.strip()
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s)
    except Exception:
        return base64.urlsafe_b64decode(s)


def parse_vmess(link):
    obj = json.loads(b64d(link[len("vmess://"):]).decode("utf-8"))
    o = {
        "address": str(obj.get("add", "")),
        "port": int(obj.get("port", 0)),
        "id": str(obj.get("id", "")),
        "aid": int(obj.get("aid", 0) or 0),
        "scy": str(obj.get("scy", "auto") or "auto"),
        "net": str(obj.get("net", "tcp") or "tcp"),
        "type": str(obj.get("type", "none") or "none"),
        "host": str(obj.get("host", "") or ""),
        "path": str(obj.get("path", "") or ""),
        "tls": str(obj.get("tls", "") or ""),
        "sni": str(obj.get("sni", "") or ""),
        "alpn": str(obj.get("alpn", "") or ""),
        "fp": str(obj.get("fp", "") or ""),
    }
    out = {
        "protocol": "vmess",
        "settings": {"vnext": [{
            "address": o["address"], "port": o["port"],
            "users": [{"id": o["id"], "alterId": o["aid"], "security": o["scy"]}],
        }]},
    }
    return out, stream_settings(o)


def stream_settings(o):
    net = o["net"]
    st = {"network": net}
    if o["tls"] == "tls":
        st["security"] = "tls"
        tls = {}
        if o["sni"]:
            tls["serverName"] = o["sni"]
        if o["alpn"]:
            tls["alpn"] = [a for a in o["alpn"].split(",") if a]
        if o["fp"]:
            tls["fingerprint"] = o["fp"]
        st["tlsSettings"] = tls

    if net == "ws":
        ws = {"path": o["path"] or "/"}
        if o["host"]:
            ws["headers"] = {"Host": o["host"]}
        st["wsSettings"] = ws
    elif net in ("h2", "http"):
        st["network"] = "h2"
        h = {"path": o["path"] or "/"}
        if o["host"]:
            h["host"] = [x for x in o["host"].split(",") if x]
        st["httpSettings"] = h
    elif net == "grpc":
        st["grpcSettings"] = {"serviceName": o["path"] or ""}
    elif net == "kcp":
        kcp = {
            "mtu": 1350, "tti": 50, "uplinkCapacity": 12, "downlinkCapacity": 100,
            "congestion": False, "readBufferSize": 2, "writeBufferSize": 2,
            "header": {"type": o["type"] or "none"},
        }
        if o["path"]:
            kcp["seed"] = o["path"]
        st["kcpSettings"] = kcp
    elif net in ("tcp", "raw"):
        if o["type"] not in ("", "none"):
            st["tcpSettings"] = {"header": {"type": o["type"]}}
    return st


def build(entry, socks_port):
    if entry.startswith(SHARE_PREFIXES):
        if not entry.startswith("vmess://"):
            raise SystemExit("only vmess:// share links are supported for now")
        outbound, stream = parse_vmess(entry)
        outbound["streamSettings"] = stream
    else:
        raise SystemExit("entry is a local proxy URL; no xray config needed")

    return {
        "log": {"loglevel": "warning"},
        "inbounds": [{
            "listen": "127.0.0.1", "port": socks_port, "protocol": "socks",
            "settings": {"udp": True, "auth": "noauth"},
        }],
        "outbounds": [outbound],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="network.cfg path")
    ap.add_argument("--out", required=True, help="xray json config output path")
    ap.add_argument("--socks-port", type=int, default=10888)
    args = ap.parse_args()

    entry = read_entry(args.config)
    if not entry.startswith(SHARE_PREFIXES):
        print("LOCAL " + entry)
        return
    cfg = build(entry, args.socks_port)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("SHARE socks5://127.0.0.1:%d" % args.socks_port)


if __name__ == "__main__":
    main()
