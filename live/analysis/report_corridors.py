"""
Step 7c — standalone HTML page from corridors_<date>.json + report_corridors_template.html.

Output: _output/tram_corridors_<date>.html  (self-contained; open in a browser or share as a file)
Usage:  py -3.11 live/analysis/report_corridors.py
"""

import ta_config as C


def main():
    data = (C.OUT / f"corridors_{C.DATE}.json").read_text(encoding="utf-8")
    tpl = (C.HERE / "report_corridors_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/{}", data.replace("</", "<\/"))
    out = C.OUT / f"tram_corridors_{C.DATE}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
