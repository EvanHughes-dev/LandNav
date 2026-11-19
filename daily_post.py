"""
Daily Instagram workflow for the land-navigation puzzles.

Commands
--------
    python daily_post.py generate              # build today's puzzle + prep 4:5 IG JPEGs
    python daily_post.py publish               # publish carousel to Instagram
    python daily_post.py all                   # generate + publish in one step
    python daily_post.py status                # inspect today's post spec and status
    python daily_post.py all --dry-run         # preview caption & slides without posting

Carousel structure
------------------
- Slide 1: TODAY's new puzzle (`ig_puzzle.jpg`)
- Slide 2: YESTERDAY's annotated solution (`ig_solution.jpg`, when available)
- Caption: Yesterday's answer summary at the top + today's puzzle instructions & bearings

Instagram Graph API requires publicly reachable JPEGs with aspect ratio between
4:5 and 1.91:1. Puzzle PNGs are letterboxed onto a 1080x1350 (4:5) white canvas.

Environment Variables (required for `publish` without `--dry-run`)
------------------------------------------------------------------
IG_USER_ID        Instagram Professional (Business/Creator) Account ID
IG_ACCESS_TOKEN   Long-lived access token with `instagram_content_publish`
IMAGE_BASE_URL    Public base URL serving the repo root, e.g.:
                  https://raw.githubusercontent.com/<owner>/<repo>/<branch>
GRAPH_VERSION     Optional Graph API version (default: v21.0)
"""
import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import land_nav_puzzle as nav
except ImportError:
    import nav_puzzles as nav  # fallback if module was renamed

ARCHIVE = nav.ARCHIVE_DIR
CANVAS = (1080, 1350)  # 4:5 Instagram portrait aspect ratio


def to_instagram_jpeg(src: Path, dst: Path):
    """Letterbox a PNG onto a 1080x1350 (4:5) white canvas and save as high-quality JPEG."""
    img = Image.open(src).convert("RGB")
    img.thumbnail(CANVAS, Image.LANCZOS)
    canvas = Image.new("RGB", CANVAS, "white")
    canvas.paste(img, ((CANVAS[0] - img.width) // 2, (CANVAS[1] - img.height) // 2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, "JPEG", quality=92, optimize=True)


def prepare(day: date, force: bool = False, init_yesterday: bool = False) -> dict:
    """Generate today's puzzle, prep Instagram slides, and compose the caption."""
    if init_yesterday:
        prev_day = day - timedelta(days=1)
        if nav.load_meta(prev_day, ARCHIVE) is None:
            nav.generate_daily(prev_day, ARCHIVE, force=False)

    today = nav.generate_daily(day, ARCHIVE, force=force)
    yday = nav.find_previous_puzzle(day, ARCHIVE)

    tdir = nav.day_dir(day, ARCHIVE)
    puzzle_jpg = tdir / "ig_puzzle.jpg"
    to_instagram_jpeg(tdir / "puzzle.png", puzzle_jpg)
    slides = [nav._to_repo_rel(puzzle_jpg)]

    base_caption = (tdir / "caption.txt").read_text(encoding="utf-8")
    if yday:
        ydir = nav.day_dir(date.fromisoformat(yday["date"]), ARCHIVE)
        sol_jpg = ydir / "ig_solution.jpg"
        to_instagram_jpeg(ydir / "solution.png", sol_jpg)
        slides.append(nav._to_repo_rel(sol_jpg))
        head = (
            f"Yesterday's Answer (#{yday['number']}): {yday['answer_text']} "
            f"— swipe to Slide 2 for the full plotted solution.\n\n"
        )
    else:
        head = f"New Series — Puzzle #{today['number']}! Full plotted solution drops tomorrow.\n\n"

    caption = (head + base_caption)[:2200]
    spec = {
        "date": day.isoformat(),
        "number": today["number"],
        "slides": slides,
        "caption": caption,
        "has_solution": bool(yday),
        "yesterday_date": yday["date"] if yday else None,
        "yesterday_answer": yday["answer_text"] if yday else None,
    }
    (tdir / "post_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (tdir / "post_spec.txt").write_text(f"slides: {slides}\n\n{caption}\n", encoding="utf-8")
    return spec


def load_or_prepare_spec(day: date, force: bool = False, init_yesterday: bool = False) -> dict:
    spec_f = nav.day_dir(day, ARCHIVE) / "post_spec.json"
    if spec_f.exists() and not force:
        return json.loads(spec_f.read_text(encoding="utf-8"))
    return prepare(day, force=force, init_yesterday=init_yesterday)


# --------------------------------------------------------------------------- Instagram Graph API
def _api(method: str, path: str, **params):
    ver = os.environ.get("GRAPH_VERSION", "v21.0")
    params["access_token"] = os.environ["IG_ACCESS_TOKEN"]
    r = requests.request(
        method,
        f"https://graph.facebook.com/{ver}/{path}",
        data=params if method == "POST" else None,
        params=params if method == "GET" else None,
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"Graph API {path} failed ({r.status_code}): {r.text}")
    return r.json()


def _wait_ready(container_id: str, timeout: int = 180):
    start = time.time()
    while time.time() - start < timeout:
        info = _api("GET", container_id, fields="status_code,status")
        status = info.get("status_code")
        if status == "FINISHED":
            return
        if status in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Container {container_id} failed ({status}): {info.get('status')}")
        time.sleep(5)
    raise TimeoutError(f"Container {container_id} not ready after {timeout}s")


def _check_url(url: str, attempts: int = 6):
    """Verify image is publicly reachable, retrying to allow GitHub raw CDN propagation."""
    last_code = None
    for attempt in range(1, attempts + 1):
        try:
            r = requests.head(url, timeout=30, allow_redirects=True)
            last_code = r.status_code
            if r.status_code == 200:
                return
        except requests.RequestException as exc:
            last_code = str(exc)
        if attempt < attempts:
            wait = attempt * 5
            print(f"[publish] waiting {wait}s for CDN propagation ({last_code}): {url}")
            time.sleep(wait)
    raise RuntimeError(f"Image not publicly reachable ({last_code}): {url}")


def publish(spec: dict) -> str:
    user = os.environ["IG_USER_ID"]
    base = os.environ["IMAGE_BASE_URL"].rstrip("/")
    urls = [f"{base}/{s}" for s in spec["slides"]]
    for u in urls:
        _check_url(u)

    if len(urls) == 1:
        c = _api("POST", f"{user}/media", image_url=urls[0], caption=spec["caption"])["id"]
        _wait_ready(c)
        return _api("POST", f"{user}/media_publish", creation_id=c)["id"]

    children = []
    for u in urls:
        cid = _api("POST", f"{user}/media", image_url=u, is_carousel_item="true")["id"]
        children.append(cid)
    for cid in children:
        _wait_ready(cid)

    parent = _api(
        "POST",
        f"{user}/media",
        media_type="CAROUSEL",
        children=",".join(children),
        caption=spec["caption"],
    )["id"]
    _wait_ready(parent)
    return _api("POST", f"{user}/media_publish", creation_id=parent)["id"]


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Daily Instagram publisher for land-nav puzzles.")
    ap.add_argument("step", choices=["generate", "publish", "all", "status"])
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--dry-run", action="store_true", help="preview without calling Instagram API")
    ap.add_argument("--init-yesterday", action="store_true",
                    help="if yesterday has no puzzle yet, generate it so Slide 2 is included")
    ap.add_argument("--force", action="store_true", help="regenerate existing files for the date")
    args = ap.parse_args()

    day = date.fromisoformat(args.date) if args.date else date.today()

    if args.step == "status":
        print(nav.archive_status(ARCHIVE))
        meta = nav.load_meta(day, ARCHIVE)
        print(f"Meta ({day})      : {json.dumps(meta, indent=2) if meta else 'not generated yet'}")
        return

    if args.step in ("generate", "all"):
        spec = prepare(day, force=args.force, init_yesterday=args.init_yesterday)
        print(f"[generate] {spec['date']} (#{spec['number']}) slides={spec['slides']}")

    if args.step in ("publish", "all"):
        spec = load_or_prepare_spec(day, force=args.force, init_yesterday=args.init_yesterday)
        meta = nav.load_meta(day, ARCHIVE)
        if meta and meta.get("posted"):
            print(f"[publish] {day} already posted (media id {meta['posted']['id']}); skipping.")
            return
        if args.dry_run:
            print("[publish] DRY RUN")
            print("--- SLIDES ---")
            for s in spec["slides"]:
                print(f"  {s}")
            print("--- CAPTION ---")
            print(spec["caption"])
            return
        post_id = publish(spec)
        nav.mark_posted(day, post_id, ARCHIVE)
        print(f"[publish] posted {day} -> media id {post_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)