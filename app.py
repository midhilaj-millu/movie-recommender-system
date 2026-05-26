import streamlit as st
import pickle, requests, os, time, concurrent.futures
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "show_recs" not in st.session_state:
    st.session_state.show_recs = False
if "recs" not in st.session_state:
    st.session_state.recs = []
if "search_title" not in st.session_state:
    st.session_state.search_title = ""

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[data-testid="stApp"]{background:#09090e!important;color:#f0f0f0!important;font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background:#0c0c14!important;border-right:1px solid #1a1a28;}
#MainMenu,footer,header{visibility:hidden;}
.ctitle{font-family:'Bebas Neue',cursive;font-size:3rem;letter-spacing:4px;
  background:linear-gradient(135deg,#e50914 0%,#ff6b35 55%,#f5c518 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;}
.csub{color:#444;font-size:.74rem;letter-spacing:2.5px;text-transform:uppercase;
  margin-top:2px;margin-bottom:1rem;}
.mc{width:168px;border-radius:11px;overflow:hidden;background:#121219;
  box-shadow:0 4px 18px rgba(0,0,0,.55);
  transition:transform .28s cubic-bezier(.34,1.56,.64,1),box-shadow .28s ease;cursor:pointer;}
.mc:hover{transform:scale(1.07) translateY(-5px);
  box-shadow:0 18px 44px rgba(229,9,20,.28),0 6px 20px rgba(0,0,0,.7);}
.mc-img{position:relative;overflow:hidden;}
.mc-img img{width:168px;height:248px;object-fit:cover;display:block;transition:transform .3s ease;}
.mc:hover .mc-img img{transform:scale(1.05);}
.mc-ov{position:absolute;inset:0;
  background:linear-gradient(to top,rgba(0,0,0,.92) 0%,rgba(0,0,0,.3) 60%,transparent 100%);
  opacity:0;transition:opacity .25s ease;
  display:flex;flex-direction:column;justify-content:flex-end;padding:10px 9px;gap:4px;}
.mc:hover .mc-ov{opacity:1;}
.ov-genre{font-size:.63rem;background:#e50914;color:#fff;padding:2px 6px;border-radius:3px;
  width:fit-content;font-weight:700;}
.ov-title{font-size:.8rem;font-weight:700;color:#fff;line-height:1.3;}
.ov-meta{font-size:.68rem;color:#ccc;}
.ov-hint{font-size:.62rem;color:#888;font-style:italic;}
.mc-foot{padding:7px 9px;background:#121219;}
.mc-name{font-size:.76rem;font-weight:600;color:#eee;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;margin-bottom:3px;}
.mc-rat{background:linear-gradient(90deg,#e50914,#b00710);color:#fff;
  font-size:.69rem;font-weight:700;padding:2px 7px;border-radius:4px;display:inline-block;}
.mg{display:flex;flex-wrap:wrap;gap:16px;margin-top:.8rem;}
/* detail */
.detail-page{max-width:1100px;margin:0 auto;padding:0 32px 60px;}
/* Backdrop banner — only shown when available */
.detail-backdrop-wrap{position:relative;width:100%;height:300px;margin-bottom:0;overflow:hidden;border-radius:0 0 16px 16px;}
.detail-backdrop-img{width:100%;height:300px;object-fit:cover;object-position:center 25%;display:block;
  -webkit-mask-image:linear-gradient(to bottom,black 40%,transparent 100%);
  mask-image:linear-gradient(to bottom,black 40%,transparent 100%);}
/* Two-column layout: poster left, info right */
.detail-main{display:flex;gap:36px;margin-top:28px;align-items:flex-start;}
.detail-poster{flex-shrink:0;width:220px;}
.detail-poster img{width:220px;height:330px;object-fit:cover;border-radius:12px;
  box-shadow:0 12px 40px rgba(0,0,0,.75);display:block;}
.detail-info{flex:1;min-width:0;}
.detail-title{font-family:'Bebas Neue',cursive;font-size:2.8rem;letter-spacing:3px;
  color:#fff;line-height:1;margin-bottom:6px;}
.detail-tagline{font-size:.88rem;color:#e50914;font-style:italic;margin-bottom:14px;}
.detail-chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;}
.chip{background:#1c1c2a;border:1px solid #2a2a3c;border-radius:20px;
  padding:4px 14px;font-size:.75rem;color:#bbb;}
.chip-red{background:rgba(229,9,20,.15);border:1px solid rgba(229,9,20,.3);
  border-radius:20px;padding:4px 14px;font-size:.75rem;color:#e50914;}
.chip-gold{background:rgba(245,197,24,.12);border:1px solid rgba(245,197,24,.25);
  border-radius:20px;padding:4px 14px;font-size:.75rem;color:#f5c518;}
/* Info cards stacked on right */
.detail-cards{display:flex;flex-direction:column;gap:10px;}
.detail-card{background:#111118;border:1px solid #1e1e2e;border-radius:10px;padding:13px 16px;}
.detail-card-label{font-size:.62rem;text-transform:uppercase;letter-spacing:1.8px;
  color:#444;margin-bottom:5px;font-weight:700;}
.detail-card-val{font-size:.86rem;color:#ddd;line-height:1.5;}
/* Overview full width below */
.detail-bottom{margin-top:28px;}
.detail-overview-label{font-size:.62rem;text-transform:uppercase;letter-spacing:1.8px;
  color:#444;font-weight:700;margin-bottom:8px;}
.detail-overview{font-size:.92rem;color:#bbb;line-height:1.75;
  background:#111118;border:1px solid #1e1e2e;border-radius:10px;padding:18px 20px;}
.detail-kw{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px;}
.kw-tag{background:#111118;border:1px solid #252535;border-radius:4px;
  padding:3px 10px;font-size:.7rem;color:#555;}
.detail-title{font-family:'Bebas Neue',cursive;font-size:3.2rem;letter-spacing:3px;
  color:#fff;line-height:1;margin-bottom:4px;}
.detail-tagline{font-size:.9rem;color:#e50914;font-style:italic;margin-bottom:14px;}
.detail-chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;}
.chip{background:#1c1c2a;border:1px solid #2a2a3c;border-radius:20px;
  padding:4px 13px;font-size:.75rem;color:#bbb;}
.chip-red{background:rgba(229,9,20,.15);border:1px solid rgba(229,9,20,.3);
  border-radius:20px;padding:4px 13px;font-size:.75rem;color:#e50914;}
.chip-gold{background:rgba(245,197,24,.12);border:1px solid rgba(245,197,24,.25);
  border-radius:20px;padding:4px 13px;font-size:.75rem;color:#f5c518;}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px;}
.detail-card{background:#111118;border:1px solid #1e1e2e;border-radius:10px;padding:14px 16px;}
.detail-card-label{font-size:.65rem;text-transform:uppercase;letter-spacing:1.5px;
  color:#555;margin-bottom:6px;font-weight:600;}
.detail-card-val{font-size:.88rem;color:#ddd;line-height:1.5;}
.detail-overview{font-size:.92rem;color:#bbb;line-height:1.7;
  background:#111118;border:1px solid #1e1e2e;border-radius:10px;padding:16px;}
.detail-kw{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px;}
.kw-tag{background:#111118;border:1px solid #252535;border-radius:4px;
  padding:3px 10px;font-size:.7rem;color:#666;}
.stButton>button{background:linear-gradient(135deg,#e50914,#b00710)!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  padding:.6rem 1.3rem!important;font-weight:700!important;font-size:.9rem!important;
  box-shadow:0 4px 14px rgba(229,9,20,.3)!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;
  box-shadow:0 8px 22px rgba(229,9,20,.5)!important;}
.sec-label{font-family:'Bebas Neue',cursive;font-size:1.4rem;letter-spacing:2px;
  color:#f0f0f0;margin:1rem 0 .3rem;}
.sidebar-lbl{font-size:.66rem;text-transform:uppercase;letter-spacing:2px;
  color:#444;font-weight:600;margin-bottom:5px;}
</style>
""", unsafe_allow_html=True)

# ── TMDB ─────────────────────────────────────────────────────────────────────
try:    TMDB_KEY = st.secrets["TMDB_API_KEY"]
except: TMDB_KEY = None
TMDB  = "https://api.themoviedb.org/3"
IMG   = "https://image.tmdb.org/t/p/w342"
IMG_L = "https://image.tmdb.org/t/p/w780"

@st.cache_data(show_spinner=False, ttl=86400)
def tmdb_poster(mid):
    if not TMDB_KEY: return None
    for a in range(3):
        try:
            r = requests.get(TMDB + "/movie/" + str(mid) + "?api_key=" + TMDB_KEY, timeout=6)
            r.raise_for_status()
            pp = r.json().get("poster_path")
            return (IMG + pp) if pp else None
        except requests.exceptions.ConnectionError: time.sleep(1.2 * (a + 1))
        except: break
    return None

@st.cache_data(show_spinner=False, ttl=86400)
def tmdb_backdrop(mid):
    if not TMDB_KEY: return None
    try:
        r = requests.get(TMDB + "/movie/" + str(mid) + "?api_key=" + TMDB_KEY, timeout=6)
        r.raise_for_status()
        bp = r.json().get("backdrop_path")
        return (IMG_L + bp) if bp else None
    except: return None

def fetch_posters(ids):
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        fs = {ex.submit(tmdb_poster, mid): mid for mid in ids}
        for f in concurrent.futures.as_completed(fs):
            out[fs[f]] = f.result()
    return out

# ── DATA ─────────────────────────────────────────────────────────────────────
MOOD_GENRES = {
    "Happy":        ["Comedy","Animation","Family"],
    "Dark":         ["Crime","Thriller","Horror","Mystery"],
    "Mind-Bending": ["Science Fiction","Mystery","Fantasy","Thriller"],
    "Romantic":     ["Romance","Drama"],
    "Emotional":    ["Drama","Family","History","Music"],
    "Sci-Fi":       ["Science Fiction","Adventure","Fantasy"],
    "Action":       ["Action","Adventure","War"],
    "Scary":        ["Horror","Thriller"],
}
MOOD_EMOJI = {
    "Happy":"😄","Dark":"🌑","Mind-Bending":"🧠",
    "Romantic":"💘","Emotional":"😢","Sci-Fi":"🚀","Action":"💥","Scary":"👻"
}

@st.cache_resource(show_spinner=False)
def load_data():
    pm, ps = "movies_new.pkl", "similarity_new.pkl"
    if os.path.exists(pm) and os.path.exists(ps):
        return pickle.load(open(pm, "rb")), pickle.load(open(ps, "rb"))
    with st.spinner("Building model (first run)..."):
        df = pd.read_csv("tmdb_movies_data.csv")
        cols = ['id','original_title','genres','keywords','cast','director',
                'overview','runtime','vote_average','release_year',
                'budget','revenue','production_companies','tagline']
        df = df[[c for c in cols if c in df.columns]].copy()
        df.dropna(subset=["overview"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        def make_tags(row):
            g = str(row.get('genres','')).replace('|',' ')
            k = str(row.get('keywords','')).replace('|',' ')
            c = ' '.join(str(row.get('cast','')).replace('|',' ').split()[:5])
            d = str(row.get('director','')).replace(' ','_')
            o = str(row.get('overview',''))
            return (g + " ") * 5 + (k + " ") * 4 + (d + " ") * 3 + (c + " ") * 2 + o
        df["tags"] = df.apply(make_tags, axis=1)
        tf  = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1,2))
        mat = tf.fit_transform(df["tags"])
        sim = cosine_similarity(mat, dense_output=False)
        pickle.dump(df,  open(pm, "wb"))
        pickle.dump(sim, open(ps, "wb"))
    return df, sim

movies_df, similarity = load_data()
all_titles = sorted(movies_df["original_title"].dropna().tolist())

# ── HELPERS ───────────────────────────────────────────────────────────────────
def safe(v, fb=""):
    s = str(v) if pd.notna(v) else fb
    return fb if s in ("nan","None","") else s

def fmt_rt(m):
    try:
        m = int(float(m)); h, r = divmod(m, 60)
        return str(h) + "h " + str(r) + "m" if h else str(r) + "m"
    except: return ""

def first_genre(g):
    sep = "|" if "|" in str(g) else ","
    parts = [p.strip() for p in str(g).split(sep)]
    return parts[0] if parts and parts[0] not in ("nan","") else "Movie"

def esc(s):
    return (str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                  .replace('"',"&quot;").replace("'","&#39;"))

def row_to_dict(row):
    cast_list = [c.strip() for c in str(row.get("cast","")).split("|") if c.strip()]
    kw_list   = [k.strip() for k in str(row.get("keywords","")).split("|") if k.strip()]
    co = safe(row.get("production_companies",""))
    return dict(
        id        = int(row["id"]),
        title     = safe(row["original_title"]),
        genre     = first_genre(safe(row.get("genres",""))),
        genres    = safe(row.get("genres","")).replace("|"," · "),
        year      = safe(row.get("release_year","")),
        rating    = safe(row.get("vote_average","")),
        runtime   = fmt_rt(row.get("runtime", 0)),
        director  = safe(row.get("director","")),
        cast      = cast_list,
        overview  = safe(row.get("overview","")),
        tagline   = safe(row.get("tagline","")),
        keywords  = kw_list,
        companies = co,
    )

def enrich(candidates, top_n, prefetch=40):
    batch = candidates[:prefetch]
    ids   = [int(movies_df.iloc[i]["id"]) for i, _ in batch]
    pm    = fetch_posters(ids)
    results = []
    for i, _ in batch:
        row    = movies_df.iloc[i]
        mid    = int(row["id"])
        poster = pm.get(mid)
        if not poster: continue
        d = row_to_dict(row)
        d["poster"] = poster
        results.append(d)
        if len(results) >= top_n: break
    return results

def recommend(title, top_n):
    try: idx = movies_df[movies_df["original_title"] == title].index[0]
    except: return []
    scores   = sorted(enumerate(similarity[idx].toarray().flatten()), key=lambda x: x[1], reverse=True)
    cands    = [(i, s) for i, s in scores[1:] if s >= 0.01]
    return enrich(cands, top_n, prefetch=max(top_n * 4, 40))

def mood_recs(mood, top_n):
    gl   = [g.lower() for g in MOOD_GENRES.get(mood, [])]
    if not gl: return []
    mask = movies_df["genres"].fillna("").str.lower().apply(lambda g: any(mg in g for mg in gl))
    filt = movies_df[mask].sort_values("vote_average", ascending=False)
    return enrich([(i, 1.0) for i in filt.index], top_n, prefetch=min(60, len(filt)))

# ── RENDER GRID ───────────────────────────────────────────────────────────────
def render_grid(items):
    if not items:
        st.info("No results — try a different selection."); return

    cols = st.columns(min(len(items), 5))
    for idx, m in enumerate(items):
        with cols[idx % 5]:
            yr  = ("📅 " + esc(str(m["year"]))) if m["year"] else ""
            rat = ("⭐ " + esc(str(m["rating"]))) if m["rating"] else ""
            rt  = ("🕐 " + esc(m["runtime"])) if m["runtime"] else ""
            meta_str = "  ".join(filter(None, [yr, rat, rt]))
            st.markdown(
                '<div class="mc">'
                '<div class="mc-img">'
                '<img src="' + esc(m["poster"]) + '" loading="lazy">'
                '<div class="mc-ov">'
                '<span class="ov-genre">' + esc(m["genre"]) + '</span>'
                '<div class="ov-title">' + esc(m["title"]) + '</div>'
                '<div class="ov-meta">' + meta_str + '</div>'
                '<div class="ov-hint">Click for full details</div>'
                '</div>'
                '</div>'
                '<div class="mc-foot">'
                '<div class="mc-name">' + esc(m["title"]) + '</div>'
                '<span class="mc-rat">⭐ ' + esc(str(m["rating"])) + '</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
            if st.button("ℹ Details", key="btn_" + str(m["id"]) + "_" + str(idx),
                         use_container_width=True):
                st.session_state.selected_movie = m
                st.rerun()

# ── FULL DETAIL VIEW ──────────────────────────────────────────────────────────
def show_detail(m):
    st.markdown("""
<style>[data-testid="stSidebar"]{display:none!important;}</style>
""", unsafe_allow_html=True)

    if st.button("← Back to results", key="close_detail"):
        st.session_state.selected_movie = None
        st.rerun()

    # Fetch backdrop (optional banner) — poster is always shown separately on left
    backdrop  = tmdb_backdrop(m["id"])
    poster_h  = esc(m.get("poster", ""))

    # Pre-build fragments
    title_h    = esc(m["title"])
    tagline_h  = ('<div class="detail-tagline">&ldquo;' + esc(m["tagline"]) + '&rdquo;</div>') if m["tagline"] else ""
    genre_h    = ('<span class="chip-red">' + esc(m["genre"]) + '</span>') if m["genre"] else ""
    rating_h   = ('<span class="chip-gold">&#11088; ' + esc(str(m["rating"])) + '</span>') if m["rating"] else ""
    year_h     = ('<span class="chip">&#128197; ' + esc(str(m["year"])) + '</span>') if m["year"] else ""
    rt_h       = ('<span class="chip">&#128336; ' + esc(m["runtime"]) + '</span>') if m["runtime"] else ""
    dir_h      = esc(m["director"]) if m["director"] else "&#8212;"
    cast_str   = ", ".join(m["cast"][:5])
    cast_h     = esc(cast_str) if cast_str else "&#8212;"
    genres_h   = esc(m["genres"]) if m["genres"] else "&#8212;"
    co         = m["companies"]
    co_short   = (co[:60] + "...") if len(co) > 60 else co
    co_h       = esc(co_short) if co_short else "&#8212;"
    overview_h = esc(m["overview"])
    kw_parts   = ["<span class='kw-tag'>" + esc(k) + "</span>" for k in m["keywords"]]
    kw_h       = ('<div class="detail-kw">' + "".join(kw_parts) + "</div>") if kw_parts else ""

    # Optional backdrop banner at the very top
    if backdrop:
        st.markdown(
            '<div class="detail-backdrop-wrap">'
            '<img src="' + esc(backdrop) + '" class="detail-backdrop-img">'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Main layout: poster LEFT  |  info RIGHT
    info_cards = (
        '<div class="detail-cards">'
        '<div class="detail-card"><div class="detail-card-label">Director</div>'
        '<div class="detail-card-val">' + dir_h + '</div></div>'
        '<div class="detail-card"><div class="detail-card-label">Cast</div>'
        '<div class="detail-card-val">' + cast_h + '</div></div>'
        '<div class="detail-card"><div class="detail-card-label">Genres</div>'
        '<div class="detail-card-val">' + genres_h + '</div></div>'
        '<div class="detail-card"><div class="detail-card-label">Production</div>'
        '<div class="detail-card-val">' + co_h + '</div></div>'
        '</div>'
    )

    st.markdown(
        '<div class="detail-page">'
        '<div class="detail-main">'
        # LEFT — poster only (one image)
        '<div class="detail-poster">'
        '<img src="' + poster_h + '" alt="' + title_h + '">'
        '</div>'
        # RIGHT — title, tagline, chips, info cards
        '<div class="detail-info">'
        '<div class="detail-title">' + title_h + '</div>'
        + tagline_h +
        '<div class="detail-chips">' + genre_h + rating_h + year_h + rt_h + '</div>'
        + info_cards +
        '</div>'
        '</div>'
        # BOTTOM — overview full width
        '<div class="detail-bottom">'
        '<div class="detail-overview-label">Overview</div>'
        '<div class="detail-overview">' + overview_h + '</div>'
        + kw_h +
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CineMatch")
    st.markdown("---")
    st.markdown('<div class="sidebar-lbl">Number of Recommendations</div>', unsafe_allow_html=True)
    top_n = st.slider("", 4, 16, 8, label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="sidebar-lbl">Mood-Based Discovery</div>', unsafe_allow_html=True)
    mood_opts = ["-- None --"] + [MOOD_EMOJI[m] + " " + m for m in MOOD_GENRES]
    sel_mood_raw = st.radio("", mood_opts, label_visibility="collapsed")
    sel_mood = None if sel_mood_raw == "-- None --" else sel_mood_raw.split(" ", 1)[1]
    st.markdown("---")
    st.caption("")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if st.session_state.selected_movie:
    show_detail(st.session_state.selected_movie)
    st.stop()

st.markdown('<div class="ctitle">CineMatch</div><div class="csub">Discover movies you\'ll love</div>',
            unsafe_allow_html=True)
st.markdown("---")

if sel_mood:
    st.markdown('<div class="sec-label">' + MOOD_EMOJI[sel_mood] + " " + sel_mood + ' picks</div>',
                unsafe_allow_html=True)
    with st.spinner("Loading..."):
        render_grid(mood_recs(sel_mood, top_n))
else:
    c1, c2 = st.columns([4, 1])
    with c1:
        pick = st.selectbox("Choose a movie you like:", [""] + all_titles, index=0,
                            format_func=lambda x: "Type or scroll to find a movie..." if x == "" else x)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Recommend", use_container_width=True)

    if go and pick:
        st.session_state.search_title = pick
        with st.spinner("Finding matches..."):
            st.session_state.recs = recommend(pick, top_n)
        st.session_state.show_recs = True

    if st.session_state.show_recs and st.session_state.recs:
        st.markdown('<div class="sec-label">Because you liked ' + st.session_state.search_title + '...</div>',
                    unsafe_allow_html=True)
        render_grid(st.session_state.recs)
    elif go and not pick:
        st.warning("Please select a movie first.")