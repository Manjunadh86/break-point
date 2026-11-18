const $ = (sel) => document.querySelector(sel);
const fmt = (n, d = 1) => (n == null || Number.isNaN(n) ? "—" : Number(n).toFixed(d));
const pct = (n) => (n == null ? "—" : `${(n * 100).toFixed(1)}%`);

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.missing ? `Not found: ${body.missing.join(", ")}` : res.statusText);
  }
  return res.json();
}

function activate(name) {
  document.querySelectorAll("[data-panel]").forEach((el) => {
    el.classList.toggle("active", el.dataset.panel === name);
  });
  document.querySelectorAll("nav button").forEach((el) => {
    el.classList.toggle("active", el.dataset.go === name);
  });
}

document.querySelectorAll("nav button").forEach((btn) => {
  btn.addEventListener("click", () => activate(btn.dataset.go));
});

async function loadOverview() {
  const s = await api("/api/overview");
  $("#s-matches").textContent = s.matches.toLocaleString();
  $("#s-chall").textContent = s.challenger_matches.toLocaleString();
  $("#s-players").textContent = s.players.toLocaleString();
  $("#s-odds").textContent = s.closing_odds.toLocaleString();
  $("#s-latest").textContent = s.latest_match_date || "—";
}

function playerCard(p, side) {
  return `
    <div class="card player">
      <h2>${p.name}</h2>
      <div class="meta">${p.ioc || "—"} · ${p.hand || "?"} · Elo ${fmt(p.elo, 0)} · surface ${p.surface_elo ? fmt(p.surface_elo, 0) : "n/a"}</div>
      <div class="kpi">
        <div><b>${pct(p.win_pct)}</b><span>Last ${p.played} loaded</span></div>
        <div><b>${pct(p.challenger_win_pct)}</b><span>Challenger (${p.challenger_played})</span></div>
        <div><b>${pct(p.hold_pct)}</b><span>Hold (sample)</span></div>
        <div><b>${p.matches_last_14d}</b><span>Matches / 14d</span></div>
      </div>
      <table>
        <thead><tr><th>Date</th><th>Opp</th><th>Score</th></tr></thead>
        <tbody>
          ${(p.last5 || []).map((m) => `
            <tr>
              <td>${m.date || ""}</td>
              <td class="${m.won ? "won" : "lost"}">${m.won ? "W" : "L"} ${m.opponent}</td>
              <td>${m.score || ""} · ${m.surface || ""}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

$("#matchup-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const a = $("#player-a").value.trim();
  const b = $("#player-b").value.trim();
  const surface = $("#surface").value || "";
  const line_a = $("#line-a").value;
  const line_b = $("#line-b").value;
  const box = $("#matchup-out");
  box.innerHTML = `<p class="empty">Reading the tape…</p>`;
  try {
    const q = new URLSearchParams({ a, b });
    if (surface) q.set("surface", surface);
    if (line_a) q.set("line_a", line_a);
    if (line_b) q.set("line_b", line_b);
    const r = await api(`/api/matchup?${q}`);
    const pa = r.model.p_a;
    box.innerHTML = `
      <div class="vs">
        ${playerCard(r.a, "a")}
        <div class="card mid">
          <div class="note">Model prior</div>
          <div class="prob">${pct(pa)}</div>
          <div class="bar"><i style="width:${(pa * 100).toFixed(1)}%"></i></div>
          <div class="note">${r.a.name.split(" ").slice(-1)} vs ${r.b.name.split(" ").slice(-1)}</div>
          ${r.market ? `<div class="note">Market ${pct(r.market.prob_a)} · gap ${fmt((r.market.disagreement_a || 0) * 100, 1)} pts</div>` : ""}
          <div class="note">H2H ${r.h2h.a_wins}–${r.h2h.b_wins} (${r.h2h.played})</div>
        </div>
        ${playerCard(r.b, "b")}
      </div>
      <div class="grid-2">
        <div class="card">
          <h3>Head to head</h3>
          ${r.h2h.recent.length ? `
            <table>
              <thead><tr><th>Date</th><th>Winner</th><th>Score</th></tr></thead>
              <tbody>${r.h2h.recent.map((h) => `<tr><td>${h.date}</td><td>${h.winner}</td><td>${h.score} · ${h.surface}</td></tr>`).join("")}</tbody>
            </table>` : `<p class="empty">No meetings in the loaded years.</p>`}
        </div>
        <div class="card">
          <h3>How to read this</h3>
          <p class="note">${r.model.note}</p>
          ${r.market ? `<p class="note">${r.market.note}</p>` : `<p class="note">Paste the two decimal prices you see on the board to de-vig them and compare to the Elo prior.</p>`}
          ${r.inplay ? `<p class="note">If A takes set 1: ${pct(r.inplay.after_a_wins_set1)} · if B takes it: ${pct(r.inplay.after_b_wins_set1)} · at 1–1: ${pct(r.inplay.at_one_all)}. ${r.inplay.note}</p>` : ""}
          <div class="flags">${(r.flags || []).map((f) => `<div class="flag">${f}</div>`).join("")}</div>
        </div>
      </div>
    `;
  } catch (err) {
    box.innerHTML = `<p class="empty">${err.message}</p>`;
  }
});

async function loadChallenger() {
  const surface = $("#chall-surface").value;
  const q = surface ? `?surface=${encodeURIComponent(surface)}` : "";
  const [rows, ctx] = await Promise.all([api(`/api/challenger${q}`), api("/api/context")]);
  $("#chall-context").innerHTML = `
    <div class="kpi">
      <div><b>${pct(ctx.challenger.upset_by_rank)}</b><span>Challenger rank upsets</span></div>
      <div><b>${pct(ctx.atp.upset_by_rank)}</b><span>ATP rank upsets</span></div>
      <div><b>${pct(ctx.challenger.ret_rate)}</b><span>Challenger RET rate</span></div>
      <div><b>${(ctx.challenger.n || 0).toLocaleString()}</b><span>Challenger matches</span></div>
    </div>
    <p class="note">${ctx.read}</p>
  `;
  $("#chall-table").innerHTML = rows.map((m) => `
    <tr>
      <td>${m.tourney_date}</td>
      <td>${m.tourney_name}</td>
      <td>${m.surface || ""}</td>
      <td>${m.round || ""}</td>
      <td>${m.winner_name} <span class="note">(${m.winner_rank || "?"})</span></td>
      <td>${m.loser_name} <span class="note">(${m.loser_rank || "?"})</span></td>
      <td>${m.score || ""}</td>
      <td>${m.completion === "Completed" ? "" : m.completion}</td>
    </tr>
  `).join("");
}

$("#chall-surface").addEventListener("change", () => loadChallenger().catch(console.error));

async function loadMarket() {
  const m = await api("/api/market");
  if (!m.n) {
    $("#market-out").innerHTML = `<p class="empty">${m.note}</p>`;
    return;
  }
  $("#market-out").innerHTML = `
    <div class="kpi">
      <div><b>${m.n.toLocaleString()}</b><span>ATP closes</span></div>
      <div><b>${pct(m.avg_overround)}</b><span>Avg juice</span></div>
      <div><b>${pct(m.favorite_win_rate)}</b><span>Close favorite won</span></div>
    </div>
    <p class="note">${m.note}</p>
    <table>
      <thead><tr><th>Band</th><th>N</th><th>Implied</th><th>Actual fav win</th><th>Gap</th></tr></thead>
      <tbody>
        ${m.calibration.map((c) => `
          <tr>
            <td>${c.band}</td>
            <td>${c.n}</td>
            <td>${pct(c.avg_implied)}</td>
            <td>${pct(c.fav_win_rate)}</td>
            <td>${c.avg_implied == null ? "—" : fmt(((c.fav_win_rate - c.avg_implied) * 100), 1) + " pts"}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
  const snaps = await api("/api/snapshots");
  $("#snap-out").innerHTML = snaps.length
    ? snaps.map((s) => `
        <div class="card" style="margin-top:12px">
          <strong>${s.player_a} vs ${s.player_b}</strong>
          <div class="note">${s.tournament} · ${s.points.length} snapshots</div>
          <table>
            <thead><tr><th>When</th><th>Book</th><th>Player</th><th>Price</th></tr></thead>
            <tbody>
              ${s.points.flatMap((p) => p.books.slice(0, 4).map((b) => `
                <tr><td>${p.snapped_at}</td><td>${b.bookmaker}</td><td>${b.player}</td><td>${fmt(b.price, 2)}</td></tr>
              `)).join("")}
            </tbody>
          </table>
        </div>
      `).join("")
    : `<p class="empty">No live snapshots yet. Optional: set ODDS_API_KEY and run <code>python -m research snapshot</code> more than once to build a movement curve.</p>`;
}

async function loadWatch() {
  const rows = await api("/api/watchlist");
  $("#watch-table").innerHTML = rows.length
    ? rows.map((w) => `
        <tr>
          <td>${w.sport}</td>
          <td>${w.player_a}</td>
          <td>${w.player_b}</td>
          <td>${w.surface || ""}</td>
          <td>${w.line_a || ""} / ${w.line_b || ""}</td>
          <td>${w.notes || ""}</td>
          <td><button class="ghost" data-del="${w.id}">Remove</button></td>
        </tr>
      `).join("")
    : `<tr><td colspan="7" class="empty">Nothing pinned. Add a Challenger or SRL card you want to study.</td></tr>`;
  document.querySelectorAll("[data-del]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await fetch(`/api/watchlist/${btn.dataset.del}`, { method: "DELETE" });
      loadWatch();
    });
  });
}

$("#watch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = {
    sport: $("#w-sport").value,
    player_a: $("#w-a").value.trim(),
    player_b: $("#w-b").value.trim(),
    surface: $("#w-surface").value.trim(),
    tournament: $("#w-tour").value.trim(),
    notes: $("#w-notes").value.trim(),
    line_a: $("#w-line-a").value ? Number($("#w-line-a").value) : null,
    line_b: $("#w-line-b").value ? Number($("#w-line-b").value) : null,
  };
  await fetch("/api/watchlist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  e.target.reset();
  loadWatch();
});

function statLine(label, s) {
  if (!s || s.rate == null) return `<tr><td>${label}</td><td>—</td><td>0</td></tr>`;
  return `<tr><td>${label}</td><td>${pct(s.rate)}</td><td>${s.n.toLocaleString()}</td></tr>`;
}

async function loadLive() {
  const level = $("#live-level").value;
  const surface = $("#live-surface").value;
  const q = new URLSearchParams({ level });
  if (surface) q.set("surface", surface);
  const s = await api(`/api/live-study?${q}`);
  $("#live-kpis").innerHTML = `
    <div class="kpi">
      <div><b>${pct(s.set1_closeout.all.rate)}</b><span>Set-1 winner closes</span></div>
      <div><b>${pct(s.set1_closeout.tiebreak_first_set.rate)}</b><span>After a 7-6 first set</span></div>
      <div><b>${pct(s.set1_closeout.bagel_or_6_1.rate)}</b><span>After 6-0 / 6-1</span></div>
      <div><b>${pct(s.decider_favorite_wins.rate)}</b><span>Favorite wins a decider</span></div>
    </div>
    <p class="note">${s.clock.read}</p>
  `;
  $("#live-phases").innerHTML = s.phases.map((p) => `
    <div class="phase">
      <div class="clock">${p.clock}</div>
      <h3>${p.label}</h3>
      <p class="note">${p.why}</p>
    </div>
  `).join("");
  $("#live-closeout").innerHTML = `
    <h3>After set 1</h3>
    <table>
      <thead><tr><th>State</th><th>Rate</th><th>N</th></tr></thead>
      <tbody>
        ${statLine("Set-1 winner closes the match", s.set1_closeout.all)}
        ${statLine("Set-1 winner then loses (the fade)", s.after_set1.set1_winner_then_lost || s.comeback_after_losing_set1)}
        ${statLine("Tight first set (one- or two-game margin / TB)", s.set1_closeout.tight_first_set)}
        ${statLine("Tiebreak first set", s.set1_closeout.tiebreak_first_set)}
        ${statLine("6-0 or 6-1 first set", s.set1_closeout.bagel_or_6_1)}
        ${statLine("Favorite took set 1 and won", s.after_set1.favorite_won_set1_and_match)}
        ${statLine("Underdog took set 1 and won", s.after_set1.underdog_won_set1_and_match)}
        ${statLine("Favorite dropped set 1 and still won", s.after_set1.favorite_lost_set1_but_won_match)}
        ${statLine("Match goes 2–0", s.straight_sets_share)}
        ${statLine("Match reaches a decider", s.reaches_decider)}
        ${statLine("Pre-match favorite wins the decider", s.decider_favorite_wins)}
      </tbody>
    </table>
  `;
  const buckets = Object.entries(s.retirement_clock.buckets);
  $("#live-ret").innerHTML = `
    <h3>Retirement clock (${s.retirement_clock.n.toLocaleString()} RET)</h3>
    <p class="note">${s.retirement_clock.read}</p>
    <table>
      <thead><tr><th>When the match stopped</th><th>Count</th></tr></thead>
      <tbody>${buckets.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</tbody>
    </table>
  `;
  const eloRows = Object.entries(s.closeout_by_elo_gap).map(([k, v]) => statLine(k, v)).join("");
  const surfRows = Object.entries(s.closeout_by_surface).map(([k, v]) => statLine(k, v)).join("");
  $("#live-tables").innerHTML = `
    <div class="grid-2">
      <div>
        <h3>Set-1 close-out by Elo gap</h3>
        <table><thead><tr><th>Gap</th><th>Close-out</th><th>N</th></tr></thead><tbody>${eloRows}</tbody></table>
      </div>
      <div>
        <h3>By surface</h3>
        <table><thead><tr><th>Surface</th><th>Close-out</th><th>N</th></tr></thead><tbody>${surfRows}</tbody></table>
      </div>
    </div>
  `;
  $("#live-rules").innerHTML = s.how_to_use.map((x) => `<li>${x}</li>`).join("");
}

$("#live-level").addEventListener("change", () => loadLive().catch(console.error));
$("#live-surface").addEventListener("change", () => loadLive().catch(console.error));

$("#quote-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const [sa, sb] = $("#q-sets").value.split("-");
  const q = new URLSearchParams({
    a: $("#q-a").value.trim(),
    b: $("#q-b").value.trim(),
    sets_a: sa,
    sets_b: sb,
    level: $("#live-level").value,
  });
  if ($("#q-surface").value) q.set("surface", $("#q-surface").value);
  if ($("#q-set1").value.trim()) q.set("set1", $("#q-set1").value.trim());
  if ($("#q-line-a").value) q.set("line_a", $("#q-line-a").value);
  if ($("#q-line-b").value) q.set("line_b", $("#q-line-b").value);
  const box = $("#quote-out");
  box.innerHTML = `<p class="empty">Pricing the state…</p>`;
  try {
    const r = await api(`/api/live-quote?${q}`);
    box.innerHTML = `
      <div class="card" style="margin-top:16px">
        <div class="kpi">
          <div><b>${pct(r.pre_match.p_a)}</b><span>Pre-match Elo (A)</span></div>
          <div><b>${pct(r.model.iid_sets)}</b><span>IID-set update</span></div>
          <div><b>${pct(r.model.blended)}</b><span>Blended live prior (A)</span></div>
          <div><b>${r.market && r.market.prob_a != null ? pct(r.market.prob_a) : "—"}</b><span>Your live implied</span></div>
        </div>
        <p class="note"><strong>${r.state.label}.</strong> ${r.when}</p>
        ${r.market && r.market.gap_a != null ? `<p class="note">Gap vs your live number: ${fmt(r.market.gap_a * 100, 1)} pts (prior minus de-vigged). A couple of points is juice, not a play.</p>` : ""}
        <p class="note">${r.model.note}</p>
        <div class="flags">${r.player_notes.map((n) => `<div class="flag">${n}</div>`).join("")}</div>
      </div>
    `;
  } catch (err) {
    box.innerHTML = `<p class="empty">${err.message}</p>`;
  }
});

let lockedId = null;
let slateTimer = null;
let forecastTimer = null;

function renderForecast(r, board) {
  const implied = r.market && r.market.prob_a != null ? r.market.prob_a : null;
  const head = board
    ? `${board.p1} vs ${board.p2} · ${board.sets} sets · ${board.games} games · ${board.points}`
    : `${r.p1.name}${r.p1.underdog ? " · underdog" : ""} vs ${r.p2.name}`;
  $("#forecast-out").innerHTML = `
    <div class="card" style="margin-top:16px">
      <div class="note">${head}</div>
      <div class="hero-prob">${pct(r.blended)}</div>
      <div class="note">P1 wins the match from this score</div>
      <div class="steps">
        <div><b>${pct(r.next.p1_wins_point)}</b><span class="note">Next point</span></div>
        <div><b>${pct(r.next.p1_wins_game)}</b><span class="note">This game</span></div>
        <div><b>${pct(r.next.p1_wins_set)}</b><span class="note">This set</span></div>
        <div><b>${pct(r.next.p1_wins_match)}</b><span class="note">Match (blended)</span></div>
      </div>
      <div class="kpi" style="margin-top:14px">
        <div><b>${pct(r.pre_match)}</b><span>Pre-match Elo</span></div>
        <div><b>${pct(r.markov.match)}</b><span>Markov match</span></div>
        <div><b>${r.gbm == null ? "—" : pct(r.gbm)}</b><span>GBM (set-state)</span></div>
        <div><b>${implied == null ? "—" : pct(implied)}</b><span>Your live implied</span></div>
      </div>
      ${(r.story || []).map((s) => `<p class="note">${s}</p>`).join("")}
      <p class="note">If P1 wins this game → match ${pct(r.branches.if_p1_wins_game)}. If they lose it → ${pct(r.branches.if_p1_loses_game)}.</p>
    </div>
  `;
}

async function lockMatch(id) {
  lockedId = id;
  const box = $("#forecast-out");
  box.innerHTML = `<p class="empty">Locking the feed…</p>`;
  try {
    const r = await api(`/api/slate/${id}/forecast`);
    const b = r.board;
    $("#live-lock").style.display = "block";
    $("#live-lock").innerHTML = `
      <div class="note">${b.live ? '<span class="dot live"></span>LIVE' : b.status} · ${b.tournament}${b.round ? " · " + b.round : ""}</div>
      <h2 style="font-family:var(--serif);margin:6px 0">${b.p1} vs ${b.p2}</h2>
      <div class="meta">${b.sets} sets · ${b.games} games · ${b.points} · server ${b.server_known ? b.server : "not in feed (P1 assumed)"}</div>
      <div class="note">Updating from the public scoreboard. Point/server only when the feed has them.</div>
    `;
    renderForecast(r.forecast, b);
    document.querySelectorAll("#slate-table tr").forEach((tr) => {
      tr.classList.toggle("active", tr.dataset.id === id);
    });
  } catch (err) {
    box.innerHTML = `<p class="empty">${err.message}</p>`;
  }
}

async function loadSlate() {
  const data = await api("/api/slate");
  const challOnly = $("#slate-chall")?.checked;
  const filtered = challOnly ? data.matches.filter((m) => m.challenger) : data.matches;
  $("#slate-meta").textContent = challOnly
    ? `${filtered.filter((m) => m.live).length} live Challenger/125 · ${filtered.filter((m) => m.state === "pre").length} upcoming · ESPN rarely carries CH — empty means the feed has tour events only`
    : `${data.meta.live} live · ${data.meta.upcoming} upcoming · ${data.meta.source}`;
  const rows = filtered.slice(0, 40);
  $("#slate-table").innerHTML = rows.length
    ? rows.map((m) => `
        <tr data-id="${m.id}">
          <td>${m.live ? '<span class="dot live"></span>' : ""}</td>
          <td>${m.p1} vs ${m.p2}</td>
          <td>${m.tournament}${m.challenger ? " · CH" : ""}</td>
          <td>${m.sets} ${m.games}</td>
          <td>${m.status}</td>
        </tr>
      `).join("")
    : `<tr><td colspan="5" class="empty">${challOnly ? "No Challenger / WTA 125 singles on this public board. Uncheck the filter to see tour events, or wait for a CH week ESPN actually lists." : "No live or upcoming singles on the public ATP/WTA board right now."}</td></tr>`;
  document.querySelectorAll("#slate-table tr[data-id]").forEach((tr) => {
    tr.addEventListener("click", () => lockMatch(tr.dataset.id));
    if (lockedId && tr.dataset.id === lockedId) tr.classList.add("active");
  });
  if ($("#auto-follow").checked && !lockedId) {
    const live = rows.find((m) => m.live) || rows.find((m) => m.state === "pre");
    if (live) lockMatch(live.id);
  }
}

$("#auto-follow")?.addEventListener("change", () => {
  if ($("#auto-follow").checked) {
    lockedId = null;
    loadSlate().catch(console.error);
  }
});
$("#slate-chall")?.addEventListener("change", () => {
  lockedId = null;
  loadSlate().catch(console.error);
});

$("#forecast-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = new URLSearchParams({
    p1: $("#f-p1").value.trim(),
    p2: $("#f-p2").value.trim(),
    sets: $("#f-sets").value,
    games: $("#f-games").value.trim() || "0-0",
    points: $("#f-points").value,
    server: $("#f-server").value,
  });
  if ($("#f-surface").value) q.set("surface", $("#f-surface").value);
  if ($("#f-set1").value.trim()) q.set("set1", $("#f-set1").value.trim());
  if ($("#f-line-1").value) q.set("line_1", $("#f-line-1").value);
  if ($("#f-line-2").value) q.set("line_2", $("#f-line-2").value);
  try {
    const r = await api(`/api/forecast?${q}`);
    renderForecast(r);
  } catch (err) {
    $("#forecast-out").innerHTML = `<p class="empty">${err.message}</p>`;
  }
});

async function loadModel() {
  const m = await api("/api/model");
  if (!m.trained) {
    $("#model-card").innerHTML = `<p class="note">${m.hint}</p>`;
    return;
  }
  const u = m.underdog_down_set1 || {};
  $("#model-card").innerHTML = `
    <h3>Challenger state model (holdout 2025–2026)</h3>
    <div class="kpi">
      <div><b>${fmt(m.auc, 3)}</b><span>AUC</span></div>
      <div><b>${fmt(m.brier, 3)}</b><span>Brier</span></div>
      <div><b>${u.n ? pct(u.actual) : "—"}</b><span>Underdog down a set, actually won</span></div>
      <div><b>${u.n ? pct(u.pred) : "—"}</b><span>Model’s average on that slice</span></div>
    </div>
    <p class="note">${m.note} A CNN is the wrong tool: a scoreboard is a small discrete state, not an image.</p>
  `;
}

// ---------------------------------------------------------------------------------
// Point engine: /api/inplay
// ---------------------------------------------------------------------------------

const PATH_LABELS = {
  c1: "P1 wins 2–0", c2: "P2 wins 2–0", c3: "P1 drops set 1, wins 2–1",
  c4: "P2 drops set 1, wins 2–1", c5: "P1 takes set 1, wins 2–1", c6: "P2 takes set 1, wins 2–1",
};

function engineQuery() {
  const q = new URLSearchParams({
    p1: $("#e-p1").value.trim(),
    p2: $("#e-p2").value.trim(),
    tour: $("#e-tour").value,
    sets: $("#e-sets").value,
    games: $("#e-games").value.trim() || "0-0",
    points: $("#e-points").value.trim().toUpperCase() || "0-0",
    server: $("#e-server").value,
    tb: $("#e-tb").checked ? "true" : "false",
    streak: $("#e-streak").value || "0",
    rule: $("#e-rule").value,
    log: $("#e-log").checked ? "true" : "false",
  });
  const opt = {
    tier: $("#e-tier").value, event: $("#e-event").value.trim(), surface: $("#e-surface").value,
    serve_1: $("#e-sv1").value.trim(), serve_2: $("#e-sv2").value.trim(), prev_set: $("#e-prev").value.trim(),
    mto_1: $("#e-mto1").value, mto_2: $("#e-mto2").value, price_1: $("#e-price1").value, price_2: $("#e-price2").value,
  };
  Object.entries(opt).forEach(([k, v]) => { if (v) q.set(k, v); });
  return q;
}

function marketCell(side, m) {
  if (!m) return `<div><b>—</b><span>${side}: no price</span></div>`;
  const tone = m.action === "back" ? "good" : m.action === "lay" ? "warn" : "";
  return `<div class="${tone}"><b>${m.action.toUpperCase()}${m.stake_fraction ? ` ${(m.stake_fraction * 100).toFixed(1)}%` : ""}</b>
    <span>${side} @ ${fmt(m.price, 2)} · market ${pct(m.market_p)} · edge ${(m.edge_prob * 100).toFixed(1)}pp (needs ${(m.threshold_prob * 100).toFixed(0)})</span></div>`;
}

function renderEngine(r) {
  const p1 = r.p1, p2 = r.p2, st = r.state, ret = r.retirement, t = r.tier || {};
  const paths = Object.entries(r.paths || {}).filter(([, v]) => v > 0.0005).sort((a, b) => b[1] - a[1]);
  const flags = Object.entries(r.flags || {}).filter(([, v]) => v).map(([k]) => k.replace(/_/g, " "));
  const modelLine = r.model_loaded
    ? `calibrator <code>${r.model_group}</code>${r.model_group !== r.segment ? ` (fallback; no usable model for ${r.segment})` : ""}`
    : "no calibrator loaded — showing the engine's own number";
  return `
    <div class="card" style="margin-top:16px">
      <div class="row" style="justify-content:space-between;align-items:baseline">
        <h2 style="margin:0">${p1.name} <span class="note">vs</span> ${p2.name}</h2>
        <span class="tag">segment ${r.segment} · tier ${t.tier} via ${t.tier_source}${t.field_rank ? ` (field rank ~${fmt(t.field_rank, 0)})` : ""}</span>
      </div>
      <p class="note">Sets ${st.sets.join("–")}, games ${st.games.join("–")}, points ${st.points.join("–")}${st.in_tb ? " (TB)" : ""}, P${st.server} serving · ~${st.points_played_est} points played (${pct(st.progress_est)} of a typical match) · κ ${fmt(st.kappa, 0)} · ${modelLine}</p>
      <h3>P(${p1.name} wins), step by step</h3>
      <div class="kpi" style="grid-template-columns:repeat(5,1fr)">
        <div><b>${pct(r.p_elo)}</b><span>1 · pre-match prior (blended Elo)</span></div>
        <div><b>${pct(r.p_prior)}</b><span>2 · prior at this score (Markov)</span></div>
        <div><b>${pct(r.p_updated)}</b><span>3 · + live serve update</span></div>
        <div><b>${pct(r.p_model)}</b><span>4 · + segment calibrator</span></div>
        <div><b>${pct(r.p_settled)}</b><span>5 · + retirement, ${ret.settlement.rule.replace("_", " ")}</span></div>
      </div>
      <div class="grid-2">
        <div>
          <h3>Serve, prior → observed</h3>
          <table>
            <thead><tr><th></th><th>Elo</th><th>surface</th><th>serve prior</th><th>serve now</th><th>ret. 12m</th><th>P(retire)</th></tr></thead>
            <tbody>
              <tr><td>${p1.name}</td><td>${fmt(p1.elo, 0)}</td><td>${p1.surface_elo ? fmt(p1.surface_elo, 0) : "—"}</td><td>${pct(p1.serve_prior)}</td><td>${pct(p1.serve_live)}</td><td>${ret.player_1.retirements_12m}/${ret.player_1.matches_12m}</td><td>${pct(ret.player_1.p_retire)}</td></tr>
              <tr><td>${p2.name}</td><td>${fmt(p2.elo, 0)}</td><td>${p2.surface_elo ? fmt(p2.surface_elo, 0) : "—"}</td><td>${pct(p2.serve_prior)}</td><td>${pct(p2.serve_live)}</td><td>${ret.player_2.retirements_12m}/${ret.player_2.matches_12m}</td><td>${pct(ret.player_2.p_retire)}</td></tr>
            </tbody>
          </table>
          <p class="note">P(${p1.name} wins this set) ${pct(r.p_set)} · this point swings the match by ${(r.importance * 100).toFixed(1)}pp · ~${fmt(ret.expected_remaining_games, 1)} games left · fatigue: P1 ${p1.load.prev_games} games ${p1.load.rest_days}d ago, P2 ${p2.load.prev_games} games ${p2.load.rest_days}d ago.</p>
        </div>
        <div>
          <h3>How the match ends from here</h3>
          <table>
            <tbody>${paths.map(([k, v]) => `<tr><td>${k}</td><td>${PATH_LABELS[k]}</td><td style="text-align:right">${pct(v)}</td></tr>`).join("")}</tbody>
          </table>
          ${r.market ? `<h3 style="margin-top:14px">Your prices</h3><div class="kpi">${marketCell("P1", r.market.p1)}${marketCell("P2", r.market.p2)}</div>` : `<p class="note">Type the two prices to see edge, stake and the 3pp threshold.</p>`}
          ${flags.length ? `<div class="flags">${flags.map((f) => `<div class="flag">${f}</div>`).join("")}</div>` : ""}
          ${r.paper_log_id ? `<p class="note">Logged to paper_log #${r.paper_log_id}.</p>` : ""}
        </div>
      </div>
      <details style="margin-top:12px"><summary class="note">Raw JSON (paste this when you report a flaw)</summary><pre style="font-size:11px;white-space:pre-wrap">${JSON.stringify(r, null, 1)}</pre></details>
    </div>
  `;
}

async function loadEngineModel(segment) {
  const box = $("#engine-model");
  try {
    const m = await api(`/api/inplay/model?group=${encodeURIComponent(segment)}`);
    if (!m.trained) { box.innerHTML = `<p class="note">${m.hint}</p>`; return; }
    const prog = m.metrics.progress;
    const bins = Object.keys(prog);
    box.innerHTML = `
      <h3>How good is the ${segment} calibrator? (held-out 2016–17, log loss; 0.693 = coin flip, lower is better)</h3>
      <table>
        <thead><tr><th>% of match played</th>${bins.map((b) => `<th>${b}</th>`).join("")}</tr></thead>
        <tbody>
          <tr><td>pre-match Elo only</td>${bins.map((b) => `<td>${fmt(prog[b].elo_static.log_loss, 3)}</td>`).join("")}</tr>
          <tr><td>Markov from the score</td>${bins.map((b) => `<td>${fmt(prog[b].elo_markov.log_loss, 3)}</td>`).join("")}</tr>
          <tr><td>+ live serve update</td>${bins.map((b) => `<td>${fmt(prog[b].dyn.log_loss, 3)}</td>`).join("")}</tr>
          <tr><td><strong>+ calibrator</strong></td>${bins.map((b) => `<td><strong>${fmt(prog[b].gbm.log_loss, 3)}</strong></td>`).join("")}</tr>
        </tbody>
      </table>
      <p class="note">${m.n_matches.test.toLocaleString()} test matches, ${m.n_test_rows.toLocaleString()} points. Early in a match the number is the prior and cannot be sharper than the ratings; it tightens from mid-match as serve points accumulate.</p>
    `;
  } catch (err) {
    box.innerHTML = `<p class="note">${err.message}</p>`;
  }
}

async function priceEngine(allowLog = true) {
  const box = $("#engine-out");
  box.innerHTML = `<p class="empty">Running the engine…</p>`;
  try {
    const q = engineQuery();
    if (!allowLog) q.set("log", "false");
    const res = await fetch(`/api/inplay?${q}`);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail ? JSON.stringify(body.detail) : res.statusText);
    box.innerHTML = renderEngine(body);
    loadEngineModel(body.segment);
    if (body.paper_log_id) loadPaper().catch(() => {});
    return body;
  } catch (err) {
    box.innerHTML = `<p class="empty">${err.message}</p>`;
    return null;
  }
}

$("#engine-form").addEventListener("submit", (e) => {
  e.preventDefault();
  priceEngine(true);
});

// ---- live matches -> form -------------------------------------------------------------
const liveState = { rows: [], follow: null, sig: null, timer: null, priced: false };

function matchSig(m) {
  return `${m.sets}|${m.games}|${m.server}|${m.state}`;
}

function fillEngineForm(m) {
  $("#e-p1").value = m.p1;
  $("#e-p2").value = m.p2;
  $("#e-tour").value = m.tour;
  $("#e-tier").value = m.tier_hint === "big" ? "big" : "";
  $("#e-event").value = m.event || "";
  if (m.surface) $("#e-surface").value = m.surface;
  if ([...$("#e-sets").options].some((o) => o.value === m.sets)) $("#e-sets").value = m.sets;
  $("#e-games").value = m.games;
  $("#e-points").value = m.points_known ? m.points : "0-0";
  $("#e-server").value = m.server === "p2" ? "2" : "1";
  $("#e-tb").checked = !!m.tiebreak;
  $("#e-prev").value = m.set1 || "";
}

function describeMatch(m) {
  const bits = [`${m.level}${m.qualifying ? " qualifying" : ""} · ${m.tournament}`];
  if (!m.server_known) bits.push("feed does not say who is serving: check the Server box");
  if (!m.surface) bits.push("surface unknown: pick it");
  if (!m.points_known) bits.push("point score and serve counts are yours to type");
  return bits.join(" · ");
}

async function loadLiveMatches() {
  const chall = $("#el-chall").checked;
  const pre = $("#el-pre").checked;
  try {
    const data = await api("/api/slate");
    liveState.rows = data.matches;
    $("#el-meta").textContent = `${data.meta.live} live · ${data.meta.upcoming} upcoming · ${data.meta.source}`;
    $("#el-note").textContent = data.meta.note;
    const rows = data.matches.filter((m) => (pre || m.live) && (!chall || m.challenger));
    $("#el-table").innerHTML = rows.map((m) => {
      const score = m.live ? `${m.per_set.length ? m.per_set.join(" ") : m.games}${m.tiebreak ? " TB" : ""}` : `starts ${m.start ? new Date(m.start).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—"}`;
      const srv = m.live ? (m.server_known ? (m.server === "p1" ? m.p1 : m.p2) : "?") : "—";
      const tag = m.live ? `<span class="pill" style="background:rgba(120,220,140,0.15)">live</span>` : "";
      return `<tr><td>${m.level}${m.qualifying ? " Q" : ""}<br><span class="note">${m.tournament}</span></td><td>${m.p1} v ${m.p2} ${tag}</td><td>${score}</td><td>${srv}</td><td>${m.surface || "?"}</td><td><button type="button" class="mini" data-load="${m.id}">Load</button></td></tr>`;
    }).join("") || `<tr><td colspan="6" class="note">Nothing ${pre ? "scheduled" : "live"} that matches the filter. ${chall ? "Untick “Challenger / 125 only” for the main tours." : ""}</td></tr>`;
  } catch (err) {
    $("#el-meta").textContent = `feed error: ${err.message}`;
  }
}

function stopFollowing() {
  if (liveState.timer) clearInterval(liveState.timer);
  liveState.timer = null;
  liveState.follow = null;
  $("#el-follow").style.display = "none";
}

async function pollFollowed() {
  if (!liveState.follow) return;
  try {
    const m = await api(`/api/slate/${liveState.follow}`);
    const sig = matchSig(m);
    const changed = sig !== liveState.sig;
    liveState.sig = sig;
    $("#el-follow-text").innerHTML = `Following <b>${m.p1} v ${m.p2}</b> — ${m.per_set.join(" ") || m.games} · ${m.server_known ? (m.server === "p1" ? m.p1 : m.p2) + " serving" : "server unknown"} · checked ${new Date().toLocaleTimeString()}`;
    if (m.state === "post") {
      $("#el-follow-text").innerHTML += " · <b>match over</b> — settle your logged quotes below";
      stopFollowing();
      return;
    }
    if (changed) {
      const keepPts = $("#e-points").value;
      fillEngineForm(m);
      if (!m.points_known && keepPts && m.games === $("#e-games").value) $("#e-points").value = keepPts;
      if ($("#engine-out").innerHTML.trim()) priceEngine(false);
    }
  } catch (err) {
    $("#el-follow-text").textContent = `poll failed: ${err.message}`;
  }
}

$("#el-table").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-load]");
  if (!btn) return;
  const m = liveState.rows.find((r) => r.id === btn.dataset.load);
  if (!m) return;
  stopFollowing();
  // a different match: what you counted or priced for the last one no longer applies
  ["#e-sv1", "#e-sv2", "#e-price1", "#e-price2", "#e-mto1", "#e-mto2"].forEach((s) => { $(s).value = ""; });
  $("#e-streak").value = "0";
  fillEngineForm(m);
  $("#el-note").textContent = describeMatch(m);
  if (m.live) {
    liveState.follow = m.id;
    liveState.sig = matchSig(m);
    $("#el-follow").style.display = "flex";
    $("#el-follow-text").innerHTML = `Following <b>${m.p1} v ${m.p2}</b> — re-prices when sets, games or server change (auto re-prices never write to the paper log)`;
    liveState.timer = setInterval(pollFollowed, 10000);
  }
  priceEngine(false);
  $("#engine-form").scrollIntoView({ behavior: "smooth", block: "start" });
});
$("#el-stop").addEventListener("click", stopFollowing);
$("#el-refresh").addEventListener("click", () => loadLiveMatches());
$("#el-chall").addEventListener("change", () => loadLiveMatches());
$("#el-pre").addEventListener("change", () => loadLiveMatches());

async function loadPaper() {
  const [rows, sum] = await Promise.all([api("/api/paper/rows?limit=40"), api("/api/paper")]);
  const s = sum.model
    ? `${sum.settled}/${sum.n} settled · model log loss ${fmt(sum.model.log_loss, 3)} vs prior ${sum.prior ? fmt(sum.prior.log_loss, 3) : "—"} vs market ${sum.market ? fmt(sum.market.log_loss, 3) : "—"} · flat P&L ${fmt(sum.flat_stake_units, 2)}u on ${sum.bets_settled} bets`
    : `${sum.n} logged, none settled yet`;
  $("#paper-summary").textContent = s;
  $("#paper-table").innerHTML = rows.map((r) => {
    const st = r.state || {};
    const state = `${(st.sets || []).join("–")} · ${(st.games || []).join("–")} · ${(st.points || []).join("–")}`;
    const res = r.winner
      ? `P${r.winner} won`
      : `<button class="mini" data-settle="${r.id}" data-w="1">P1 won</button> <button class="mini" data-settle="${r.id}" data-w="2">P2 won</button>`;
    return `<tr><td>${r.id}</td><td>${r.logged_at.slice(0, 16).replace("T", " ")}</td><td>${r.p1} v ${r.p2}</td><td>${r.segment || r.tour || ""}</td><td>${state}</td><td>${pct(r.p_model)}</td><td>${r.price ? fmt(r.price, 2) : "—"}</td><td>${res}</td></tr>`;
  }).join("") || `<tr><td colspan="8" class="note">Nothing logged yet.</td></tr>`;
}

$("#paper-table").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-settle]");
  if (!btn) return;
  const close = prompt("Closing price on P1 (optional, for CLV). Leave blank to skip.");
  const body = { winner: Number(btn.dataset.w) };
  if (close && Number(close) > 1) body.closing_price = Number(close);
  const res = await fetch(`/api/paper/${btn.dataset.settle}/settle`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) { alert(`Settle failed: ${res.statusText}`); return; }
  loadPaper().catch(() => {});
});

loadOverview().catch(() => {});
loadModel().catch(() => {});
loadEngineModel("CH_SMALL").catch(() => {});
loadPaper().catch(() => {});
loadLiveMatches();
setInterval(loadLiveMatches, 60000);
loadSlate().catch(() => {});
loadLive().catch(() => {});
loadChallenger().catch(() => {});
loadMarket().catch(() => {});
loadWatch().catch(() => {});
setInterval(() => loadSlate().catch(() => {}), 12000);
setInterval(() => { if (lockedId) lockMatch(lockedId).catch(() => {}); }, 6000);
